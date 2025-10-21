import os
import json
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional

# =========================================================
# 1️⃣ Setup
# =========================================================

load_dotenv()

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize embeddings & vectorstore (for food recommendation search)
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db_path = "./src/embeddings/chroma_db"
vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings_model)

# Smart LLM for advanced recommendations
smart_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=120
)

# =========================================================
# 2️⃣ Conversation Memory
# =========================================================

conversation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="output",
    input_key="input"
)

def add_user_message_to_memory(message: str):
    conversation_memory.chat_memory.add_user_message(HumanMessage(content=message))

def add_ai_message_to_memory(message: str):
    conversation_memory.chat_memory.add_ai_message(AIMessage(content=message))

def get_conversation_history():
    return conversation_memory.chat_memory.messages

def clear_conversation_memory():
    conversation_memory.chat_memory.clear()

def save_conversation_memory(file_path: str = "conversation_memory.json"):
    try:
        memory_data = {
            "messages": [
                {"type": "human" if isinstance(msg, HumanMessage) else "ai", "content": msg.content}
                for msg in conversation_memory.chat_memory.messages
            ]
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        print(f"Conversation memory saved to {file_path}")
    except Exception as e:
        print(f"Error saving conversation memory: {e}")

def load_conversation_memory(file_path: str = "conversation_memory.json"):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            conversation_memory.chat_memory.clear()
            for msg in memory_data.get("messages", []):
                if msg["type"] == "human":
                    conversation_memory.chat_memory.add_user_message(msg["content"])
                else:
                    conversation_memory.chat_memory.add_ai_message(msg["content"])
            print(f"Conversation memory loaded from {file_path}")
        else:
            print(f"No conversation memory file found at {file_path}")
    except Exception as e:
        print(f"Error loading conversation memory: {e}")

# =========================================================
# 3️⃣ Output Parsing (Pydantic Models)
# =========================================================

class FoodRecommendation(BaseModel):
    name: str = Field(..., description="Name of the recommended food item")
    reason: Optional[str] = Field(None, description="Why this item was recommended")
    description: Optional[str] = Field(None, description="Brief summary of the dish")

class FoodRecommendationsOutput(BaseModel):
    recommendations: List[FoodRecommendation] = Field(..., description="List of recommended dishes")

recommendation_parser = PydanticOutputParser(pydantic_object=FoodRecommendationsOutput)

# =========================================================
# 4️⃣ Core Recommendation Logic
# =========================================================

import re

def get_recommendations(query: str):
    """Query the Chroma database and use the LLM to return structured recommendations."""
    try:
        recent_messages = get_conversation_history()
        context_messages = recent_messages[-4:] if len(recent_messages) > 4 else recent_messages

        # Get more results for better filtering
        docs = vectorstore.similarity_search(query, k=8)

        if not docs:
            return "No matching dishes found."

        # Extract better formatted recommendations with metadata
        recommendations = []
        for doc in docs[:5]:  # Limit to top 5
            metadata = doc.metadata
            name = metadata.get("name", "Unknown")
            description = metadata.get("description", "No description available")
            restaurant = metadata.get("restaurant_name", "Unknown restaurant")
            price = metadata.get("price", "N/A")
            category = metadata.get("category_name", "Unspecified")

            recommendations.append({
                "name": name,
                "description": description,
                "restaurant": restaurant,
                "price": price,
                "category": category
            })

        # Create a more user-friendly prompt
        menu_context = json.dumps(recommendations, indent=2)

        prompt = (
            f"You are a friendly food recommendation assistant.\n"
            f"Help users find delicious food based on their query.\n"
            f"Return a **valid JSON only** following this format:\n\n"
            f"{recommendation_parser.get_format_instructions()}\n\n"
            f"User query: {query}\n\n"
            f"Available menu items:\n{menu_context}\n\n"
            f"Guidelines:\n"
            f"- Recommend dishes that best match the user's request\n"
            f"- Include 2-3 relevant recommendations\n"
            f"- Provide specific reasons why each dish matches\n"
            f"- Keep descriptions brief but informative\n"
        )

        messages = [SystemMessage(content=prompt)]
        for msg in context_messages:
            messages.append(msg)

        response = smart_llm.invoke(messages)
        raw_output = response.content.strip()

        # Extract JSON if the model added text before/after it
        json_match = re.search(r'\{[\s\S]*\}', raw_output)
        if not json_match:
            return f"Error: Could not find valid JSON in model output:\n{raw_output}"
        json_str = json_match.group(0)

        # Parse structured output
        parsed = recommendation_parser.parse(json_str)

        # Create a more user-friendly display
        display = "\nRecommended Dishes:\n"
        for rec in parsed.recommendations:
            display += f"- {rec.name}"
            if rec.description:
                display += f": {rec.description}"
            else:
                display += ": No description."
            if rec.reason:
                display += f" -> {rec.reason}"
            display += "\n"

        return display

    except Exception as e:
        return f"Error during recommendation: {e}"


# =========================================================
# 5️⃣ Smart Planning (Agentic) Tools
# =========================================================

cart = []

def query_menu_from_supabase(name: str):
    try:
        res = supabase.table("menu_items").select("id, name, price, description").execute()
        items = res.data or []
        search_lower = name.lower()
        matches = [i for i in items if search_lower in str(i.get('name', '')).lower()]
        if not matches:
            return {"success": False, "message": f"No items found for '{name}'."}
        first = matches[0]
        return {
            "success": True,
            "item_id": first["id"],
            "name": first["name"],
            "price": first["price"],
            "description": first.get("description", {}),
            "message": f"Found '{first['name']}'"
        }
    except Exception as e:
        return {"success": False, "message": str(e)}

def add_to_cart(item_id=None, quantity=1):
    try:
        # Handle different input formats from the agent
        if isinstance(item_id, dict):
            if "item_id" in item_id:
                item_id = item_id["item_id"]
                quantity = item_id.get("quantity", 1)
            else:
                return {"success": False, "message": "Invalid item_id format provided."}

        # Ensure we have a valid item_id
        if not item_id:
            return {"success": False, "message": "No item ID provided."}

        # Convert string UUID to proper format if needed
        if isinstance(item_id, str):
            # Remove any quotes that might be in the string
            item_id = item_id.strip('"\'')

        # Query the database for the item
        res = supabase.table("menu_items").select("id, name, price").eq("id", item_id).execute()
        if not res.data:
            return {"success": False, "message": f"Item with ID '{item_id}' not found."}

        item = res.data[0]
        item_name = item["name"]["en"] if isinstance(item["name"], dict) else str(item["name"])
        cart.append({"id": item["id"], "name": item_name, "price": float(item["price"]), "quantity": quantity})
        return {"success": True, "message": f"Added {quantity} × {item_name} to cart."}
    except Exception as e:
        return {"success": False, "message": f"Error adding to cart: {e}"}

def view_cart(*args, **kwargs):
    if not cart:
        return "Cart is empty."
    total = sum(c["price"] * c["quantity"] for c in cart)
    items = "\n".join([f"- {c['quantity']} × {c['name']} (QR{c['price']:.2f})" for c in cart])
    return f"Your Cart:\n{items}\nTotal: QR{total:.2f}"

def checkout(*args, **kwargs):
    if not cart:
        return "Cart is empty."
    total = sum(c["price"] * c["quantity"] for c in cart)
    items = [{"name": c["name"], "quantity": c["quantity"], "price": c["price"]} for c in cart]
    cart.clear()
    return f"Order placed successfully!\nTotal: QR{total:.2f}\nItems: {items}"

# =========================================================
# 6️⃣ Agent Setup
# =========================================================

tools = [
    Tool(name="QueryMenu", func=query_menu_from_supabase, description="Search for menu items in Supabase."),
    Tool(name="AddToCart", func=add_to_cart, description="Add an item to the cart."),
    Tool(name="ViewCart", func=view_cart, description="View the current cart."),
    Tool(name="Checkout", func=checkout, description="Checkout and place the order."),
]

smart_agent = initialize_agent(
    tools=tools,
    llm=smart_llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    memory=conversation_memory
)

# =========================================================
# 7️⃣ Main Loop
# =========================================================

def main():
    print("Smart Food Agent - Welcome!")
    load_conversation_memory()
    while True:
        query = input("\nYou: ").strip()
        if query.lower() in ["exit", "quit"]:
            save_conversation_memory()
            print("Goodbye!")
            break

        if query.lower() == "save memory":
            save_conversation_memory(); continue
        elif query.lower() == "load memory":
            load_conversation_memory(); continue
        elif query.lower() == "clear memory":
            clear_conversation_memory(); print("Memory cleared."); continue

        add_user_message_to_memory(query)

        if any(word in query.lower() for word in ["add", "order", "checkout", "cart"]):
            print("\nSmart Agent handling your request...")
            try:
                result = smart_agent.invoke(query)
                response = result["output"] if isinstance(result, dict) else str(result)
                print(response)
                add_ai_message_to_memory(response)
            except Exception as e:
                print(f"Error: {e}")
            continue

        response = get_recommendations(query)
        print(response)
        add_ai_message_to_memory(response)

if __name__ == "__main__":
    main()
