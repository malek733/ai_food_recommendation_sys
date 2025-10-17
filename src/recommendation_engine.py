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
chroma_db_path = "./chroma_db"
vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings_model)

# Smart LLM for advanced recommendations
smart_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    api_key=os.getenv("GOOGLE_API_KEY"),
    request_timeout=120  # ⏰ timeout in seconds (increase if you have long tasks)
)

# =========================================================
# 2️⃣ Chat history class
# =========================================================

class ChatHistory:
    def __init__(self):
        self.messages = []

    def add_user_message(self, message: str):
        self.messages.append(HumanMessage(content=message))

    def add_ai_message(self, message: str):
        self.messages.append(AIMessage(content=message))

    def get_history(self):
        return self.messages


chat_history = ChatHistory()

# =========================================================
# 3️⃣ Core food recommendation logic
# =========================================================

def get_recommendations(query: str, history: ChatHistory):
    """Query the Chroma database to find the best matching food items."""
    try:
        docs = vectorstore.similarity_search(query, k=5)
        if not docs:
            return "No matching dishes found."

        recommendations = [doc.page_content for doc in docs]
        prompt = f"User asked: {query}\nHere are the top menu items:\n\n" + "\n".join(recommendations)

        response = smart_llm.invoke([
            SystemMessage(content="You are a helpful food recommendation assistant."),
            HumanMessage(content=prompt)
        ])
        return response.content

    except Exception as e:
        return f"Error during recommendation: {e}"

# =========================================================
# 4️⃣ Smart Planning (Agentic) Tools
# =========================================================

cart = []

def query_menu_from_supabase(name: str):
    """Search for menu items and pick the best match (case-insensitive, partial)."""
    try:
        res = supabase.table("menu_items").select("id, name, price, description").execute()
        items = res.data or []
        search_lower = name.lower()

        matches = []
        for item in items:
            # handle both dict and string names
            if isinstance(item.get("name"), dict):
                en_name = item["name"].get("en", "").lower()
                ar_name = item["name"].get("ar", "").lower()
                if search_lower in en_name or search_lower in ar_name:
                    matches.append(item)
            else:
                if search_lower in str(item.get("name", "")).lower():
                    matches.append(item)

        # 👇 This is where you add the Chroma fallback
        if not matches:
            # Fallback to Chroma search if Supabase has no match
            docs = vectorstore.similarity_search(name, k=1)
            if docs:
                return {
                    "success": True,
                    "message": f"Found via vector search: {docs[0].page_content}"
                }
            return {
                "success": False,
                "message": f"No items found for '{name}' in Supabase or Chroma."
            }

        # If found in Supabase, return the first match
        first = matches[0]
        return {
            "success": True,
            "item_id": first["id"],
            "name": first["name"],
            "price": first["price"],
            "description": first.get("description", {}),
            "message": f"Found item '{first['name'].get('en', first['name'])}' with ID {first['id']}"
        }

    except Exception as e:
        return {"success": False, "message": f"Error querying menu items: {e}"}




import re
import json

def add_to_cart(item_id: str = None, quantity: int = 1):
    """Add an item to the cart by item_id. Handles both structured and raw string inputs."""
    try:
        # Handle if agent passed a raw string like: 'item_id="uuid", quantity=1'
        if isinstance(item_id, str) and ("item_id" in item_id or "quantity" in item_id):
            # Try to parse it as JSON first
            try:
                data = json.loads(item_id)
                item_id = data.get("item_id", item_id)
                quantity = int(data.get("quantity", 1))
            except json.JSONDecodeError:
                # Fallback to regex parsing
                item_id_match = re.search(r'item_id="?([a-f0-9\-]+)"?', item_id)
                quantity_match = re.search(r'quantity=?(\d+)', item_id)
                if item_id_match:
                    item_id = item_id_match.group(1)
                if quantity_match:
                    quantity = int(quantity_match.group(1))

        # 🟢 Now item_id should be a clean UUID
        res = supabase.table("menu_items").select("id, name, price").eq("id", item_id).execute()
        if not res.data:
            return {"success": False, "message": f"Item with id {item_id} not found."}

        item = res.data[0]
        item_name = item["name"]["en"] if isinstance(item["name"], dict) and "en" in item["name"] else str(item["name"])

        cart.append({
            "id": item["id"],
            "name": item_name,
            "price": float(item["price"]),
            "quantity": quantity
        })
        return {"success": True, "message": f"Added {quantity} × {item_name} (QR{item['price']:.2f}) to cart."}

    except Exception as e:
        return {"success": False, "message": f"Error adding item to cart: {e}"}


def view_cart():
    if not cart:
        return "Your cart is empty."
    total = sum(c["price"] * c["quantity"] for c in cart)
    summary = "\n".join([f"- {c['quantity']} × {c['name']} (QR{c['price']:.2f})" for c in cart])
    return f"Your Cart:\n{summary}\n\nTotal: QR{total:.2f}"


def checkout():
    if not cart:
        return "Your cart is empty. Please add items first."
    total = sum(c["price"] * c["quantity"] for c in cart)
    items = [{"name": c["name"], "quantity": c["quantity"], "price": c["price"]} for c in cart]
    cart.clear()
    return f"Order placed successfully!\nTotal: QR{total:.2f}\nItems: {items}"

# =========================================================
# 5️⃣ Smart Agent setup
# =========================================================

tools = [
    Tool(name="QueryMenu", func=query_menu_from_supabase, description="Search for menu items in Supabase."),
    Tool(name="AddToCart", func=add_to_cart, description="Add item to cart by item_id and quantity."),
    Tool(name="ViewCart", func=view_cart, description="Show user's current cart."),
    Tool(name="Checkout", func=checkout, description="Checkout and confirm order."),
]

smart_agent = initialize_agent(
    tools,
    smart_llm,
    agent_type="zero-shot-react-description",
    verbose=True,
    max_iterations=15,  # increase to allow more reasoning steps
    handle_parsing_errors=True,
    max_execution_time=180  # optional: total timeout in seconds for whole reasoning chain
)

# =========================================================
# 6️⃣ Main interactive loop
# =========================================================

def main():
    print("Welcome to Smart Food Agent!")
    print("Ask about food, calories, or say things like:")
    print("  - add spicy tuna roll to my cart")
    print("  - view cart")
    print("  - checkout\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        chat_history.add_user_message(query)

        # Route to smart agent if it's an action
        if any(word in query.lower() for word in ["add", "order", "checkout", "cart"]):
            print("\nSmart Agent handling your request...")
            result = smart_agent.invoke(query)
            if isinstance(result, dict):
                result_str = result.get("message", str(result))
            else:
                result_str = str(result)
            print(result_str)
            chat_history.add_ai_message(result_str)
            continue

        # Otherwise, do normal food recommendation
        response = get_recommendations(query, chat_history)
        print(response)
        chat_history.add_ai_message(response)


# =========================================================
# 7️⃣ Run
# =========================================================

if __name__ == "__main__":
    main()
