import os
import json
import re
import time
import warnings
from dotenv import load_dotenv
from supabase import create_client, Client
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
from pydantic import BaseModel, Field
from typing import List, Optional

# Suppress deprecation warnings for cleaner output
warnings.filterwarnings("ignore", category=DeprecationWarning)

# =========================================================
# 1️⃣ Setup
# =========================================================
load_dotenv()

# Supabase connection
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Embeddings + Vectorstore
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db_path = "./src/embeddings/chroma_db"
vectorstore = Chroma(persist_directory=chroma_db_path, embedding_function=embeddings_model)

# Smart LLM
smart_llm = ChatGroq(
    model="moonshotai/kimi-k2-instruct-0905",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
    request_timeout=60  # Reduced to avoid conflicts with agent timeout
)

# =========================================================
# 2️⃣ Conversation Memory
# =========================================================
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    input_key="input"
)
# =========================================================
# 3️⃣ Pydantic Models
# =========================================================
class MenuItem(BaseModel):
    name: str
    restaurant: str
    price: float
    category: str
    description: Optional[str] = None
    spiciness: Optional[str] = None

class RecommendationResponse(BaseModel):
    matches: List[MenuItem]
    explanation: str
    suggested_filters: Optional[List[str]] = []

# =========================================================
# 4️⃣ Helper Functions
# =========================================================
def retry_llm_call(llm_call_func, max_retries=3, initial_delay=1):
    """Retry LLM calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return llm_call_func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            delay = initial_delay * (2 ** attempt)
            print(f"LLM call failed (attempt {attempt + 1}/{max_retries}), retrying in {delay}s...")
            time.sleep(delay)
    return None
def extract_metadata_filters(query):
    filters = {}
    query_lower = query.lower()
    restaurant_indicators = ["from", "at", "of", "in"]
    for indicator in restaurant_indicators:
        if f" {indicator} " in f" {query_lower} ":
            parts = query_lower.split(f" {indicator} ")
            if len(parts) > 1:
                filters['restaurant_name'] = parts[1].strip()
                break

    price_patterns = {
        "cheap": {"price": {"max": 15}},
        "expensive": {"price": {"min": 30}},
        "moderate": {"price": {"min": 15, "max": 30}},
        "budget": {"price": {"max": 20}},
        "high-end": {"price": {"min": 40}},
    }
    for pattern, price_filter in price_patterns.items():
        if pattern in query_lower:
            filters["price"] = price_filter["price"]
            break

    category_mapping = {
        "appetizer": "Appetizers", "starter": "Appetizers",
        "main": "Main Course", "entree": "Main Course",
        "dessert": "Desserts", "sweet": "Desserts",
        "pizza": "Pizzas", "burger": "Burgers",
        "sushi": "Sushi Rolls", "pasta": "Pasta",
        "salad": "Salads", "soup": "Soups",
        "sandwich": "Sandwiches", "kebab": "Kabab",
        "seafood": "Seafood",
    }
    for keyword, category in category_mapping.items():
        if keyword in query_lower:
            filters["category_name"] = category
            break

    if "spicy" in query_lower:
        filters["spiciness"] = "spicy"

    return filters


def filter_results_by_metadata(results, filters=None):
    if not filters or not results:
        return results

    filtered_results = []
    for doc in results:
        metadata = doc.metadata
        include_doc = True

        # Early exit for common filters
        if "restaurant_name" in filters:
            rest_name = metadata.get("restaurant_name", "").lower()
            if filters["restaurant_name"].lower() not in rest_name:
                continue

        if "price" in filters:
            price_value = metadata.get("price", 0)
            if price_value == "" or price_value is None:
                price_value = 0
            try:
                price = float(price_value)
            except (ValueError, TypeError):
                price = 0.0
            price_filter = filters["price"]
            if ("min" in price_filter and price < price_filter["min"]) or \
               ("max" in price_filter and price > price_filter["max"]):
                continue

        if "spiciness" in filters:
            desc = metadata.get("description", "").lower()
            if "spicy" not in desc:
                continue

        if "category_name" in filters:
            cat = metadata.get("category_name", "").lower()
            if filters["category_name"].lower() not in cat:
                continue

        filtered_results.append(doc)
    return filtered_results

# =========================================================
# 5️⃣ Enhanced Recommendation Logic
# =========================================================
def _extract_json_from_markdown(content):
    if not content:
        return content
    content = content.strip()
    if content.startswith("```json"):
        content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
    elif content.startswith("```"):
        content = content[3:]
        if content.endswith("```"):
            content = content[:-3]
    start_idx = content.find('{')
    end_idx = content.rfind('}') + 1
    if start_idx != -1 and end_idx > start_idx:
        content = content[start_idx:end_idx]
    return content


def get_enhanced_recommendations(query, chat_history=None):
    parser = PydanticOutputParser(pydantic_object=RecommendationResponse)

    try:
        filters = extract_metadata_filters(query)
        results = vectorstore.similarity_search(query, k=8)
        filtered_results = filter_results_by_metadata(results, filters)

        # If no filtered results, try a broader search with fewer filters
        if not filtered_results:
            # Try with just category filter if available
            category_only_filters = {}
            if "category_name" in filters:
                category_only_filters["category_name"] = filters["category_name"]
            elif "price" in filters:
                category_only_filters["price"] = filters["price"]

            if category_only_filters:
                filtered_results = filter_results_by_metadata(results, category_only_filters)
            else:
                # If no filters to fall back to, use broader search
                filtered_results = vectorstore.similarity_search(query, k=5)

        if not filtered_results:
            return "Sorry, I couldn't find any matching dishes."

        menu_items = []
        for doc in filtered_results[:5]:
            m = doc.metadata

            # Skip items with missing essential data
            if not m.get("name") or not m.get("restaurant_name"):
                continue

            # Safely parse price with better error handling
            price_value = m.get("price", 0)
            if price_value == "" or price_value is None:
                price_value = 0
            try:
                price_float = float(price_value)
            except (ValueError, TypeError):
                price_float = 0.0

            # Skip items with invalid data
            if price_float < 0 or m.get("name") == "Unknown":
                continue

            menu_items.append({
                "name": m.get("name", "Unknown"),
                "restaurant": m.get("restaurant_name", "Unknown"),
                "price": price_float,
                "category": m.get("category_name", "Unknown"),
                "description": m.get("description", "No description available")
            })

        # If no valid items found, return error
        if not menu_items:
            # Try to get data directly from Supabase as fallback
            try:
                res = supabase.table("menu_items").select("name, price, description").limit(5).execute()
                if res.data:
                    fallback_items = []
                    for item in res.data[:3]:
                        fallback_items.append({
                            "name": item.get("name", "Unknown"),
                            "restaurant": "Available Restaurant",
                            "price": float(item.get("price", 0)) if item.get("price") else 0.0,
                            "category": "General",
                            "description": item.get("description", "No description available")
                        })
                    return f"I found these options from our menu:\n" + "\n".join([f"- {m['name']} — QR{m['price']:.2f}\n  {m['description']}" for m in fallback_items])
            except Exception:
                pass
            return "Sorry, I couldn't find any valid menu items matching your search."

        prompt = f"""
        You are a friendly food assistant.
        User query: "{query}"

        Menu items:
        {json.dumps(menu_items, indent=2)}

        Return strictly valid JSON:
        {{
            "matches": [...],
            "explanation": "...",
            "suggested_filters": [...]
        }}
        """

        response = retry_llm_call(lambda: smart_llm.invoke([HumanMessage(content=prompt)]))
        if response is None:
            raise Exception("Failed to get response from LLM after retries")
        raw = _extract_json_from_markdown(response.content.strip())

        try:
            parsed = parser.parse(raw)
            out = [f"{parsed.explanation}\n", "Here are your matches:"]
            for i in parsed.matches:
                out.append(f"- **{i.name}** ({i.category}) from *{i.restaurant}* — QR{i.price:.2f}\n  {i.description}")
            if parsed.suggested_filters:
                out.append("\nTry filtering by:")
                for f in parsed.suggested_filters:
                    out.append(f"  • {f}")
            return "\n".join(out)
        except Exception as e:
            # Improved error handling with fallback
            return f"I found these options:\n" + "\n".join([f"- {m['name']} at {m['restaurant']}" for m in menu_items[:3]])

    except Exception as e:
        return f"Sorry, I encountered an error while searching for recommendations: {str(e)}"

# =========================================================
# 6️⃣ Cart System
# =========================================================
cart = []

def query_menu_from_supabase(name: str):
    try:
        # Handle case where name is passed as a dict from the agent
        if isinstance(name, dict) and "name" in name:
            name = name["name"]

        # Ensure name is a string
        if not isinstance(name, str):
            name = str(name)

        res = supabase.table("menu_items").select("id, name, price, description").execute()
        items = res.data or []
        search_lower = name.lower()
        matches = [i for i in items if search_lower in str(i.get('name', '')).lower()]
        if not matches:
            return {"success": False, "message": f"No items found for '{name}'."}
        f = matches[0]
        return {"success": True, "item_id": f["id"], "name": f["name"], "price": f["price"], "message": f"Found '{f['name']}'"}
    except Exception as e:
        return {"success": False, "message": str(e)}

def add_to_cart(item_id=None, quantity=1):
    try:
        # Handle case where item_id is passed as a JSON string from the agent
        if isinstance(item_id, str):
            try:
                import json
                parsed_input = json.loads(item_id)
                if isinstance(parsed_input, dict):
                    item_id = parsed_input.get("item_id", item_id)
                    quantity = parsed_input.get("quantity", quantity)
            except json.JSONDecodeError:
                # If it's not JSON, treat it as a direct item_id
                pass

        # Handle case where item_id is passed as a dict
        if isinstance(item_id, dict) and "item_id" in item_id:
            item_id = item_id["item_id"]
            quantity = item_id.get("quantity", quantity)

        # Ensure item_id is a string and clean it up
        if item_id is None:
            return {"success": False, "message": "Item ID is required."}

        item_id = str(item_id).strip().strip('"\'')  # Remove quotes if present

        # Validate UUID format (basic check)
        if len(item_id) != 36 or item_id.count('-') != 4:
            return {"success": False, "message": f"Invalid item ID format: '{item_id}'"}

        res = supabase.table("menu_items").select("id, name, price").eq("id", item_id).execute()
        if not res.data:
            return {"success": False, "message": f"Item with ID '{item_id}' not found."}
        item = res.data[0]

        # Safely parse price
        price_value = item.get("price", 0)
        if price_value == "" or price_value is None:
            price_value = 0
        try:
            price_float = float(price_value)
        except (ValueError, TypeError):
            price_float = 0.0

        cart.append({"id": item["id"], "name": item["name"], "price": price_float, "quantity": quantity})
        return {"success": True, "message": f"Added {quantity} × {item['name']} to cart."}
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
# 7️⃣ Agent Setup
# =========================================================
tools = [
    Tool(name="QueryMenu", func=query_menu_from_supabase, description="Search for menu items by name. Input: item name as string."),
    Tool(name="AddToCart", func=add_to_cart, description="Add item to cart. Input: item_id (UUID string) and optional quantity (integer)."),
    Tool(name="ViewCart", func=view_cart, description="View cart contents."),
    Tool(name="Checkout", func=checkout, description="Checkout and place order."),
]

system_message = SystemMessage(content="You are a concise, direct agent. Use at most 5 reasoning steps.")
smart_agent = initialize_agent(
    tools=tools,
    llm=smart_llm,
    agent="zero-shot-react-description",
    verbose=True,
    handle_parsing_errors=True,
    memory=memory,
    max_iterations=20,          # Reduced to prevent timeout
    max_execution_time=120,      # Reduced to prevent timeout
    system_message=system_message
)


# =========================================================
# 8️⃣ Chat System
# =========================================================
def sync_memory_with_chat(chat_history, memory):
    """Efficiently sync chat history with memory, only adding new messages"""
    if not hasattr(sync_memory_with_chat, '_last_message_count'):
        sync_memory_with_chat._last_message_count = 0

    current_count = len(chat_history.messages)
    if current_count > sync_memory_with_chat._last_message_count:
        # Only add new messages
        for msg in chat_history.messages[sync_memory_with_chat._last_message_count:]:
            memory.chat_memory.add_message(msg)

    sync_memory_with_chat._last_message_count = current_count

def handle_greeting(query):
    greetings = {
        "hello": "Hello! I'm your food assistant. What are you craving today?",
        "hi": "Hi there! Ready to discover some delicious food?",
        "hey": "Hey! What kind of food can I help you find?",
        "help": "You can ask for food recommendations, calories, or manage your cart."
    }
    return greetings.get(query.lower())

def main():
    print("Welcome to the AI Food Assistant!\n")

    chat_history = ChatMessageHistory()
    while True:
        try:
            query = input("You: ").strip()
            if not query:
                continue
            if query.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
            if query.lower() == 'clear memory':
                chat_history.clear()
                memory.clear()
                print("Memory cleared ✅")
                continue

            greeting = handle_greeting(query)
            if greeting:
                print(f"Assistant: {greeting}")
                chat_history.add_ai_message(greeting)
                continue

            chat_history.add_user_message(query)
            sync_memory_with_chat(chat_history, memory)

            if any(w in query.lower() for w in ["add", "order", "cart", "checkout"]):
                print("🔧 Agent handling your request...")
                result = smart_agent.invoke({"input": query})
                response = result["output"] if isinstance(result, dict) else str(result)
                print(response)
                chat_history.add_ai_message(response)
                continue

            if "calorie" in query.lower():
                calorie_prompt = f"You are a nutrition assistant. Estimate calories for: {query}"
                response = retry_llm_call(lambda: smart_llm.invoke([HumanMessage(content=calorie_prompt)]))
                if response is None:
                    print("Sorry, I couldn't get calorie information right now.")
                    continue
                print(response.content.strip())
                chat_history.add_ai_message(response.content.strip())
                continue

            print("🔍 Finding recommendations...")
            response = get_enhanced_recommendations(query, chat_history)
            print(response)
            chat_history.add_ai_message(response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    
    main()