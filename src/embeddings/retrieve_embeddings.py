from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.output_parsers import PydanticOutputParser
from langchain_community.chat_message_histories import ChatMessageHistory
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
import json
import time
import os
from datetime import datetime

# Load environment variables
load_dotenv()

class MenuItem(BaseModel):
    name: str = Field(description="Name of the dish")
    restaurant: str = Field(description="Restaurant name")
    price: float = Field(description="Price of the dish")
    category: str = Field(description="Category of the dish")
    description: Optional[str] = Field(description="Description of the dish", default=None)
    spiciness: Optional[str] = Field(description="Spiciness level of the dish", default=None)

class RecommendationResponse(BaseModel):
    matches: List[MenuItem] = Field(description="List of matching menu items")
    explanation: str = Field(description="Natural language explanation of the recommendations")
    suggested_filters: Optional[List[str]] = Field(
        description="Suggested filters for narrowing down results",
        default_factory=list
    )

class ConversationMemory:
    """Enhanced conversation memory with persistence and context awareness."""

    def __init__(self, memory_file="conversation_memory.json", max_messages=50):
        self.memory_file = memory_file
        self.max_messages = max_messages
        self.chat_history = ChatMessageHistory()
        self.load_memory()
        self.current_session_start = datetime.now()

    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for message_data in data.get('messages', []):
                        msg_type = message_data['type']
                        content = message_data['content']

                        if msg_type == 'human':
                            self.chat_history.add_user_message(content)
                        elif msg_type == 'ai':
                            self.chat_history.add_ai_message(content)

                    if len(self.chat_history.messages) > self.max_messages:
                        self.chat_history.messages = self.chat_history.messages[-self.max_messages:]
        except Exception as e:
            print(f"[WARNING] Could not load conversation memory: {e}")

    def save_memory(self):
        try:
            messages = []
            for message in self.chat_history.messages:
                msg_data = {
                    'type': 'human' if isinstance(message, HumanMessage) else 'ai',
                    'content': message.content,
                    'timestamp': datetime.now().isoformat()
                }
                messages.append(msg_data)

            if len(messages) > self.max_messages:
                messages = messages[-self.max_messages:]

            data = {
                'session_start': self.current_session_start.isoformat(),
                'last_updated': datetime.now().isoformat(),
                'messages': messages
            }

            with open(self.memory_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"[WARNING] Could not save conversation memory: {e}")

    def add_user_message(self, message):
        self.chat_history.add_user_message(message)

    def add_ai_message(self, message):
        self.chat_history.add_ai_message(message)

    def get_recent_context(self, limit=10):
        messages = self.chat_history.messages[-limit*2:]
        context_parts = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")

        return "\n".join(context_parts)

    def get_user_preferences(self):
        preferences = {
            'liked_restaurants': set(),
            'liked_categories': set(),
            'price_range': None,
            'avoided_items': set()
        }
        return preferences

    def clear_memory(self):
        self.chat_history.clear()
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        self.current_session_start = datetime.now()

    def get_memory_stats(self):
        message_count = len(self.chat_history.messages)
        return {
            'message_count': message_count,
            'session_duration': str(datetime.now() - self.current_session_start),
            'memory_file_size': os.path.getsize(self.memory_file) if os.path.exists(self.memory_file) else 0
        }

def extract_metadata_filters(query):
    filters = {}
    query_lower = query.lower()

    restaurant_indicators = ["from", "at", "of", "in"]
    for indicator in restaurant_indicators:
        if f" {indicator} " in f" {query_lower} ":
            parts = query_lower.split(f" {indicator} ")
            if len(parts) > 1:
                rest_name = parts[1].strip()
                filters['restaurant_name'] = rest_name
                break

    price_patterns = {
        "cheap": {"price": {"max": 15}},
        "expensive": {"price": {"min": 30}},
        "moderate": {"price": {"min": 15, "max": 30}},
        "budget": {"price": {"max": 20}},
        "high-end": {"price": {"min": 40}},
    }

    if "under $" in query_lower or "less than $" in query_lower:
        try:
            price_text = query_lower.split("$")[1].split()[0]
            max_price = float(price_text)
            filters["price"] = {"max": max_price}
        except (IndexError, ValueError):
            pass
    else:
        for pattern, price_filter in price_patterns.items():
            if pattern in query_lower:
                filters["price"] = price_filter["price"]
                break

    category_mapping = {
        "appetizer": "Appetizers",
        "starter": "Appetizers",
        "main": "Main Course",
        "entree": "Main Course",
        "dessert": "Desserts",
        "sweet": "Desserts",
        "pizza": "Pizzas",
        "burger": "Burgers",
        "sushi": "Sushi Rolls",
        "pasta": "Pasta",
        "salad": "Salads",
        "soup": "Soups",
        "sandwich": "Sandwiches",
        "kebab": "Kabab",
        "seafood": "Seafood",
    }

    cuisine_mapping = {
        "indian": "indian",
        "italian": "italian",
        "american": "american",
        "mediterranean": "mediterranean",
        "asian": "asian",
        "middle eastern": "middle_eastern",
        "fast food": "fast_food",
        "healthy": "healthy",
        "vegetarian": "vegetarian",
        "arabic": "arabic",
    }

    if "spicy" in query_lower:
        filters["spiciness"] = "spicy"

    for keyword, category in category_mapping.items():
        if keyword in query_lower:
            filters["category_name"] = category
            break

    for cuisine, value in cuisine_mapping.items():
        if cuisine in query_lower:
            filters["restaurant_category"] = value
            break

    return filters

def filter_results_by_metadata(results, filters=None):
    if not filters or not results:
        return results

    filtered_results = []
    
    for doc in results:
        metadata = doc.metadata
        include_doc = True

        for filter_key, filter_value in filters.items():
            if filter_key == "restaurant_name":
                rest_name = metadata.get("restaurant_name", "").lower()
                filter_terms = filter_value.lower().split()
                if not any(term in rest_name for term in filter_terms):
                    include_doc = False
                    break
                continue

            if filter_key == "price" and isinstance(filter_value, dict):
                try:
                    price = float(metadata.get("price", 0))
                except (ValueError, TypeError):
                    price = 0.0
                
                if ("min" in filter_value and price < filter_value["min"]) or \
                   ("max" in filter_value and price > filter_value["max"]):
                    include_doc = False
                    break
                continue

            if filter_key == "spiciness":
                description = metadata.get("description", "").lower()
                if "spicy" not in description and "hot" not in description:
                    include_doc = False
                    break
                continue

            if filter_key in ["category_name", "restaurant_category"]:
                value = metadata.get(filter_key, "").lower()
                if filter_value.lower() not in value:
                    include_doc = False
                    break
                continue

        if include_doc:
            filtered_results.append(doc)

    return filtered_results

def handle_greeting(query):
    greetings = {
        "hello": "Hello! I'm your food assistant. What are you craving today?",
        "hi": "Hi there! Ready to discover some delicious food?",
        "hey": "Hey! What kind of food can I help you find?",
        "good morning": "Good morning! Ready for some breakfast recommendations?",
        "good afternoon": "Good afternoon! Looking for lunch ideas?",
        "good evening": "Good evening! How about some dinner suggestions?",
        "help": """I can help you find food in several ways:
- Search by restaurant, category, or price
- Ask about calories (e.g., "how many calories in fries?")
- Type 'history' to view past chats
- Type 'clear memory' to reset memory"""
    }
    return greetings.get(query.lower())

def get_recommendations(query, conversation_memory=None):
    parser = PydanticOutputParser(pydantic_object=RecommendationResponse)

    conversation_context = ""
    user_preferences = {}
    if conversation_memory:
        conversation_context = conversation_memory.get_recent_context()
        user_preferences = conversation_memory.get_user_preferences()

    filters = extract_metadata_filters(query)
    results = chroma_db.similarity_search(query, k=8)
    filtered_results = filter_results_by_metadata(results, filters)

    if not filtered_results:
        filtered_results = chroma_db.similarity_search(query, k=5)

    if not filtered_results:
        return "Sorry, I couldn't find any matching dishes. 😔 Try adjusting your search."

    menu_items = []
    for doc in filtered_results[:5]:
        metadata = doc.metadata
        try:
            price = float(metadata.get("price", 0))
        except (ValueError, TypeError):
            price = 0.0

        menu_items.append({
            "name": metadata.get("name", "Unknown item"),
            "restaurant": metadata.get("restaurant_name", "Unknown restaurant"),
            "price": price,
            "category": metadata.get("category_name", "Unspecified category"),
            "description": metadata.get("description", "No description available")
        })

    context = json.dumps(menu_items, indent=2)

    conversation_section = f"CONVERSATION HISTORY:\n{conversation_context}\n" if conversation_context else ""

    prompt = f"""
    You are a friendly AI food assistant. You must respond with ONLY valid JSON that matches the required format.

    User query: "{query}"

    {conversation_section}
    Available items:
    {context}

    CRITICAL: Respond with ONLY a JSON object in this exact format:
    {{
        "matches": [
            {{
                "name": "Dish Name",
                "restaurant": "Restaurant Name",
                "price": 15.99,
                "category": "Category",
                "description": "Description"
            }}
        ],
        "explanation": "Brief explanation of recommendations",
        "suggested_filters": ["filter1", "filter2"]
    }}

    - Use only the dishes from the available items list.
    - Do NOT include any text before or after the JSON.
    - Do NOT wrap the JSON in code blocks.
    - Ensure all required fields are present and valid.
    - The response must be parseable by JSON.parse().
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])

        # Clean the response content to extract only JSON
        content = response.content.strip()

        # Remove any markdown code block markers
        if content.startswith("```json"):
            content = content[7:]
        if content.endswith("```"):
            content = content[:-3]

        # Remove any leading/trailing text that might interfere with JSON parsing
        content = content.strip()

        # Try to find JSON object boundaries
        start_idx = content.find('{')
        end_idx = content.rfind('}') + 1

        if start_idx != -1 and end_idx > start_idx:
            content = content[start_idx:end_idx]

        parsed_response = parser.parse(content)

        output = [f"Assistant: {parsed_response.explanation}\n", "Matching dishes:"]
        for item in parsed_response.matches:
            output.append(
                f"\n- **{item.name}** ({item.category}) from *{item.restaurant}*\n"
                f"  Price: ${item.price:.2f}\n"
                f"  Description: {item.description or 'No description available'}"
            )

        if parsed_response.suggested_filters:
            output.append("\nYou can refine your search by:")
            for f in parsed_response.suggested_filters:
                output.append(f"  • {f}")

        return "\n".join(output)
    except Exception as e:
        # Fallback: try to extract and display what we can from the available items
        fallback_output = ["I found some great options for you!\n", "Matching dishes:"]
        for item in menu_items[:3]:  # Show first 3 items as fallback
            fallback_output.append(
                f"\n- **{item['name']}** ({item['category']}) from *{item['restaurant']}*\n"
                f"  Price: ${item['price']:.2f}\n"
                f"  Description: {item['description'] or 'No description available'}"
            )

        fallback_output.append(f"\n\nNote: There was an issue formatting the response: {str(e)}")
        return "\n".join(fallback_output)

# Initialize embeddings, Chroma, and LLM
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1)

def main():
    print("\n" + "="*60)
    print("Welcome to the Interactive Food Recommendation System!")
    print("With Conversation Memory and Calorie Lookup!")
    print("="*60 + "\n")

    memory = ConversationMemory()

    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue

            if query.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye! Have a tasty day!")
                memory.save_memory()
                break

            if query.lower() == 'history':
                stats = memory.get_memory_stats()
                print(f"\n📚 Memory: {stats}")
                print(memory.get_recent_context(5))
                continue

            if query.lower() == 'clear memory':
                memory.clear_memory()
                print("Memory cleared!")
                continue

            greeting = handle_greeting(query)
            if greeting:
                print(f"\nAssistant: {greeting}")
                memory.add_ai_message(greeting)
                continue

            memory.add_user_message(query)
            print("\nProcessing...")

            # ✅ Calorie mode detection
            if "calorie" in query.lower() or "calories" in query.lower():
                calorie_prompt = f"""
                You are a nutrition assistant. The user asked: "{query}"
                Answer briefly with the approximate calories in kcal.
                Format example: "🍕 Pepperoni pizza: about 285 kcal per slice."
                Be direct. No extra explanations.
                """
                response = llm.invoke([HumanMessage(content=calorie_prompt)]).content.strip()
            else:
                response = get_recommendations(query, memory)

            print(f"\n{response}")
            memory.add_ai_message(response)

        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            memory.save_memory()
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

        print("\n" + "="*60)

if __name__ == "__main__":
    main()
