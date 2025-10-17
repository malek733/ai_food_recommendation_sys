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

    if "under QR" in query_lower or "less than QR" in query_lower:
        try:
            price_text = query_lower.split("QR")[1].split()[0]
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

    # Special handling for healthy food requests
    healthy_keywords = ["healthy", "health", "light", "fresh", "nutritious", "low calorie", "diet"]
    if any(keyword in query_lower for keyword in healthy_keywords):
        # Look for healthy categories
        healthy_categories = ["Salads", "Soups", "Seafood", "Appetizers"]
        filters["healthy_categories"] = healthy_categories

    cuisine_mapping = {
        "indian": "indian",
        "italian": "italian",
        "american": "american",
        "mediterranean": "mediterranean",
        "asian": "asian",
        "middle eastern": "middle_eastern",
        "fast food": "fast_food",
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

            if filter_key == "healthy_categories":
                category = metadata.get("category_name", "").lower()
                if category not in [cat.lower() for cat in filter_value]:
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

def _extract_json_from_markdown(content):
    """
    Extract JSON content from markdown-formatted text.

    Args:
        content (str): Raw response content that may contain markdown formatting

    Returns:
        str: Clean JSON string ready for parsing
    """
    if not content:
        return content

    # Remove common markdown code block patterns
    content = content.strip()

    # Handle ```json ... ``` blocks
    if content.startswith("```json"):
        content = content[7:]
        if content.endswith("```"):
            content = content[:-3]
    # Handle ``` ... ``` blocks (generic)
    elif content.startswith("```"):
        content = content[3:]
        if content.endswith("```"):
            content = content[:-3]

    # Remove any remaining leading/trailing whitespace
    content = content.strip()

    # Find JSON object boundaries
    start_idx = content.find('{')
    end_idx = content.rfind('}') + 1

    if start_idx != -1 and end_idx > start_idx:
        content = content[start_idx:end_idx]

    return content


def get_recommendations(query, chat_history=None):
    parser = PydanticOutputParser(pydantic_object=RecommendationResponse)

    conversation_context = ""
    user_preferences = {}
    if chat_history and chat_history.messages:
        # Get recent context from chat history
        recent_messages = chat_history.messages[-10:] if len(chat_history.messages) > 10 else chat_history.messages
        context_parts = []
        for msg in recent_messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")
        conversation_context = "\n".join(context_parts)
        user_preferences = {}

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
    You are a friendly and helpful AI food assistant. Respond in a warm, engaging manner.
    Your goal is to help customers find delicious food while making them feel welcomed and valued.

    User query: "{query}"

    {conversation_section}
    Available items:
    {context}

    Respond with a JSON object that includes:
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
        "explanation": "Friendly explanation of why these dishes match their preferences",
        "suggested_filters": ["helpful filter suggestions"],
        "follow_up": "A friendly question to help them further (e.g., 'Would you like to know about our vegetarian options?' or 'Can I help you with spice levels?')"
    }}

    Guidelines:
    - Be warm and conversational in the explanation
    - Suggest relevant alternatives
    - Include helpful tips about dishes
    - Add a follow-up question to continue the conversation
    - Keep the tone friendly and engaging
    """


    try:
        response = llm.invoke([HumanMessage(content=prompt)])

        # Clean the response content to extract only JSON
        content = response.content.strip()

        # Remove markdown code block markers and extract JSON
        content = _extract_json_from_markdown(content)

        parsed_response = parser.parse(content)

        output = []
        output.append(f"🤖 {parsed_response.explanation}\n")
        output.append("📋 Here are some great matches for you:")
        
        for item in parsed_response.matches:
            output.append(
                f"\n- **{item.name}** ({item.category}) from *{item.restaurant}*\n"
                f"  💰 Price: QR{item.price:.2f}\n"
                f"  📝 {item.description}"
            )

        if parsed_response.suggested_filters:
            output.append("\n💡 You might also want to try:")
            for filter_suggestion in parsed_response.suggested_filters:
                output.append(f"  • {filter_suggestion}")

        # Add the follow-up question
        output.append(f"\n👋 {getattr(parsed_response, 'follow_up', 'Is there anything else I can help you find?')}")

        return "\n".join(output)

    except Exception as e:
        # Update the fallback response to be more friendly
        fallback_output = ["I found these delicious options for you:"]
        for item in menu_items[:3]:
            fallback_output.append(
                f"- **{item['name']}** at {item['restaurant']} - QR{item['price']:.2f}"
            )
        fallback_output.append("\n💭 Would you like to know more about any of these dishes?")
        return "\n".join(fallback_output)

# Initialize embeddings, Chroma, and LLM
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)


def main():
    print("\n" + "="*60)
    print("Welcome to the Interactive Food Recommendation System!")
    print("With Conversation Memory and Calorie Lookup!")
    print("="*60 + "\n")

    chat_history = ChatMessageHistory()
    last_items = []  # Track last shown items

    while True:
        try:
            query = input("\nYou: ").strip()
            if not query:
                continue

            if query.lower() in ['exit', 'quit', 'bye']:
                print("\nGoodbye! Have a tasty day!")
                break

            if query.lower() == 'history':
                # Get chat history statistics
                message_count = len(chat_history.messages)
                stats = {
                    'message_count': message_count,
                    'memory_type': 'ChatMessageHistory',
                    'chat_history_size': message_count
                }
                print(f"\n📚 Chat History: {stats}")
                # Get recent messages from chat history
                recent_messages = chat_history.messages[-10:] if len(chat_history.messages) > 10 else chat_history.messages
                context_parts = []
                for msg in recent_messages:
                    if isinstance(msg, HumanMessage):
                        context_parts.append(f"User: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        context_parts.append(f"Assistant: {msg.content}")
                print("\n".join(context_parts))
                continue

            if query.lower() == 'clear memory':
                chat_history.clear()
                last_items.clear()  # Clear tracked items too
                print("Chat history cleared!")
                continue

            greeting = handle_greeting(query)
            if greeting:
                print(f"\nAssistant: {greeting}")
                chat_history.add_ai_message(greeting)
                continue

            chat_history.add_user_message(query)
            print("\nProcessing...")

            # Handle references to previous items
            if any(x in query.lower() for x in ["first", "second", "third", "last"]) and last_items:
                try:
                    index = -1
                    if "first" in query.lower(): index = 0
                    elif "second" in query.lower(): index = 1
                    elif "third" in query.lower(): index = 2
                    elif "last" in query.lower(): index = -1
                    
                    referenced_item = last_items[index]
                    query = query.lower().replace("first", "").replace("second", "").replace("third", "").replace("last", "")
                    query = f"{query} {referenced_item['name']}"
                except IndexError:
                    print("I couldn't find that item in the previous results.")
                    continue

            # Calorie query handling
            if "calorie" in query.lower() or "calories" in query.lower():
                # Check if user is asking about a specific item or "this item" from last recommendation
                query_lower = query.lower()
                item_context = ""

                if "this item" in query_lower and last_items:
                    # Use the first item from last recommendation
                    referenced_item = last_items[0]
                    item_context = f" for {referenced_item['name']} from {referenced_item['restaurant']}"
                elif "first" in query_lower and last_items:
                    referenced_item = last_items[0]
                    item_context = f" for {referenced_item['name']} from {referenced_item['restaurant']}"
                elif "second" in query_lower and len(last_items) > 1:
                    referenced_item = last_items[1]
                    item_context = f" for {referenced_item['name']} from {referenced_item['restaurant']}"
                elif "third" in query_lower and len(last_items) > 2:
                    referenced_item = last_items[2]
                    item_context = f" for {referenced_item['name']} from {referenced_item['restaurant']}"

                calorie_prompt = f"""
                You are a nutrition assistant providing brief, focused responses.
                Query: "{query}"{item_context}

                Respond with:
                1. Estimated calorie range for the item
                2. Keep it concise and factual
                3. If no specific item mentioned, ask for clarification
                """

                response = llm.invoke([
                    SystemMessage(content="You are a precise nutrition assistant."),
                    HumanMessage(content=calorie_prompt)
                ]).content.strip()
            else:
                response = get_recommendations(query, chat_history)
                # Store items for reference
                try:
                    parsed = json.loads(response)
                    if "matches" in parsed:
                        last_items = parsed["matches"]
                except:
                    # If parsing fails, try to extract items from formatted response
                    items = []
                    for line in response.split("\n"):
                        if line.startswith("- **"):
                            item_info = line.split("**")[1:]
                            if item_info:
                                items.append({"name": item_info[0]})
                    last_items = items

            print(f"\n{response}")
            chat_history.add_ai_message(response)

        except KeyboardInterrupt:
            print("\nExiting gracefully...")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

        print("\n" + "="*60)

if __name__ == "__main__":
    main()