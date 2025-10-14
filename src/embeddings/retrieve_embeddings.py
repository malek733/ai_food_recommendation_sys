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
        """Load conversation history from file."""
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

                    # Keep only recent messages to maintain window
                    if len(self.chat_history.messages) > self.max_messages:
                        self.chat_history.messages = self.chat_history.messages[-self.max_messages:]
        except Exception as e:
            print(f"[WARNING] Could not load conversation memory: {e}")

    def save_memory(self):
        """Save conversation history to file."""
        try:
            messages = []
            for message in self.chat_history.messages:
                msg_data = {
                    'type': 'human' if isinstance(message, HumanMessage) else 'ai',
                    'content': message.content,
                    'timestamp': datetime.now().isoformat()
                }
                messages.append(msg_data)

            # Keep only recent messages to avoid file bloat
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
        """Add user message to memory."""
        self.chat_history.add_user_message(message)

    def add_ai_message(self, message):
        """Add AI message to memory."""
        self.chat_history.add_ai_message(message)

    def get_recent_context(self, limit=10):
        """Get recent conversation context for recommendations."""
        messages = self.chat_history.messages[-limit*2:]  # *2 for user+ai pairs
        context_parts = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                context_parts.append(f"User: {msg.content}")
            elif isinstance(msg, AIMessage):
                context_parts.append(f"Assistant: {msg.content}")

        return "\n".join(context_parts)

    def get_user_preferences(self):
        """Extract user preferences from conversation history."""
        preferences = {
            'liked_restaurants': set(),
            'liked_categories': set(),
            'price_range': None,
            'avoided_items': set()
        }

        messages = self.chat_history.messages
        for msg in messages:
            if isinstance(msg, HumanMessage):
                content = msg.content.lower()
                # Extract positive mentions
                if any(word in content for word in ['love', 'like', 'great', 'excellent', 'amazing', 'perfect']):
                    # This is a simplified preference extraction
                    # In a real system, you'd use more sophisticated NLP
                    pass

        return preferences

    def clear_memory(self):
        """Clear all conversation memory."""
        self.chat_history.clear()
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)
        self.current_session_start = datetime.now()

    def get_memory_stats(self):
        """Get memory usage statistics."""
        message_count = len(self.chat_history.messages)
        return {
            'message_count': message_count,
            'session_duration': str(datetime.now() - self.current_session_start),
            'memory_file_size': os.path.getsize(self.memory_file) if os.path.exists(self.memory_file) else 0
        }

def extract_metadata_filters(query):
    """Extract metadata filters from user query."""
    filters = {}
    query_lower = query.lower()
    
    # Restaurant name filters
    restaurant_indicators = ["from", "at", "of", "in"]
    for indicator in restaurant_indicators:
        if f" {indicator} " in f" {query_lower} ":
            parts = query_lower.split(f" {indicator} ")
            if len(parts) > 1:
                rest_name = parts[1].strip()
                filters['restaurant_name'] = rest_name
                break

    # Price filters with extended patterns
    price_patterns = {
        "cheap": {"price": {"max": 15}},
        "expensive": {"price": {"min": 30}},
        "moderate": {"price": {"min": 15, "max": 30}},
        "budget": {"price": {"max": 20}},
        "high-end": {"price": {"min": 40}},
    }

    # Handle explicit price mentions
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

    # Category mapping
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

    # Cuisine type mapping
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

    # Taste preferences
    if "spicy" in query_lower:
        filters["spiciness"] = "spicy"

    # Check categories
    for keyword, category in category_mapping.items():
        if keyword in query_lower:
            filters["category_name"] = category
            break

    # Check cuisine types
    for cuisine, value in cuisine_mapping.items():
        if cuisine in query_lower:
            filters["restaurant_category"] = value
            break

    return filters

def filter_results_by_metadata(results, filters=None):
    """Filter search results based on metadata criteria."""
    if not filters or not results:
        return results

    filtered_results = []
    
    for doc in results:
        metadata = doc.metadata
        include_doc = True

        for filter_key, filter_value in filters.items():
            # Restaurant name matching
            if filter_key == "restaurant_name":
                rest_name = metadata.get("restaurant_name", "").lower()
                filter_terms = filter_value.lower().split()
                if not any(term in rest_name for term in filter_terms):
                    include_doc = False
                    break
                continue

            # Price range filtering
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

            # Spiciness filter
            if filter_key == "spiciness":
                description = metadata.get("description", "").lower()
                if "spicy" not in description and "hot" not in description:
                    include_doc = False
                    break
                continue

            # Category and cuisine type matching
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
    """Handle common greetings and return appropriate responses."""
    greetings = {
        "hello": "Hello! 👋 I'm your food assistant. What are you craving today?",
        "hi": "Hi there! 😊 Ready to discover some delicious food?",
        "hey": "Hey! 🌟 What kind of food can I help you find?",
        "good morning": "Good morning! ☀️ Ready for some breakfast recommendations?",
        "good afternoon": "Good afternoon! 🌞 Looking for lunch ideas?",
        "good evening": "Good evening! 🌙 How about some dinner suggestions?",
        "help": """I can help you find food in several ways:
- Search by restaurant (e.g., "what's available at Ocean Lobster?")
- Search by price (e.g., "show me cheap burgers" or "dishes under $20")
- Search by category (e.g., "show me appetizers" or "what desserts are available?")
- Search by cuisine (e.g., "show me Italian dishes")
- Search by taste (e.g., "I want something spicy")

🍽️ FOOD COMMANDS:
Just ask naturally and I'll help you find what you're looking for!

🧠 MEMORY COMMANDS:
- 'history' - View conversation history and memory stats
- 'memory stats' - Show detailed memory statistics
- 'clear memory' - Clear all conversation history

The system remembers your preferences and previous conversations to provide better recommendations!"""
    }
    return greetings.get(query.lower())

def get_recommendations(query, conversation_memory=None):
    """Get filtered recommendations based on user query with conversation context."""
    parser = PydanticOutputParser(pydantic_object=RecommendationResponse)

    # Get conversation context if memory is available
    conversation_context = ""
    user_preferences = {}
    if conversation_memory:
        conversation_context = conversation_memory.get_recent_context()
        user_preferences = conversation_memory.get_user_preferences()

    filters = extract_metadata_filters(query)
    results = chroma_db.similarity_search(query, k=8)
    filtered_results = filter_results_by_metadata(results, filters)

    if not filtered_results:
        # Try without filters if no results found
        filtered_results = chroma_db.similarity_search(query, k=5)

    if not filtered_results:
        return "Sorry, I couldn't find any matching dishes with those criteria. 😔\nTry adjusting your search criteria or ask for 'help'."

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

    # Enhanced prompt with conversation context
    conversation_section = f"CONVERSATION HISTORY:\n{conversation_context}\n" if conversation_context else ""

    prompt = f"""
    You are a friendly AI food assistant with conversation memory.

    User query: "{query}"

    {conversation_section}
    Available matching items:
    {context}

    Generate a response in the following format:
    {parser.get_format_instructions()}

    Requirements:
    1. Use ONLY the items provided above
    2. Reference previous conversation if relevant
    3. Remember user preferences from conversation history
    4. Explanation should be friendly and conversational
    5. Include specific prices and restaurant names
    6. Suggest relevant filters based on the available options
    7. If this follows up from previous recommendations, acknowledge the continuity
    """

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        parsed_response = parser.parse(response.content)

        output = []
        output.append(f"🤖 {parsed_response.explanation}\n")
        output.append("📋 Here are the matching dishes:")

        for item in parsed_response.matches:
            output.append(
                f"\n- **{item.name}** ({item.category}) from *{item.restaurant}*\n"
                f"  💰 Price: ${item.price:.2f}\n"
                f"  📝 {item.description or 'No description available'}"
            )

        if parsed_response.suggested_filters:
            output.append("\n🔍 You can refine your search by:")
            for filter_suggestion in parsed_response.suggested_filters:
                output.append(f"  • {filter_suggestion}")

        return "\n".join(output)

    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}\nPlease try again with a different query."

# Initialize embeddings and Chroma database
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

# Initialize LLM (Groq)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.1
)

def main():
    """Main interactive loop for the recommendation system with conversation memory."""
    print("\n" + "="*60)
    print("🍽️ Welcome to the Interactive Food Recommendation System!")
    print("🧠 Enhanced with Conversation Memory!")
    print("Type 'help' for guidance, 'history' for conversation history, or 'exit' to quit")
    print("="*60 + "\n")

    # Initialize conversation memory
    memory = ConversationMemory()

    while True:
        try:
            query = input("\nYou: ").strip()

            if not query:
                print("Please ask me something! Type 'help' if you need suggestions.")
                continue

            if query.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Thank you for using our service! Have a great meal!")
                memory.save_memory()  # Save memory before exit
                break

            # Handle special commands
            if query.lower() == 'history':
                stats = memory.get_memory_stats()
                print(f"\n📚 Conversation History Stats:")
                print(f"  • Messages in memory: {stats['message_count']}")
                print(f"  • Session duration: {stats['session_duration']}")
                print(f"  • Memory file size: {stats['memory_file_size']} bytes")

                recent_context = memory.get_recent_context(5)
                if recent_context:
                    print(f"\n💬 Recent conversation:\n{recent_context}")
                else:
                    print("\n💬 No conversation history yet.")
                continue

            if query.lower() == 'clear memory':
                memory.clear_memory()
                print("\n🗑️ Conversation memory cleared!")
                continue

            if query.lower() == 'memory stats':
                stats = memory.get_memory_stats()
                print(f"\n📊 Memory Statistics:")
                for key, value in stats.items():
                    print(f"  • {key.replace('_', ' ').title()}: {value}")
                continue

            greeting_response = handle_greeting(query)
            if greeting_response:
                print(f"\n🤖 {greeting_response}")
                memory.add_ai_message(greeting_response)
                continue

            # Add user message to memory
            memory.add_user_message(query)

            print("\n🔍 Searching for recommendations...")
            time.sleep(0.5)  # Add a small delay for better UX

            response = get_recommendations(query, memory)
            print(f"\n{response}")

            # Add AI response to memory
            memory.add_ai_message(response)

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Have a great meal!")
            memory.save_memory()  # Save memory before exit
            break
        except Exception as e:
            print(f"\n😅 Oops! Something went wrong: {str(e)}")
            print("Please try rephrasing your question or ask for 'help'")

        print("\n" + "="*60)

if __name__ == "__main__":
    main()