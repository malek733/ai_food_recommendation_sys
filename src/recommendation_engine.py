from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

# Load environment variables
load_dotenv()

# Initialize embeddings and Chroma
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
chroma_db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings_model)

# Initialize LLM (Groq)
llm = ChatGroq(
    model="Qwen/Qwen3-32b",
    temperature=0.4
)

# Define structured output schema
response_schemas = [
    ResponseSchema(
        name="recommendations",
        description="List of recommended dishes with name, restaurant, price, and description"
    ),
    ResponseSchema(
        name="summary",
        description="A human-like explanation of why these dishes match the user query"
    )
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

def extract_metadata_filters(query):
    """Extract metadata filters from user query."""
    filters = {}
    query_lower = query.lower()
    
    # Restaurant filters
    restaurant_keywords = {
        "from": lambda x: x.split("from")[1].strip().split()[0],
        "at": lambda x: x.split("at")[1].strip().split()[0],
        "restaurant": lambda x: x.split("restaurant")[1].strip().split()[0]
    }
    
    for keyword, extractor in restaurant_keywords.items():
        if keyword in query_lower:
            try:
                restaurant_name = extractor(query_lower)
                if restaurant_name:
                    filters['restaurant_name'] = restaurant_name
                break
            except (IndexError, AttributeError):
                continue

    # Price range filters with error handling
    try:
        price_patterns = {
            "cheap": {"price": {"max": 15}},
            "expensive": {"price": {"min": 30}},
            "moderate": {"price": {"min": 15, "max": 30}},
        }

        # Handle "under $X" pattern separately
        if "under $" in query_lower:
            try:
                max_price = float(query_lower.split("under $")[1].split()[0])
                filters["price"] = {"max": max_price}
            except (IndexError, ValueError):
                pass
        else:
            for pattern, price_filter in price_patterns.items():
                if pattern in query_lower:
                    filters["price"] = price_filter["price"]
                    break
    except Exception as e:
        print(f"[WARNING] Error processing price filters: {e}")

    # Category filters based on actual categories
    category_mapping = {
        "appetizer": "Appetizers",
        "starter": "Appetizers",
        "main": "Main Course",
        "dessert": "Desserts",
        "sweet": "Desserts",
        "pizza": "Pizzas",
        "burger": "Burgers",
        "sushi": "Sushi Rolls",
        "pasta": "Pasta",
        "side": "Sides"
    }

    # Cuisine/Restaurant type filters with normalized values
    cuisine_mapping = {
        "indian": "indian",
        "italian": "italian",
        "american": "american",
        "seafood": "seafood",
        "healthy": "healthy",
        "fast food": "fast_food",
        "arabic": "arabic",
        "asian": "asian",
        "iranian": "iranian",
        "qatari": "qatari"
    }

    # Check for category matches
    for keyword, category in category_mapping.items():
        if keyword in query_lower:
            filters["category_name"] = category
            break

    # Check for cuisine type matches
    for cuisine, normalized_value in cuisine_mapping.items():
        if cuisine in query_lower:
            filters["restaurant_category"] = normalized_value
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
            # Skip if the metadata doesn't have the key we're filtering on
            if filter_key not in metadata and filter_key != "restaurant_name":
                continue

            # Handle restaurant category filtering
            if filter_key == "restaurant_category":
                restaurant_cats = metadata.get("restaurant_category", "").lower()
                if filter_value.lower() not in restaurant_cats:
                    include_doc = False
                    break
                continue

            # Handle price range filtering
            if filter_key == "price" and isinstance(filter_value, dict):
                price = float(metadata.get("price", 0))
                if ("min" in filter_value and price < filter_value["min"]) or \
                   ("max" in filter_value and price > filter_value["max"]):
                    include_doc = False
                    break
                continue

            # Handle restaurant name filtering
            if filter_key == "restaurant_name":
                rest_name = metadata.get("restaurant_name", "").lower()
                if filter_value.lower() not in rest_name:
                    include_doc = False
                    break
                continue

            # Handle exact match filtering for other fields
            if str(metadata.get(filter_key, "")).lower() != str(filter_value).lower():
                include_doc = False
                break

        if include_doc:
            filtered_results.append(doc)

    return filtered_results

# Welcome message
print("Welcome to GoFood AI Assistant! 🍽️")
print("Ask me about food recommendations. Type 'exit' to quit.\n")

# Define greetings and their responses
greetings = {
    "hello": "Hello! I'm excited to help you find delicious food! 🍽️",
    "hi": "Hi there! Ready to discover some amazing dishes? 😊",
    "hey": "Hey! What kind of food are you craving today? 🌟",
    "good morning": "Good morning! Hope you're having a great start to your day! ☀️",
    "good afternoon": "Good afternoon! Looking for a tasty lunch or snack? 🍽️",
    "good evening": "Good evening! How about some dinner recommendations? 🍽️",
    "how are you": "I'm doing great, thank you for asking! Ready to help you find the perfect meal! 😊",
    "what's up": "Not much, just ready to recommend some delicious food! What are you in the mood for? 🍕",
    "help": "I can help you find great food! Just tell me what you're craving - spicy food, healthy options, or specific cuisines! 🍽️"
}

# Start chat loop
while True:
    try:
        query = input("You: ").strip()

        if not query:
            print("Please enter a valid query!")
            continue

        if query.lower() in ["exit", "quit", "bye"]:
            print("Goodbye! Hope you enjoy your meal! 👋")
            break

        # Check for greetings
        query_lower = query.lower()
        if query_lower in greetings:
            print(f"GoFood AI: {greetings[query_lower]}\n")
            continue

        # Extract metadata filters
        filters = extract_metadata_filters(query)

        # Perform similarity search with error handling
        try:
            results = chroma_db.similarity_search(query, k=8)
        except Exception as e:
            print(f"Error performing search: {e}")
            continue

        # Apply metadata filters
        filtered_results = filter_results_by_metadata(results, filters)

        if not filtered_results:
            print("Sorry, I couldn't find any matching dishes in our database. 😔\n")
            continue

        # Build context for LLM
        context_parts = []
        for doc in filtered_results[:4]:  # Limit to top 4 results
            metadata = doc.metadata
            item_info = [
                f"Name: {metadata.get('name', 'N/A')}",
                f"Restaurant: {metadata.get('restaurant_name', 'N/A')}",
                f"Price: ${metadata.get('price', 'N/A')}",
                f"Description: {metadata.get('description', 'N/A')}",
                f"Category: {metadata.get('category_name', 'N/A')}"
            ]
            context_parts.append("\n".join(item_info))

        context = "\n\n".join(context_parts)

        # Build prompt
        prompt = f"""
        You are an intelligent AI food recommendation assistant. Use ONLY the information provided in the database context below.

        CRITICAL INSTRUCTIONS:
        - ONLY recommend items that are EXPLICITLY listed in the database context provided
        - DO NOT create or invent new menu items, restaurants, or descriptions
        - DO NOT hallucinate or make up information
        - If no suitable items exist in the context, say "I couldn't find matching dishes in our current database"

        User query: "{query}"

        DATABASE CONTEXT (Only use this information):
        {context}

        RESPONSE FORMAT:
        {format_instructions}
        """

        # Get LLM response with error handling
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            parsed_output = output_parser.parse(response.content)
            
            recommendations = parsed_output.get("recommendations", [])
            summary = parsed_output.get("summary", "")

            print("\nGoFood AI:\n")
            print(summary)
            print("\nRecommendations:\n")

            if recommendations:
                for i, rec in enumerate(recommendations, 1):
                    print(f"{i}. {rec['name']} from {rec['restaurant']}")
                    print(f"   Price: ${rec['price']}")
                    print(f"   {rec['description']}\n")
            else:
                print("No specific recommendations found for your query.\n")

        except Exception as e:
            print("Sorry, I had trouble processing the recommendations. Please try again. 😅\n")
            print(f"Debug info: {e}\n")

        print("---------------------------------------\n")

    except KeyboardInterrupt:
        print("\nGoodbye! Hope you enjoy your meal! 👋")
        break
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please try again.")
        continue