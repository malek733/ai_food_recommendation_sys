import os
from supabase import create_client
from dotenv import load_dotenv
import random
from faker import Faker

load_dotenv()

# Initialize Supabase and Faker
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
fake = Faker()

def create_fake_data():
    print("🍽️ Creating fake GoFood data...")
    
    # Create restaurants
    restaurants_data = [
        {"name": "Tokyo Sushi", "cuisine_type": "Japanese"},
        {"name": "Mama Mia Pizzeria", "cuisine_type": "Italian"},
        {"name": "Burger Barn", "cuisine_type": "American"},
        {"name": "Taco Fiesta", "cuisine_type": "Mexican"},
        {"name": "Green Leaf Cafe", "cuisine_type": "Vegetarian"},
    ]
    
    restaurant_ids = []
    for restaurant in restaurants_data:
        result = supabase.table("restaurants").insert(restaurant).execute()
        restaurant_ids.append(result.data[0]["id"])
        print(f"✅ Created restaurant: {restaurant['name']}")
    
    # Create menu items for each restaurant
    menu_items_data = {
        "Tokyo Sushi": [
            {"name": "California Roll", "description": "Crab, avocado, cucumber", "price": 12.99, "category": "main", "calories": 320, "tags": ["seafood", "popular"], "preparation_time": 10},
            {"name": "Spicy Tuna Roll", "description": "Fresh tuna with spicy mayo", "price": 14.99, "category": "main", "calories": 280, "tags": ["spicy", "seafood"], "preparation_time": 12},
            {"name": "Salmon Nigiri", "description": "Fresh salmon over rice", "price": 9.99, "category": "main", "calories": 180, "tags": ["seafood", "fresh"], "preparation_time": 8},
        ],
        "Mama Mia Pizzeria": [
            {"name": "Margherita Pizza", "description": "Tomato, mozzarella, basil", "price": 16.99, "category": "main", "calories": 850, "tags": ["vegetarian", "classic"], "preparation_time": 15},
            {"name": "Pepperoni Pizza", "description": "Pepperoni, cheese, tomato sauce", "price": 18.99, "category": "main", "calories": 920, "tags": ["meat", "popular"], "preparation_time": 18},
        ],
        "Burger Barn": [
            {"name": "Classic Burger", "description": "Beef patty, lettuce, tomato", "price": 10.99, "category": "main", "calories": 650, "tags": ["beef", "classic"], "preparation_time": 12},
            {"name": "BBQ Bacon Burger", "description": "Beef patty, bacon, BBQ sauce", "price": 13.99, "category": "main", "calories": 850, "tags": ["beef", "bacon"], "preparation_time": 15},
        ],
        "Taco Fiesta": [
            {"name": "Beef Tacos", "description": "Seasoned ground beef in crispy shells", "price": 8.99, "category": "main", "calories": 420, "tags": ["beef", "mexican"], "preparation_time": 8},
            {"name": "Chicken Quesadilla", "description": "Grilled chicken and cheese", "price": 9.99, "category": "main", "calories": 520, "tags": ["chicken", "cheese"], "preparation_time": 10},
        ],
        "Green Leaf Cafe": [
            {"name": "Avocado Salad", "description": "Mixed greens with avocado", "price": 9.99, "category": "main", "calories": 320, "tags": ["vegetarian", "healthy"], "preparation_time": 7},
            {"name": "Quinoa Bowl", "description": "Quinoa with roasted vegetables", "price": 12.99, "category": "main", "calories": 450, "tags": ["vegetarian", "healthy"], "preparation_time": 9},
        ]
    }
    
    menu_item_ids = []
    for restaurant_name, items in menu_items_data.items():
        # Get restaurant ID
        restaurant_result = supabase.table("restaurants").select("id").eq("name", restaurant_name).execute()
        if restaurant_result.data:
            restaurant_id = restaurant_result.data[0]["id"]
            
            for item in items:
                item["restaurant_id"] = restaurant_id
                result = supabase.table("menu_items").insert(item).execute()
                menu_item_ids.append(result.data[0]["id"])
                print(f"✅ Created menu item: {item['name']}")
    
    # Create fake users
    users_data = []
    for _ in range(10):
        user = {
            "email": fake.email(),
            "first_name": fake.first_name(),
            "last_name": fake.last_name()
        }
        users_data.append(user)
    
    user_ids = []
    for user in users_data:
        result = supabase.table("users").insert(user).execute()
        user_ids.append(result.data[0]["id"])
        print(f"✅ Created user: {user['email']}")
    
    # Create fake orders
    for user_id in user_ids[:5]:  # First 5 users make orders
        restaurant_id = random.choice(restaurant_ids)
        
        # Get 2-3 random menu items from this restaurant
        menu_items_result = supabase.table("menu_items").select("id, price").eq("restaurant_id", restaurant_id).execute()
        if menu_items_result.data:
            selected_items = random.sample(menu_items_result.data, min(3, len(menu_items_result.data)))
            
            total_amount = sum(item["price"] for item in selected_items)
            
            # Create order
            order_result = supabase.table("orders").insert({
                "user_id": user_id,
                "restaurant_id": restaurant_id,
                "total_amount": total_amount,
                "status": "completed"
            }).execute()
            
            order_id = order_result.data[0]["id"]
            
            # Create order items
            for item in selected_items:
                supabase.table("order_items").insert({
                    "order_id": order_id,
                    "menu_item_id": item["id"],
                    "quantity": random.randint(1, 2)
                }).execute()
            
            print(f"✅ Created order for user {user_id} at restaurant {restaurant_id}")
    
    print("🎉 Fake data creation completed!")

if __name__ == "__main__":
    create_fake_data()