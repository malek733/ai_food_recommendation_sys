import os
from supabase import create_client
from dotenv import load_dotenv
import random
from faker import Faker

load_dotenv()

# Initialize Supabase and Faker
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
)
fake = Faker()

def create_fake_data():
    print("Creating fake GoFood data...")

    # Create locations first
    locations_data = [
        {"name": {"en": "Doha", "ar": "الدوحة"}},
        {"name": {"en": "Al Rayyan", "ar": "الريان"}},
    ]

    location_ids = []
    for location in locations_data:
        result = supabase.table("locations").insert(location).execute()
        location_ids.append(result.data[0]["id"])
        print(f"Created location: {location['name']['en']}")

    # Create restaurants
    restaurants_data = [
        {"name": {"en": "Tokyo Sushi", "ar": "توكيو سوشي"}, "location_id": location_ids[0]},
        {"name": {"en": "Mama Mia Pizzeria", "ar": "ماما ميا بيتزريا"}, "location_id": location_ids[0]},
        {"name": {"en": "Burger Barn", "ar": "برجر بارن"}, "location_id": location_ids[0]},
        {"name": {"en": "Taco Fiesta", "ar": "تاكو فييستا"}, "location_id": location_ids[1]},
        {"name": {"en": "Green Leaf Cafe", "ar": "غرين ليف كافيه"}, "location_id": location_ids[1]},
    ]

    restaurant_ids = []
    for restaurant in restaurants_data:
        result = supabase.table("restaurants").insert(restaurant).execute()
        restaurant_ids.append(result.data[0]["id"])
        print(f"Created restaurant: {restaurant['name']['en']}")

    # Create categories
    categories_data = [
        {"name": {"en": "Sushi", "ar": "سوشي"}, "restaurant_id": restaurant_ids[0]},
        {"name": {"en": "Pizza", "ar": "بيتزا"}, "restaurant_id": restaurant_ids[1]},
        {"name": {"en": "Burgers", "ar": "برجر"}, "restaurant_id": restaurant_ids[2]},
        {"name": {"en": "Tacos", "ar": "تاكو"}, "restaurant_id": restaurant_ids[3]},
        {"name": {"en": "Salads", "ar": "سلطات"}, "restaurant_id": restaurant_ids[4]},
    ]

    category_ids = []
    for category in categories_data:
        result = supabase.table("categories").insert(category).execute()
        category_ids.append(result.data[0]["id"])
        print(f"Created category: {category['name']['en']}")
    
    # Create menu items for each category
    menu_items_data = {
        "Sushi": [
            {"name": {"en": "California Roll", "ar": "كاليفورنيا رول"}, "description": {"en": "Crab, avocado, cucumber", "ar": "سلطعون، أفوكادو، خيار"}, "price": 12.99, "preparation_time": 10},
            {"name": {"en": "Spicy Tuna Roll", "ar": "سبايسي تونا رول"}, "description": {"en": "Fresh tuna with spicy mayo", "ar": "تونة طازجة مع مايو حار"}, "price": 14.99, "preparation_time": 12},
            {"name": {"en": "Salmon Nigiri", "ar": "سالمون نيجيري"}, "description": {"en": "Fresh salmon over rice", "ar": "سالمون طازج على الأرز"}, "price": 9.99, "preparation_time": 8},
        ],
        "Pizza": [
            {"name": {"en": "Margherita Pizza", "ar": "بيتزا مارغريتا"}, "description": {"en": "Tomato, mozzarella, basil", "ar": "طماطم، موزاريلا، ريحان"}, "price": 16.99, "preparation_time": 15},
            {"name": {"en": "Pepperoni Pizza", "ar": "بيتزا بيبروني"}, "description": {"en": "Pepperoni, cheese, tomato sauce", "ar": "بيبروني، جبنة، صلصة طماطم"}, "price": 18.99, "preparation_time": 18},
        ],
        "Burgers": [
            {"name": {"en": "Classic Burger", "ar": "برجر كلاسيكي"}, "description": {"en": "Beef patty, lettuce, tomato", "ar": "لحم بقري، خس، طماطم"}, "price": 10.99, "preparation_time": 12},
            {"name": {"en": "BBQ Bacon Burger", "ar": "برجر لحم مع بيكون"}, "description": {"en": "Beef patty, bacon, BBQ sauce", "ar": "لحم بقري، بيكون، صلصة باربيكيو"}, "price": 13.99, "preparation_time": 15},
        ],
        "Tacos": [
            {"name": {"en": "Beef Tacos", "ar": "تاكو لحم بقري"}, "description": {"en": "Seasoned ground beef in crispy shells", "ar": "لحم بقري متبل في قشور مقرمشة"}, "price": 8.99, "preparation_time": 8},
            {"name": {"en": "Chicken Quesadilla", "ar": "كيساديلا دجاج"}, "description": {"en": "Grilled chicken and cheese", "ar": "دجاج مشوي وجبنة"}, "price": 9.99, "preparation_time": 10},
        ],
        "Salads": [
            {"name": {"en": "Avocado Salad", "ar": "سلطة أفوكادو"}, "description": {"en": "Mixed greens with avocado", "ar": "خضروات مشكلة مع أفوكادو"}, "price": 9.99, "preparation_time": 7},
            {"name": {"en": "Quinoa Bowl", "ar": "كينوا بول"}, "description": {"en": "Quinoa with roasted vegetables", "ar": "كينوا مع خضروات مشوية"}, "price": 12.99, "preparation_time": 9},
        ],
        "Pizza": [
            {"name": {"en": "Tiramisu", "ar": "تيراميسو"}, "description": {"en": "Classic Italian dessert made of layers of coffee-soaked ladyfingers and whipped mascarpone cream, topped with cocoa powder", "ar": "حلوى إيطالية كلاسيكية مصنوعة من طبقات من بسكويت اللادي فينجر المنقوع بالقهوة وكريمة الماسكاربوني المخفوقة، مغطاة بمسحوق الكاكاو"}, "price": 13.00, "preparation_time": 5},
        ]
    }

    menu_item_ids = []
    for category_name, items in menu_items_data.items():
        # Get category ID
        category_result = supabase.table("categories").select("id").eq("name->>en", category_name).execute()
        if category_result.data:
            category_id = category_result.data[0]["id"]

            for item in items:
                item["category_id"] = category_id
                result = supabase.table("menu_items").insert(item).execute()
                menu_item_ids.append(result.data[0]["id"])
                print(f"Created menu item: {item['name']['en']}")
    
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
        print(f"Created user: {user['email']}")
    
    # Create fake orders
    for user_id in user_ids[:5]:  # First 5 users make orders
        # Get 2-3 random menu items
        menu_items_result = supabase.table("menu_items").select("id, price").eq("is_available", True).execute()
        if menu_items_result.data:
            selected_items = random.sample(menu_items_result.data, min(3, len(menu_items_result.data)))

            total_amount = sum(float(item["price"]) for item in selected_items)

            # Create order
            order_result = supabase.table("orders").insert({
                "user_id": None,
                "total_amount": total_amount,
                "status": "open",
                "customer_name": fake.name(),
                "customer_phone": fake.phone_number()
            }).execute()

            order_id = order_result.data[0]["id"]

            # Create order items
            for item in selected_items:
                supabase.table("order_items").insert({
                    "order_id": order_id,
                    "menu_item_id": item["id"],
                    "quantity": random.randint(1, 2),
                    "unit_price": float(item["price"]),
                    "total_price": float(item["price"]) * random.randint(1, 2)
                }).execute()

            print(f"Created order for user {user_id}")
    
    print("Fake data creation completed!")

if __name__ == "__main__":
    create_fake_data()