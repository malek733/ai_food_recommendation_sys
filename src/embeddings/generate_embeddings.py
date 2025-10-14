import os
from supabase import create_client
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# =========================
# 1️⃣ Environment & Client
# =========================
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("[ERROR] Missing SUPABASE_URL or SUPABASE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# =========================
# 2️⃣ Utility to flatten JSONB
# =========================
def parse_jsonb(value):
    if isinstance(value, dict):
        return value.get("en") or value.get("ar") or str(value)
    return str(value) if value else ""

# =========================
# 3️⃣ Fetch any table
# =========================
def fetch_table(table_name):
    """Fetch all rows from a Supabase table."""
    resp = supabase.table(table_name).select("*").execute()
    print(f"[INFO] Fetched {len(resp.data)} rows from {table_name}")
    return resp.data

# =========================
# 4️⃣ Build and Chunk Documents
# =========================
def build_and_chunk_documents(table_name, rows, chunk_size=500, chunk_overlap=50):
    """Create LangChain Documents for a table with JSONB flattening and chunking."""
    base_docs = []

    for row in rows:
        text_lines = []
        metadata = {"table": table_name}

        # Special handling for menu_items to include restaurant information
        if table_name == "menu_items":
            # Add restaurant information through category relationship
            category_id = row.get("category_id")
            if category_id:
                try:
                    # Fetch category and restaurant info
                    category_resp = supabase.table("categories").select("*, restaurants(*)").eq("id", category_id).execute()
                    if category_resp.data:
                        category = category_resp.data[0]
                        restaurant = category.get("restaurants", {})

                        # Add restaurant info to metadata
                        if restaurant:
                            restaurant_name = parse_jsonb(restaurant.get("name", ""))
                            metadata["restaurant_id"] = restaurant.get("id")
                            metadata["restaurant_name"] = restaurant_name
                            text_lines.append(f"restaurant_id: {restaurant.get('id')}")
                            text_lines.append(f"restaurant_name: {restaurant_name}")

                        # Add category info to metadata
                        category_name = parse_jsonb(category.get("name", ""))
                        metadata["category_name"] = category_name
                        text_lines.append(f"category_name: {category_name}")

                except Exception as e:
                    print(f"[WARNING] Could not fetch restaurant info for menu_item {row.get('id')}: {e}")

        # Process regular fields
        for key, value in row.items():
            parsed_value = parse_jsonb(value)
            text_lines.append(f"{key}: {parsed_value}")
            metadata[key] = parsed_value

        text_content = "\n".join(text_lines)
        base_docs.append(Document(page_content=text_content, metadata=metadata))

    print(f"[INFO] Prepared {len(base_docs)} base documents for {table_name}")

    # ✅ Chunking step
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunked_docs = splitter.split_documents(base_docs)

    print(f"[INFO] After chunking: {len(chunked_docs)} chunks for {table_name}")
    return chunked_docs

# =========================
# 5️⃣ Create and persist embeddings
# =========================
def create_chroma_db(documents, persist_dir="./chroma_db"):
    """Generate embeddings and store them in Chroma."""
    print("[INFO] Creating embeddings with HuggingFace...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print(f"[INFO] Saving Chroma DB to {persist_dir}")
    db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_dir
    )

    if hasattr(db, "persist"):
        db.persist()

    print("[SUCCESS] Chroma DB created and saved successfully!")

# =========================
# 6️⃣ Main entry point
# =========================
if __name__ == "__main__":
    tables = [
        "restaurants",
        "categories",
        "menu_items",
        "menu_item_options",
        "option_choices",
        "orders",
        "order_items",
        "users",
        "profiles",
        "locations",
        "site_settings",
        "user_invitations",
        "user_restaurant_assignments",
        "payments"
    ]

    all_docs = []
    for table in tables:
        rows = fetch_table(table)
        docs = build_and_chunk_documents(table, rows)
        all_docs.extend(docs)

    create_chroma_db(all_docs)
