import os
from supabase import create_client
from dotenv import load_dotenv
from collections import defaultdict

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("[ERROR] Missing SUPABASE_URL or SUPABASE_KEY in .env")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def find_duplicates(table_name, unique_fields):
    """
    Find duplicate entries in a table based on unique fields.

    Args:
        table_name (str): Name of the table to check
        unique_fields (list): List of fields that should be unique together

    Returns:
        dict: Dictionary of duplicate groups
    """
    print(f"[INFO] Checking for duplicates in {table_name}...")

    # Fetch all data from the table
    response = supabase.table(table_name).select("*").execute()

    if not response.data:
        print(f"[INFO] No data found in {table_name}")
        return {}

    rows = response.data
    print(f"[INFO] Found {len(rows)} rows in {table_name}")

    # Group by unique fields
    groups = defaultdict(list)

    for row in rows:
        # Create a key from the unique fields
        key_parts = []
        for field in unique_fields:
            value = row.get(field, "")
            if isinstance(value, dict):
                # Handle JSONB fields - use English text if available
                value = value.get("en", str(value))
            key_parts.append(str(value).lower().strip())

        key = " | ".join(key_parts)
        groups[key].append(row)

    # Find groups with more than one entry (duplicates)
    duplicates = {key: entries for key, entries in groups.items() if len(entries) > 1}

    if duplicates:
        print(f"[WARNING] Found {len(duplicates)} groups of duplicates in {table_name}")
        for key, entries in duplicates.items():
            print(f"  - Key '{key}': {len(entries)} duplicates")
    else:
        print(f"[INFO] No duplicates found in {table_name}")

    return duplicates

def remove_duplicates(table_name, duplicates, keep_strategy="first"):
    """
    Remove duplicate entries, keeping one based on strategy.

    Args:
        table_name (str): Name of the table
        duplicates (dict): Dictionary of duplicate groups
        keep_strategy (str): Strategy for which duplicate to keep ("first", "last", "newest", "oldest")
    """
    total_removed = 0

    for key, entries in duplicates.items():
        # Determine which entry to keep
        if keep_strategy == "first":
            keep_entry = entries[0]
            remove_entries = entries[1:]
        elif keep_strategy == "last":
            keep_entry = entries[-1]
            remove_entries = entries[:-1]
        elif keep_strategy == "newest":
            # Keep the one with latest created_at
            keep_entry = max(entries, key=lambda x: x.get("created_at", ""))
            remove_entries = [e for e in entries if e["id"] != keep_entry["id"]]
        elif keep_strategy == "oldest":
            # Keep the one with earliest created_at
            keep_entry = min(entries, key=lambda x: x.get("created_at", ""))
            remove_entries = [e for e in entries if e["id"] != keep_entry["id"]]
        else:
            print(f"[ERROR] Unknown keep_strategy: {keep_strategy}")
            continue

        print(f"[INFO] For key '{key}': keeping ID {keep_entry['id']}, removing {len(remove_entries)} entries")

        # Remove duplicate entries
        for entry in remove_entries:
            try:
                supabase.table(table_name).delete().eq("id", entry["id"]).execute()
                total_removed += 1
                print(f"  - Removed ID: {entry['id']}")
            except Exception as e:
                print(f"  - Error removing ID {entry['id']}: {e}")

    print(f"[SUCCESS] Removed {total_removed} duplicate entries from {table_name}")
    return total_removed

def clean_all_tables():
    """
    Clean duplicates from all relevant tables.
    """
    # Define tables and their unique field combinations
    table_configs = {
        "restaurants": {
            "unique_fields": ["name"],  # JSONB field with en/ar keys
            "keep_strategy": "newest"
        },
        "categories": {
            "unique_fields": ["restaurant_id", "name"],  # JSONB field
            "keep_strategy": "newest"
        },
        "menu_items": {
            "unique_fields": ["category_id", "name"],  # JSONB field
            "keep_strategy": "newest"
        },
        "menu_item_options": {
            "unique_fields": ["menu_item_id", "name"],  # JSONB field
            "keep_strategy": "newest"
        },
        "option_choices": {
            "unique_fields": ["option_id", "name"],  # JSONB field
            "keep_strategy": "newest"
        },
        "locations": {
            "unique_fields": ["name"],  # JSONB field
            "keep_strategy": "newest"
        },
        "profiles": {
            "unique_fields": ["email"],
            "keep_strategy": "newest"
        },
        "users": {
            "unique_fields": ["email"],
            "keep_strategy": "newest"
        }
    }

    total_duplicates_removed = 0

    for table_name, config in table_configs.items():
        print(f"\n{'='*50}")
        print(f"Processing table: {table_name}")
        print(f"{'='*50}")

        # Find duplicates
        duplicates = find_duplicates(table_name, config["unique_fields"])

        if duplicates:
            # Remove duplicates
            removed = remove_duplicates(table_name, duplicates, config["keep_strategy"])
            total_duplicates_removed += removed
        else:
            print(f"[INFO] No duplicates to remove from {table_name}")

    print(f"\n{'='*50}")
    print(f"[SUMMARY] Total duplicates removed: {total_duplicates_removed}")
    print(f"{'='*50}")

if __name__ == "__main__":
    print("Starting Supabase duplicate cleanup...")
    clean_all_tables()
    print("Duplicate cleanup completed!")