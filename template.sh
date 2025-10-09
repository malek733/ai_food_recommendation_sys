# Root files
touch .env requirements.txt main.py README.md

# Create src structure
mkdir -p src/config src/embeddings src/data src/utils

# Add __init__.py files
touch src/__init__.py
touch src/config/__init__.py
touch src/embeddings/__init__.py
touch src/data/__init__.py
touch src/utils/__init__.py

# Create the Python files
touch src/config/supabase_client.py
touch src/embeddings/generate_embeddings.py
touch src/embeddings/retrieve_embeddings.py
touch src/data/sample_data.py
touch src/utils/chunk_text.py

echo "✅ Project structure created successfully!"