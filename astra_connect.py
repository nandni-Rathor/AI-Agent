import os
from astrapy.db import AstraDB, AstraDBCollection  # Correct import
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

if not ASTRA_DB_API_ENDPOINT:
    raise ValueError("ASTRA_DB_API_ENDPOINT environment variable not set.")

if not ASTRA_DB_APPLICATION_TOKEN:
    raise ValueError("ASTRA_DB_APPLICATION_TOKEN environment variable not set.")

# Initialize AstraDB
db = AstraDB(
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace="default_keyspace"
)

COLLECTION_NAME = "my_first_collection"

def main():
    print("Attempting to connect to Astra DB...")

    try:
        # Test connection by getting collections
        collections = db.get_collections()
        print("✓ Successfully connected to Astra DB!")
        print(f"✓ Available collections: {collections}")
        print("\n=== Database Operations ===")

        # Create collection if it doesn't exist
        collection = db.create_collection(COLLECTION_NAME)
        print(f"✓ Collection '{COLLECTION_NAME}' created/accessed successfully")

        # Insert document
        print(f"\nTesting write operations - Inserting a document into '{COLLECTION_NAME}'...")
        doc = {"name": "example_user", "value": 123}
        insert_result = collection.insert_one(doc)
        inserted_id = insert_result['status']['insertedIds'][0]
        print(f"✓ Document inserted successfully with ID: {inserted_id}")

        # Retrieve document
        print("\nTesting read operations - Finding the inserted document...")
        found_doc = collection.find_one({"_id": inserted_id})
        print(f"✓ Document retrieved successfully: {found_doc}")

        print("\n✓ All database operations completed successfully!")

    except Exception as e:
        print(f"\n❌ Connection error: {e}")
        print("Please check your database credentials and try again.")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
