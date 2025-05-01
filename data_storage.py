import os
from datetime import datetime, timedelta
from astrapy.db import AstraDB

class DataStorage:
    def __init__(self, collection_name="user_data"):
        # Initialize AstraDB connection
        self.db = AstraDB(
            api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
            token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
            namespace="default_keyspace"
        )
        self.collection = self.db.create_collection(collection_name)

    def store_input(self, user_input):
        current_time = datetime.now()
        deletion_time = current_time + timedelta(hours=12)
        unused_deletion_time = current_time + timedelta(hours=4)
        
        timestamp = current_time.isoformat()
        
        document = {
            '_id': timestamp,
            'input': user_input,
            'timestamp': timestamp,
            'last_accessed': timestamp,
            'deletion_time': deletion_time.isoformat(),
            'unused_deletion_time': unused_deletion_time.isoformat()
        }
        
        self.collection.insert_one(document)
        
        print("\nStorage Information:")
        print(f"Stored at: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Will be deleted if unused for 4 hours at: {unused_deletion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Will be permanently deleted at: {deletion_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        return timestamp

    def get_input(self, timestamp):
        entry = self.collection.find_one({"_id": timestamp})
        if entry:
            current_time = datetime.now()
            
            # Update last accessed time and unused deletion time
            update_data = {
                "last_accessed": current_time.isoformat(),
                "unused_deletion_time": (current_time + timedelta(hours=4)).isoformat()
            }
            self.collection.update_one({"_id": timestamp}, {"$set": update_data})
            
            # Calculate remaining time
            deletion_time = datetime.fromisoformat(entry['deletion_time'])
            time_left = deletion_time - current_time
            
            print("\nStorage Status:")
            print(f"Last accessed: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Time until permanent deletion: {self.format_timedelta(time_left)}")
            
            return entry['input']
        return None

    def format_timedelta(self, td):
        hours = td.seconds // 3600 + td.days * 24
        minutes = (td.seconds % 3600) // 60
        return f"{hours} hours and {minutes} minutes"

    def get_storage_status(self, timestamp):
        entry = self.collection.find_one({"_id": timestamp})
        if entry:
            current_time = datetime.now()
            
            created_time = datetime.fromisoformat(entry['timestamp'])
            deletion_time = datetime.fromisoformat(entry['deletion_time'])
            unused_deletion_time = datetime.fromisoformat(entry['unused_deletion_time'])
            
            time_until_deletion = deletion_time - current_time
            time_until_unused_deletion = unused_deletion_time - current_time
            
            print("\nDetailed Storage Status:")
            print(f"Created at: {created_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Last accessed: {datetime.fromisoformat(entry['last_accessed']).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Time until unused deletion: {self.format_timedelta(time_until_unused_deletion)}")
            print(f"Time until permanent deletion: {self.format_timedelta(time_until_deletion)}")
            
            return True
        return False