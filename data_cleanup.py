from datetime import datetime, timedelta
from data_storage import DataStorage
import time
import threading

class DataCleanup:
    def __init__(self, storage):
        self.storage = storage
        self.running = True
        self.cleanup_thread = threading.Thread(target=self.cleanup_loop)
        self.cleanup_thread.daemon = True

    def start(self):
        self.cleanup_thread.start()

    def stop(self):
        self.running = False
        self.cleanup_thread.join()

    def cleanup_loop(self):
        while self.running:
            self.cleanup_unused_data(hours=4)  # Cleanup unused data after 4 hours
            self.cleanup_old_data(hours=12)    # Cleanup all data after 12 hours
            time.sleep(3600)  # Check every hour

    def cleanup_unused_data(self, hours=4):
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=hours)
        
        # Find documents that haven't been accessed in 4 hours
        query = {
            "last_accessed": {
                "$lt": cutoff_time.isoformat()
            }
        }
        
        # Delete the documents
        self.storage.collection.delete_many(query)

    def cleanup_old_data(self, hours=12):
        current_time = datetime.now()
        cutoff_time = current_time - timedelta(hours=hours)
        
        # Find documents older than 12 hours
        query = {
            "timestamp": {
                "$lt": cutoff_time.isoformat()
            }
        }
        
        # Delete the documents
        self.storage.collection.delete_many(query)