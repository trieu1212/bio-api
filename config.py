from dotenv import load_dotenv
import os
load_dotenv()

class Config:
    MONGO_URI = os.getenv("MONGO_URI")
    MONGO_DB_NAME = os.getenv("MONGO_DB_NAME")
    THRESHOLD = os.getenv("THRESHOLD")
    FACES_DIR = os.getenv("FACES_DIR")