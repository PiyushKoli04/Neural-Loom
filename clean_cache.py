import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/neuralloom")

client = MongoClient(MONGO_URI)
db = client.neuralloom

res1 = db.lessons.delete_many({"content": {"$regex": r"\[Gemini error:|\[AI error:"}})
print(f"Deleted {res1.deleted_count} cached lessons with API errors.")

res2 = db.quizzes.delete_many({"questions.question": {"$regex": "Error generating quiz"}})
print(f"Deleted {res2.deleted_count} cached quizzes with API errors.")
