# RFuntion.py
from dotenv import load_dotenv
load_dotenv()

import os
import pymongo
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- การตั้งค่า Gemini API ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    logging.info("Gemini API configured successfully.")
except KeyError:
    logging.error("GOOGLE_API_KEY environment variable not set.")
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to proceed with Gemini API.")

gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- MongoDB Atlas Setup ---
MONGO_CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING")
DB_NAME_MONGO = os.environ.get("MONGO_DB_NAME", "fitderdb")
COLLECTION_NAME = os.environ.get("MONGO_COLLECTION_NAME", "documents")
VECTOR_INDEX_NAME = os.environ.get("MONGO_VECTOR_INDEX_NAME", "vector_search_index")

if not MONGO_CONNECTION_STRING:
    logging.error("MONGO_CONNECTION_STRING environment variable not set.")
    raise ValueError("MONGO_CONNECTION_STRING environment variable not set. Please set it in your .env file.")

try:
    client = pymongo.MongoClient(MONGO_CONNECTION_STRING)
    db_mongo = client[DB_NAME_MONGO]
    collection_mongo = db_mongo[COLLECTION_NAME]
    client.admin.command('ping')
    logging.info(f"Connected to MongoDB Atlas database: {DB_NAME_MONGO}, collection: {COLLECTION_NAME}")
except pymongo.errors.ConnectionFailure as e:
    logging.error(f"MongoDB connection failed: {e}")
    raise

embedder = SentenceTransformer("BAAI/bge-m3")
logging.info("SentenceTransformer model loaded for embedding queries.")

# --- MongoDB Atlas Retrieval Function ---
def query_mongodb_atlas_vector_search(query_text, k=3):
    query_embedding = embedder.encode(query_text).tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100,
                "limit": k,
                "index": VECTOR_INDEX_NAME
            }
        },
        {
            "$project": {
                "_id": 0,
                "content": 1,
                "score": { "$meta": "vectorSearchScore" }
            }
        }
    ]

    try:
        results = list(collection_mongo.aggregate(pipeline))
        logging.info(f"MongoDB Vector Search Results for '{query_text}': {results}") # เพิ่มบรรทัดนี้เพื่อดูค่า Score
        return results
    except Exception as e:
        logging.error(f"Error during MongoDB vector search: {e}")
        return []

# --- MongoDB Atlas Generation Function ---
def gen_res_mongodb(query_text):
    retrieved_docs_with_score = query_mongodb_atlas_vector_search(query_text)

    # ปรับค่า similarity_threshold ให้น้อยลง หรือเอาออกไปก่อนเพื่อดูค่า score
    similarity_threshold = 0.5 # เราจะลองปรับค่านี้
    meaningful_docs = [doc['content'] for doc in retrieved_docs_with_score if doc['score'] >= similarity_threshold]

    if not meaningful_docs:
        logging.info("No meaningful context found. Answering using general knowledge.")
        prompt = (
            f"You are a fitness trainer and a health expert who provides detailed and helpful advice. "
            f"No matter what language the user asks in, always answer in Thai. "
            f"Please answer the following question using your general knowledge. "
            f"Provide advice that is as helpful and comprehensive as possible.\n\n"
            f"Question: {query_text}"
        )
    else:
        logging.info("Context found. Answering based on context.")
        context = "\n".join(meaningful_docs)
        prompt = (
            f"You are a fitness trainer and an expert in providing detailed and helpful advice. "
            f"No matter what language you ask, always answer in Thai."
            f"Answer the following question based **ONLY** on the provided context. "
            f"Provide as much detail as possible from the context to fully answer the question. "
            f"If the answer cannot be found in the context, state that you cannot answer from the given information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query_text}"
        )

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        logging.error(f"Error calling Gemini API: {e}")
        return "ขออภัยครับ ตอนนี้ไม่สามารถสร้างคำตอบได้เนื่องจากมีข้อผิดพลาดทางเทคนิค"

def close_mongodb_connection():
    if 'client' in globals() and client:
        client.close()
        logging.info("MongoDB connection closed.")