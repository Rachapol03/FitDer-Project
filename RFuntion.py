# NewRag_Funtion.py (Modified for Debugging)
from dotenv import load_dotenv
load_dotenv() # โหลด environment variables จากไฟล์ .env

import os
import pymongo
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

# --- การตั้งค่า Gemini API ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to proceed with Gemini API.")

gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- MongoDB Atlas Setup ---
MONGO_CONNECTION_STRING = os.environ.get("MONGO_CONNECTION_STRING")
DB_NAME_MONGO = os.environ.get("MONGO_DB_NAME", "fitderdb") # Renamed for clarity
COLLECTION_NAME = os.environ.get("MONGO_COLLECTION_NAME", "documents")
VECTOR_INDEX_NAME = os.environ.get("MONGO_VECTOR_INDEX_NAME", "vector_search_index")

if not MONGO_CONNECTION_STRING:
    raise ValueError("MONGO_CONNECTION_STRING environment variable not set. Please set it in your .env file.")

try:
    client = pymongo.MongoClient(MONGO_CONNECTION_STRING)
    db_mongo = client[DB_NAME_MONGO] # Using renamed variable
    collection_mongo = db_mongo[COLLECTION_NAME] # Using renamed variable
    # print(f"Connected to MongoDB Atlas database: {DB_NAME_MONGO}, collection: {COLLECTION_NAME}")
except pymongo.errors.ConnectionFailure as e:
    print(f"MongoDB connection failed: {e}")

embedder = SentenceTransformer("BAAI/bge-m3")
# print("SentenceTransformer model loaded for embedding queries.")

# --- MongoDB Atlas Retrieval Function ---
def query_mongodb_atlas_vector_search(query_text, k=3):
    query_embedding = embedder.encode(query_text).tolist()

    pipeline = [
        {
            "$vectorSearch": {
                "queryVector": query_embedding,
                "path": "embedding",
                "numCandidates": 100, # ลองเพิ่มเป็น 200 หรือ 500 ถ้ายังไม่เจอ
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

    results = list(collection_mongo.aggregate(pipeline)) # Use collection_mongo
        # print(f"DEBUG: MongoDB Vector Search Raw Results for '{query_text}': {results}") # *** เพิ่มบรรทัดนี้ ***
    return results

# --- MongoDB Atlas Generation Function ---
def gen_res_mongodb(query_text): # Renamed function for clarity
    retrieved_docs_with_score = query_mongodb_atlas_vector_search(query_text)
    
    # *** เปลี่ยนเครื่องหมายเปรียบเทียบจาก '>' เป็น '<' อย่างที่แนะนำไปแล้ว ***
    # ค่า score ยิ่งน้อยยิ่งคล้าย เพราะเป็นระยะห่าง
    similarity_threshold = 0.5 # ค่านี้ควรปรับตามผลลัพธ์จริงที่ได้จากการ Debug
    meaningful_docs = [doc['content'] for doc in retrieved_docs_with_score if doc['score'] > similarity_threshold]

    # print(f"DEBUG: Documents after threshold filter (score > {similarity_threshold}):") # *** เพิ่มบรรทัดนี้ ***

    if not meaningful_docs:
        prompt = (
            f"You are a fitness trainer and an expert in providing detailed and helpful advice. "
            f"No matter what language you ask, always answer in Thai."
            f"The following question is outside the specific knowledge of your database. "
            f"Provide as much detail as possible from the context to fully answer the question. "
            f"Please answer the following question using your general knowledge, and try to be as helpful and informative as possible.\n\n"
            f"Question: {query_text}"
        )
        # print("DEBUG: No meaningful context found. Answering using general knowledge.")
    else:
        # print("DEBUG: Context found. Answering based on context.")
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
        # print(f"DEBUG: Error calling Gemini API: {e}")
        return "Sorry, I could not generate a response at this time due to an API error."
    
def close_mongodb_connection():
    if 'client' in globals() and client:
        client.close()
        # print("MongoDB connection closed.")
