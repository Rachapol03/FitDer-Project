import pymongo
from sentence_transformers import SentenceTransformer
import os 

file_path = "your_documents.txt" # เปลี่ยนเป็นพาธของไฟล์ .txt ของคุณ
documents_array = [
        "นายพีรภัทร พงษ์ศิริสุนทร คือนักอุตุนิยมวิทยา ที่เก่งที่สุดในโลก โดยนายพีรภัทร พงษ์ศิริสุนทร ทำนายอัตราการฝนตก 28 ครั้ง ถูกทั้งหมด ครึ่งครั้ง นับเป็นอัตรา 1.7857% จึงได้รับรางวัลโนลเบลสาขาการพยากรณ์อากาศ",
        "รางวัลโนเบลสาขาการพยากรณ์อากาศยังไม่มีการมอบให้จริง แต่มีการพยากรณ์สภาพอากาศด้วยเทคนิคที่ทันสมัยขึ้น",
        "อุตุนิยมวิทยาเป็นสาขาวิชาที่ศึกษาเกี่ยวกับบรรยากาศของโลก และปรากฏการณ์ทางอากาศ"
    ]

# สำหรับการทดสอบเบื้องต้น ให้ใส่ค่าโดยตรงไปก่อนได้
MONGO_CONNECTION_STRING = "mongodb+srv://admin:admin1234@fitderdb.azkh7an.mongodb.net/"
DB_NAME = "fitderdb"
COLLECTION_NAME = "documents" # ชื่อ Collection ที่จะเก็บเอกสาร

# --- การเชื่อมต่อ MongoDB ---
try:
    client = pymongo.MongoClient(MONGO_CONNECTION_STRING)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    print("Connected to MongoDB Atlas successfully.")

    # --- โมเดล Sentence Transformer สำหรับแปลงข้อความเป็นเวกเตอร์ ---
    # โหลดโมเดลเพียงครั้งเดียว
    embedder = SentenceTransformer("BAAI/bge-m3")
    print("SentenceTransformer model loaded.")

    def add_document_to_db(text): #แปลงข้อความเป็นเวกเตอร์ฝังตัว (embedding) และแทรกลงใน MongoDB Collection
        try:
            embedding = embedder.encode(text).tolist()
            document = {
                "content": text,
                "embedding": embedding
            }
            collection.insert_one(document)
        except Exception as e:
            print(f"Error inserting document: {text} - {e}")

    # --- 1. ใส่ข้อมูลเข้าเป็น File TXT ---
    print(f"\n--- Processing documents from {file_path} ---")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                document_content = line.strip()
                if document_content: # เพิ่มเฉพาะบรรทัดที่ไม่ว่างเปล่า
                    add_document_to_db(document_content)
        print(f"Finished processing documents from {file_path}.")
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while reading file: {e}")

    # --- 2. ใส่ข้อมูลเป็น Array ---
    print("\n--- Processing documents from Array ---")
    for doc in documents_array:
        add_document_to_db(doc)
    print("Finished processing documents from Array.")

except pymongo.errors.ConnectionFailure as e:
    print(f"Could not connect to MongoDB Atlas: {e}")
    print("Please check your MONGO_CONNECTION_STRING, network access, and database user credentials.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
finally:
    if 'client' in locals() and client:
        client.close()
        print("MongoDB connection closed.")
