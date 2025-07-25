# from sentence_transformers import SentenceTransformer
# import psycopg2
# import ollama

# def query_post(q_text, conn, k=3): # หาความเหมือนของ vector
#     embed = SentenceTransformer("BAAI/bge-m3") # ตัวแปลง text to vec
#     query_em = embed.encode(q_text).tolist()

#     cur = conn.cursor()
#     query_em_str = "[" + ", ".join(map(str, query_em)) + "]" 

#     sql_query = """
#         SELECT content, embedding <=> %s::vector AS similarity_score
#         FROM documents
#         ORDER BY similarity_score ASC
#         LIMIT %s
#     """

#     cur.execute(sql_query, (query_em_str, k))
#     results = cur.fetchall()
#     cur.close()
#     #conn.close()
#     return results

# def gen_res(query_text, conn): # ยัด LLM
#     ret_docs = query_post(query_text, conn)
#     context = "\n".join([doc[0] for doc in ret_docs])
#     prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query_text}"

#     response = ollama.chat(model="llama3.2", messages=[
#         {"role": "system", "content": "You are a trainer fitness."},
#         {"role": "user", "content": prompt}
#     ])
#     return response["message"]["context"]
from dotenv import load_dotenv
load_dotenv() # โหลด environment variables จากไฟล์ .env

from sentence_transformers import SentenceTransformer
import psycopg2
import os
import google.generativeai as genai

# --- การตั้งค่า Gemini API ---
try:
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except KeyError:
    raise ValueError("GOOGLE_API_KEY environment variable not set. Please set it to proceed with Gemini API.")

# gemini_model = genai.GenerativeModel('gemini-pro')
gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')

# --- ฟังก์ชันค้นหาเอกสารที่คล้ายกัน (Retrieval) ---
def query_post(q_text, conn, k=3):
    embed = SentenceTransformer("BAAI/bge-m3")
    query_em = embed.encode(q_text).tolist()

    cur = conn.cursor()
    query_em_str = "[" + ", ".join(map(str, query_em)) + "]"

    sql_query = """
        SELECT content, embedding <=> %s::vector AS similarity_score
        FROM documents
        ORDER BY similarity_score ASC
        LIMIT %s
    """

    try:
        cur.execute(sql_query, (query_em_str, k))
        results = cur.fetchall()
    except Exception as e:
        # print(f"Error querying database: {e}")
        results = []
    finally:
        cur.close()
    return results

# # --- ฟังก์ชันสร้างคำตอบโดยใช้ LLM (Generation) ---
# def gen_res(query_text, conn):
#     retrieved_docs = query_post(query_text, conn)

#     if not retrieved_docs:
#         # กรณีไม่พบเอกสารที่เกี่ยวข้อง:
#         # 1. แจ้งให้ทราบว่าไม่พบข้อมูลเฉพาะ
#         # 2. ปรับ Prompt ให้ Gemini ใช้ความรู้ทั่วไปในการตอบ
#         # print("No specific context from documents found. Gemini will use its general knowledge to answer.")
#         prompt = (
#             f"You are a fitness trainer and an expert in providing concise and helpful advice. "
#             f"You were unable to find specific context in your database related to the user's question. "
#             f"Please answer the following question using your general knowledge, and try to be as helpful and informative as possible.\n\n"
#             f"Question: {query_text}"
#         )
#     else:
#         # กรณีพบเอกสารที่เกี่ยวข้อง:
#         # 1. รวมเนื้อหาจากเอกสารที่ดึงมาเป็น context
#         # 2. ปรับ Prompt ให้ Gemini ตอบตาม context เท่านั้น
#         context = "\n".join([doc[0] for doc in retrieved_docs])
#         prompt = (
#             f"You are a fitness trainer and an expert in providing concise and helpful advice. "
#             f"Answer the following question based **ONLY** on the provided context. "
#             f"If the answer cannot be found in the context, state that you cannot answer from the given information.\n\n"
#             f"Context:\n{context}\n\n"
#             f"Question: {query_text}"
#         )

#     try:
#         response = gemini_model.generate_content(prompt)
#         return response.text
#     except Exception as e:
#         # print(f"Error calling Gemini API: {e}")
#         print(f"DEBUG: Error calling Gemini API: {e}") # เพิ่ม DEBUG: เพื่อให้เห็นชัดเจน
#         return "Sorry, I could not generate a response at this time due to an API error."
    
# --- ฟังก์ชันสร้างคำตอบโดยใช้ LLM (Generation) ---
def gen_res(query_text, conn):
    retrieved_docs_with_score = query_post(query_text, conn)

    # กำหนด threshold สำหรับ score ที่ถือว่า "เกี่ยวข้อง"
    # ค่านี้อาจต้องปรับตามการทดลองของคุณ (0.5 เป็นจุดเริ่มต้นที่ดี)
    # ค่าที่น้อยลงหมายถึงคล้ายกันมากขึ้น (เพราะ <=> คือระยะห่าง)
    similarity_threshold = 0.5 # หรืออาจจะ 0.6, 0.7 แล้วแต่ว่าข้อมูลคุณเป็นไง

    # กรองเอกสารที่เกี่ยวข้องจริงๆ ตาม threshold
    meaningful_docs = [doc[0] for doc in retrieved_docs_with_score if doc[1] < similarity_threshold]

    if not meaningful_docs: # ถ้าไม่มีเอกสารที่เกี่ยวข้องจริงๆ
        prompt = (
            f"You are a fitness trainer and an expert in providing concise and helpful advice. "
            f"The following question is outside the specific knowledge of your database. " # ปรับ wording นิดหน่อย
            f"Please answer the following question using your general knowledge, and try to be as helpful and informative as possible.\n\n"
            f"Question: {query_text}"
        )
    else: # ถ้ามีเอกสารที่เกี่ยวข้อง
        print("DEBUG: Context found. Answering based on context.")
        context = "\n".join(meaningful_docs)
        prompt = (
            f"You are a fitness trainer and an expert in providing concise and helpful advice. "
            f"Answer the following question based **ONLY** on the provided context. "
            f"If the answer cannot be found in the context, state that you cannot answer from the given information.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query_text}"
        )

    try:
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"DEBUG: Error calling Gemini API: {e}")
        return "Sorry, I could not generate a response at this time due to an API error."