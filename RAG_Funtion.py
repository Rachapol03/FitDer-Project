from sentence_transformers import SentenceTransformer
import psycopg2
import ollama

def query_post(q_text, conn, k=3): # หาความเหมือนของ vector
    embed = SentenceTransformer("BAAI/bge-m3") # ตัวแปลง text to vec
    query_em = embed.encode(q_text).tolist()

    cur = conn.cursor()
    query_em_str = "[" + ", ".join(map(str, query_em)) + "]" 

    sql_query = """
        SELECT content, embedding <=> %s::vector AS similarity_score
        FROM documents
        ORDER BY similarity_score ASC
        LIMIT %s
    """

    cur.execute(sql_query, (query_em_str, k))
    results = cur.fetchall()
    cur.close()
    #conn.close()
    return results

def gen_res(query_text, conn): # ยัด LLM
    ret_docs = query_post(query_text, conn)
    context = "\n".join([doc[0] for doc in ret_docs])
    prompt = f"Answer the question based on the following context:\n{context}\n\nQuestion: {query_text}"

    response = ollama.chat(model="llama3.2", messages=[
        {"role": "system", "content": "You are a trainer fitness."},
        {"role": "user", "content": prompt}
    ])
    return response["message"]["context"]
