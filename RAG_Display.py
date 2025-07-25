import psycopg2
import RAG_Funtion
from RAG_Funtion import query_post, gen_res

conn = psycopg2.connect(
            dbname="fitderdb",
            user="admin",
            password="1234",
            host="localhost",
            port="5432")

while True:
    user_input = input("กรอกคำถาม หากหมดคำถามแล้วพิมพ์ 'end' :")
    if user_input == "end":
        break
    else:
        response_from_llm = gen_res(user_input, conn)
        print("\n--- AI Answer ---")
        print(response_from_llm)
        