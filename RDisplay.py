from RFuntion import gen_res_mongodb, close_mongodb_connection

while True:
    user_input = input("กรอกคำถาม หากหมดคำถามแล้วพิมพ์ 'end' :")
    if user_input.lower() == "end":
        break
    else:
        # Call the MongoDB-specific generation function
        response_from_llm = gen_res_mongodb(user_input)
        print("\n--- AI Answer ---")
        print(response_from_llm)

# ปิดการเชื่อมต่อ MongoDB เมื่อโปรแกรมหลักจบ
close_mongodb_connection()
