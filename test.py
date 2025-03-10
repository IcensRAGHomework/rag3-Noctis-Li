import student_assignment
import datetime

# print(student_assignment.generate_hw01(debug=True))

# result = student_assignment.generate_hw02(
#     question="我想要找有關茶餐點的店家",
#     city=["宜蘭縣", "新北市"],
#     store_type=["美食"],
#     start_date=datetime.datetime(2024, 4, 1),
#     end_date=datetime.datetime(2024, 5, 1)
# )
# print(result)
# # 輸出格式：['茶香小館', '古早味茶坊', ...]（最多10個符合條件的名稱）


result = student_assignment.generate_hw03(
    question="找南投縣的田媽媽餐廳，招牌是蕎麥麵",
    store_name="耄饕客棧",
    new_store_name="田媽媽（耄饕客棧）",
    city=["南投縣"],
    store_type=["美食"]
)

print(result)
# 輸出格式：['田媽媽社區餐廳', '田媽媽（耄饕客棧）', ...]