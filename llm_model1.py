from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.utilities import SQLDatabase

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyB9gtjY5CAqkFy3xWP-pmiyS0QEZO_lXRI",
)

result = llm.invoke("capital of france")
print(result.content)

db_user="root"
db_password="RAHUL5rahul@"
db_host="localhost"
db_name="atliq_tshirts"
db_port=3600

db = SQLDatabase.from_uri(
    f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}",
    sample_rows_in_table_info=3
)
print(db.table_info)

