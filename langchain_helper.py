
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

#------model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key="AIzaSyBbt83r8dK5jp6T3oX8aP8nH28Ys2LkByc",
    temperature=.1
)

model=HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

vector_db_path="faiss_index"

def create_vector_db():
   
    loader=CSVLoader(file_path=r"C:\Users\LENOVO\OneDrive\Desktop\lc_project\codebasic\codebasics.csv",source_column='prompt')
    data=loader.load()
    vectordb=FAISS.from_documents(documents=data,embedding=model)
    vectordb.save_local(vector_db_path) 

def get_qa_chain():

    #load the vector database from the local folder
    vectordb=FAISS.load_local(vector_db_path,model,allow_dangerous_deserialization=True)

    #create a retreiver for the querying the vector database
    retriever=vectordb.as_retriever()

    prompt_template = """Given the following context and a question, generate an answer based on this context only.
    In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
    If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.

    CONTEXT: {context}

    QUESTION: {question}"""


    PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
    )
    chain_type_kwargs = {"prompt": PROMPT}

    chains=RetrievalQA.from_chain_type(llm=llm,
                                   chain_type='stuff',
                                   retriever=retriever,
                                   input_key='query',
                                   return_source_documents=True,
                                  chain_type_kwargs=chain_type_kwargs)

    return chains


# ans=get_qa_chain()
# print(ans.invoke("do you provide any intership"))







