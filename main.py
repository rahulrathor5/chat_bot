import streamlit as st
from langchain_helper import create_vector_db,get_qa_chain
st.title("RAHUL Lifeline")
btn=st.button('create knowledgebase')

if btn:
    pass

question=st.text_input("question:")

if question:
    chain=get_qa_chain()
    response=chain.invoke(question)

    st.header("Answer: ")
    st.write(response['result'])

     

 


