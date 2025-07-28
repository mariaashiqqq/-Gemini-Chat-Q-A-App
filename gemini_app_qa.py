import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# ---- API Key (embedded directly) ----
GOOGLE_API_KEY = "AIzaSyCs7CwHif0-5OyHa22mXBVU3Z7x4u8D_yM"

# ---- Load Gemini LLM ----
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GOOGLE_API_KEY
)

# ---- Prompt Template ----
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful AI assistant."),
    ("human", "Question: {question}")
])

# ---- Output Parser ----
output_parser = StrOutputParser()

# ---- Build the Chain ----
chain = prompt | llm | output_parser

# ---- Streamlit UI ----
st.set_page_config(page_title="Gemini Q&A", layout="centered")
st.title("🤖 Gemini Chat Q&A App")

input_text = st.text_input("Ask a question:")

if input_text:
    with st.spinner("Thinking..."):
        try:
            response = chain.invoke({"question": input_text})
            st.success("✅ Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"❌ Error: {e}")
