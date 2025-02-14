import os
import streamlit as st
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from dotenv import load_dotenv
from difflib import get_close_matches

# Load OpenAI API Key from Streamlit Secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("❌ ERROR: OpenAI API key not found. Please set it in Streamlit Secrets.")
    st.stop()

# Load FAISS Database
FAISS_INDEX_FILE = "vectorstore/db_faiss"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

if "faiss_db" not in st.session_state:
    try:
        st.session_state.faiss_db = FAISS.load_local(FAISS_INDEX_FILE, embedding_model, allow_dangerous_deserialization=True)
        print("✅ FAISS index loaded successfully.")
    except Exception as e:
        st.error(f"❌ FAISS loading error: {e}")
        st.stop()

db = st.session_state.faiss_db  # Use cached FAISS instance

# Define LLM with Temperature Setting
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.3
)

# Contextualize Question
contextualize_q_system_prompt = """
Given a chat history and the latest user question, which might reference context in the chat history, 
formulate a standalone question that can be understood without the chat history. 
Do NOT answer the question, just reformulate it if needed and otherwise return it as is.
"""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, db.as_retriever(search_kwargs={"k": 10}), contextualize_q_prompt
)

# QA Prompt
qa_system_prompt = """
You are an expert in Irish immigration laws. Answer questions **ONLY** using the provided data.
If the answer is **not found** in the data:
- **If the question contains a minor spelling mistake**, correct it and provide the correct answer.
- **If the term does not exist in Irish immigration law**, clearly say:  
  **"There is no such term in Irish immigration law. Please recheck your query or clarify."**
"""

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("system", "{context}"),
    ]
)

# Create Chains
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Streamlit UI Setup
st.set_page_config(page_title="Irish Immigration Chatbot", layout="wide")

st.title("🇮🇪 Irish Immigration Chatbot")
st.markdown("#### *Built by Abdul Rub aka Rajar*")
st.markdown("💬 Ask me anything related to Irish immigration laws.")
st.markdown("---")

# Initialize Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display Chat Messages
for msg in st.session_state.chat_history:
    role, text = msg
    with st.chat_message(role):
        st.markdown(text)

# User Input Handling
user_query = st.chat_input("Type your message here...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    # Process Query Using RAG
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
            answer = response["answer"]
        except Exception as e:
            answer = f"⚠️ An error occurred: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("assistant", answer))

st.sidebar.button("🔄 Clear Chat", on_click=lambda: st.session_state.clear())
