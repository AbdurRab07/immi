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
import openai

# **🔹 Load OpenAI API Key from Streamlit Secrets**
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("❌ ERROR: OpenAI API key not found. Please check your Streamlit Secrets configuration.")
    st.stop()

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
        raise e

db = st.session_state.faiss_db  # Use cached FAISS instance

# Define LLM with Temperature Setting (Prevents Overuse of Model Knowledge)
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.0  # Ensuring it only uses retrieved data
)

# Generalized Context Retention Instruction (No Overfitting)
contextualize_q_system_prompt = """
Given the conversation history and the latest user question, ensure the question retains relevant context from previous messages.

- If a follow-up question contains vague references (e.g., "the decision," "this rule," "my application"), infer what the user is referring to from prior exchanges.
- If a multi-part question involves multiple topics, ensure responses address all relevant aspects.
- If the user corrects a previous query, adjust the interpretation accordingly.
- If there's ambiguity in the question, ask for clarification rather than making assumptions.
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 🔥 **Strictly Enforce Dataset-Based Retrieval (No General Knowledge)**
def retrieve_with_history(query):
    """
    Retrieves documents with query correction and ensures that if no documents are found,
    the bot does NOT generate an answer from external knowledge.
    """
    retrieved_docs = db.similarity_search(query, k=20)  # v14 had better retrieval, so increasing k from 15 to 20
    corrected_query = correct_query_dynamically(query, retrieved_docs)
    
    if corrected_query != query:
        retrieved_docs = db.similarity_search(corrected_query, k=20)

    # 🚨 **If no relevant documents exist, prevent hallucination**
    if not retrieved_docs:
        return ["I don't have enough information to answer this right now."]

    return retrieved_docs

# 🔧 **Fixes Query Correction for Better Accuracy**
def correct_query_dynamically(query, retrieved_docs):
    """
    Dynamically corrects the query based on the context of the retrieved documents.
    """
    retrieved_texts = " ".join([doc.page_content.lower() for doc in retrieved_docs])
    valid_terms = set(retrieved_texts.split())

    words = query.lower().split()
    corrected_words = []

    for word in words:
        closest_match = get_close_matches(word, valid_terms, n=1, cutoff=0.5)
        if closest_match:
            corrected_words.append(closest_match[0])
        else:
            corrected_words.append(word)
    
    return " ".join(corrected_words)

# 🚀 **QA System Prompt - Preventing Hallucinations & Enforcing Context**
qa_system_prompt = """
You are an AI assistant answering user questions based only on retrieved information.

- If the retrieved documents **do not contain** an answer, respond: "I don't have enough information to answer this right now."
- Do **not** use external knowledge or general information beyond the dataset.
- If a query is **completely unrelated** to the dataset, respond: "This is outside my scope. I can only answer questions related to Irish immigration laws."
- If the retrieved data **partially** answers the question, use it to provide the best possible response.
- If the answer is **not explicitly found in the data**, do **NOT** generate an answer from general knowledge.
- If a follow-up question **depends on previous context**, use chat history to **understand and answer it correctly**.
- If the user asks about **universities or degrees**, ensure you retrieve that information properly (fixing v15's issue where it didn't answer correctly).
- If the user asks about **stamps**, retrieve and match from the database accurately (keeping v14's better response to stamp-related queries).
- **Do not mention "dataset" explicitly** in responses.
- Only provide external website recommendations if the user explicitly asks for sources or official links.
- Always ensure that responses are user-friendly and conversational.
"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
        ("system", "{context}"),
    ]
)

# Create QA Chain with Strict Retrieval Enforcement
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# 🖥 **Streamlit UI Setup**
st.set_page_config(page_title="Irish Immigration Chatbot", layout="wide")
st.title("🇮🇪 Irish Immigration Chatbot")
st.markdown("#### *Built by Abdul Rub aka Rajar*")
st.markdown("💬 Ask me anything related to Irish immigration laws.")
st.markdown("---")

# Display Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role, text = msg
    with st.chat_message(role):
        st.markdown(text)

# 📝 **User Input Processing**
user_query = st.chat_input("Type your message here...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Thinking..."):
        try:
            retrieved_docs = retrieve_with_history(user_query)
            if retrieved_docs:
                response = question_answer_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history, "context": retrieved_docs})
                answer = response if isinstance(response, str) else response.get("answer", "I don't have enough information to answer this right now.")
            else:
                answer = "I don't have enough information to answer this right now."
        except Exception as e:
            answer = f"⚠️ An error occurred: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("assistant", answer))

st.sidebar.button("🔄 Clear Chat", on_click=lambda: st.session_state.clear())

