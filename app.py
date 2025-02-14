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
from difflib import get_close_matches

# 🔹 **Load OpenAI API Key from Streamlit Secrets**
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("❌ ERROR: OpenAI API key not found. Please check your Streamlit Secrets configuration.")
    st.stop()

# 🔹 **Load FAISS Database**
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

# 🔹 **Define LLM with Temperature Setting**
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.0  # Prevents hallucinations, ensuring only dataset-based responses
)

# 🔹 **Generalized Context Retention Instruction**
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

history_aware_retriever = create_history_aware_retriever(
    llm, db.as_retriever(search_kwargs={"k": 20}), contextualize_q_prompt
)

# 🔹 **QA System Prompt - Preventing Hallucinations & Enforcing Context**
qa_system_prompt = """
You are an AI assistant answering user questions based only on retrieved information.

- If the retrieved documents **do not contain** an answer, respond: "I don't have enough information to answer this right now."
- Do **not** use external knowledge or general information beyond the dataset.
- If a query is **completely unrelated** to the dataset, respond: "This is outside my scope. I can only answer questions related to Irish immigration laws."
- If the retrieved data **partially** answers the question, use it to provide the best possible response.
- If the answer is **not explicitly found in the data**, do **NOT** generate an answer from general knowledge.
- If a follow-up question **depends on previous context**, use chat history to **understand and answer it correctly**.
- If the user asks about **universities or degrees**, ensure you retrieve that information properly.
- If the user asks about **stamps**, retrieve and match from the database accurately.
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

# 🔹 **Create Chains**
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 🔹 **Streamlit UI Setup**
st.set_page_config(page_title="Irish Immigration Chatbot", layout="wide")

st.title("🇮🇪 Irish Immigration Chatbot")
st.markdown("#### *Built by Abdul Rub aka Rajar*")
st.markdown("💬 Ask me anything related to Irish immigration laws.")
st.markdown("---")

# 🔹 **Initialize Chat History**
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# 🔹 **Display Chat Messages**
for msg in st.session_state.chat_history:
    role, text = msg
    with st.chat_message(role):
        st.markdown(text)

# 🔹 **User Input Handling**
user_query = st.chat_input("Type your message here...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    # 🔹 **Process Query Using RAG**
    with st.spinner("Thinking..."):
        try:
            response = rag_chain.invoke({"input": user_query, "chat_history": st.session_state.chat_history})
            answer = response.get("answer", "I don't have enough information to answer this right now.")
        except Exception as e:
            answer = f"⚠️ An error occurred: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("assistant", answer))

st.sidebar.button("🔄 Clear Chat", on_click=lambda: st.session_state.clear())
