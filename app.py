import streamlit as st
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationSummaryMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate

# **🔹 Load OpenAI API Key from Streamlit Secrets**
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("❌ ERROR: OpenAI API key not found. Please check your Streamlit Secrets configuration.")
    st.stop()

# **🔹 Load FAISS Database**
FAISS_INDEX_FILE = "vectorstore/db_faiss"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

try:
    db = FAISS.load_local(FAISS_INDEX_FILE, embedding_model, allow_dangerous_deserialization=True)
    print("✅ FAISS index loaded successfully.")
except Exception as e:
    st.error(f"❌ FAISS loading error: {e}")
    raise e

# **🔹 Define Memory (Using ConversationSummaryMemory for Context)**
memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4o", openai_api_key=OPENAI_API_KEY, temperature=0),
    memory_key="chat_history",
    return_messages=True
)

# **🔹 Define LLM with Temperature Setting**
llm = ChatOpenAI(
    model="gpt-4o",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.3
)

# **🔹 Prompt Template**
CONTROLLED_PROMPT_TEMPLATE = """
You are an expert in Irish immigration laws. Answer questions accurately using only the provided data.
If the answer is not found, say:  
**'I am continuously learning. Maybe I can respond to this question later.'**  
NEVER say 'Google search' or provide unrelated information.

### **🔹 Context & Query Handling**
{context}

### **User Question:**
{question}

Retrieve all **relevant** and **detailed** responses from the dataset. If applicable, provide structured formatting.
"""

# **🔹 Conversational RAG Chain**
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 8}),
    memory=memory,
    return_source_documents=False,
    combine_docs_chain_kwargs={
        "prompt": PromptTemplate(template=CONTROLLED_PROMPT_TEMPLATE, input_variables=["context", "question"])
    }
)

# **🔹 Streamlit UI Setup**
st.set_page_config(page_title="Irish Immigration Chatbot", layout="wide")

st.title("🇮🇪 Irish Immigration Chatbot")
st.markdown("#### *Built by Abdul Rub aka Rajar*")
st.markdown("💬 Ask me anything related to Irish immigration laws.")
st.markdown("---")

# **🔹 Chat History Handling**
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

# **🔹 User Input Handling**
user_query = st.chat_input("Type your message here...")

# if user_query:
#     user_query = user_query.lower().strip()

#     with st.chat_message("user"):
#         st.markdown(user_query)

#     # **🔹 Handle Simple Greetings & Thanks**
#     greetings = ["hi", "hello", "hey", "good morning", "good evening"]
#     thanks_messages = ["thanks", "thank you", "appreciate it"]

#     if user_query in greetings:
#         answer = "Hello! How can I help you today?"
#     elif user_query in thanks_messages:
#         answer = "You're welcome! Let me know if you have any more questions."
#     else:
#         # **🔹 Process Query Using RAG**
#         with st.spinner("Thinking..."):
#             try:
#                 response = qa_chain.invoke({"question": user_query})
#                 answer = response.get("answer", "I am continuously learning. Maybe I can respond to this question later.")
#             except Exception as e:
#                 answer = f"⚠️ An error occurred: {e}"

#     # **🔹 Display Answer**
#     with st.chat_message("assistant"):
#         st.markdown(answer)

#     # **🔹 Store Chat History**
#     st.session_state.chat_history.append(("user", user_query))
#     st.session_state.chat_history.append(("assistant", answer))

# **🔹 User Input Processing**
if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    # **🔹 Process Query Using RAG (Including Greetings & Thanks via Prompt)**
    with st.spinner("Thinking..."):
        try:
            response = qa_chain.invoke({"question": user_query})
            answer = response.get("answer", "I am continuously learning. Maybe I can respond to this question later.")
        except Exception as e:
            answer = f"⚠️ An error occurred: {e}"

    # **🔹 Display Response**
    with st.chat_message("assistant"):
        st.markdown(answer)

    # **🔹 Store Chat History**
    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("assistant", answer))

# **🔹 Clear Chat Button**
st.sidebar.button("🔄 Clear Chat", on_click=lambda: st.session_state.clear())

# **🔹 Debug OpenAI API Key Test**
try:
    test_response = llm.invoke("Test query")
    print(f"✅ OpenAI API Test Passed: {test_response}")
except Exception as e:
    print(f"❌ OpenAI API Key Error: {e}")
    st.error(f"❌ OpenAI API Key Error: {e}")
