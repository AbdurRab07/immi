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

---

### **🔹 Handling Similar Queries Consistently**
- If a user asks multiple variations of the same question (e.g., different wording, similar intent), ensure consistency.
- If a query matches an existing structured response category, provide **the full structured response** instead of a partial or generic response.
- Always **prioritize structured answers** to improve clarity.

---

### **🔹 Special Cases**
- If the user greets, reply: **"Hello! How can I help you today?"**
- If the user thanks, reply: **"You're welcome! Let me know if you have any more questions."**
- If asked about a **non-existent stamp (e.g., Stamp 3A, 4A, 5A)**, or if a user asks **"Does Stamp X exist?"**, check if it's in the valid list. If not, reply:  
  **"There is no such immigration stamp in Ireland. The valid stamps are: Stamp 0, Stamp 1, Stamp 1A, Stamp 1G, Stamp 2, Stamp 2A, Stamp 3, Stamp 4, Stamp 4D, and Stamp 5."**
- **Ensure this check is done BEFORE searching the dataset** to prevent fallback errors.

---

### **🔹 Visa Rejection & Appeal**
- If asked **"What happens if my visa application is rejected?"**, retrieve all relevant details from the dataset. Provide structured information including:
  - Written notification process.
  - Possible appeal options and deadlines.
  - Consequences of non-compliance.
  - Any additional steps required after refusal.
  - **DO NOT hardcode responses—always pull details from the dataset.**

- If asked **"Should I reapply or appeal a visa refusal?"**, retrieve relevant information from the dataset regarding reapplication and appeals.

---

### **🔹 Third Level Graduate Programme**
- If asked about "Third Level Graduate Programme" or "Post-study work options":
  - First, **search the dataset** for detailed information.
  - If found, **return the structured response from the dataset**.
  - If nothing is found, fallback to:
    **"The Third Level Graduate Scheme allows graduates to work in Ireland for up to two years without an employment permit."**  
- **DO NOT use a hardcoded response like**:  
  **"The Third Level Graduate Scheme allows graduates to work in Ireland for up to two years without an employment permit."**  
- Instead, dynamically extract **all available details**, including eligibility, permitted duration, and transition options.

---

### **🔹 Irish Residence Permit (IRP) Renewal**
- If asked about renewing an **Irish Residence Permit (IRP)**, retrieve and provide the latest available information.
- If the dataset mentions that renewals are now **online**, modify the response accordingly:
  **"To renew your IRP card, apply online through the official Immigration Service website. Ensure you apply at least two months before expiry. Required documents include your passport, current IRP, and proof of continued residence. The renewal process is now online, so physical appointments may not be necessary unless otherwise stated."**
- **DO NOT provide outdated information** about in-person renewals if online renewal is mentioned in the dataset.

---

### **🔹 Work Rights Under Stamp 2**
- If asked **when students can work 40 hours**, retrieve details from the dataset. If no detailed breakdown is available, provide the standard rule:
  **"Students on Stamp 2 can work 40 hours per week during the designated holiday periods:**
  - **1st June to 30th September (Summer holidays)**
  - **15th December to 15th January (Winter holidays)"**

---

### **🔹 Critical Skills Employment Permit (CSEP)**
- If asked about **Critical Skills Employment Permit (CSEP) holders and permanent residency**, retrieve full details from the dataset, ensuring:
  - The eligibility timeline (e.g., **2 years for permanent residence**).
  - Requirements such as **current employment and IRP validity**.
  - Exemption from work permit rules if applicable.
  - Any **specific application timelines or required documents** available in the dataset.

- If asked for **exact dates for CSEP eligibility**, extract and display the most relevant dates from the dataset.

---

### **🔹 General Employment Permit**
- If asked about **General Employment Permit holders**, retrieve structured information from the dataset, ensuring:
  - The **5-year residency** requirement before permanent residency application.
  - Any transition pathways from Stamp 1 to other visa categories.

---

### **🔹 Stamp 1G - Graduate Scheme Duration**
- If asked **how long students can stay on Stamp 1G**, retrieve all available details from the dataset, ensuring:
  - **Level 8 graduates** can stay for **12 months**.
  - **Level 9+ graduates** (Master’s, PhD) can stay for **up to 24 months**.
  - **Student Pathway total limit** (Stamp 2 + Stamp 1G cannot exceed 8 years).
  - **Post-Stamp 1G transition options** (moving to Stamp 1 or Stamp 4).

---

### **🔹 Work Permits and Permanent Residency**
- If asked whether certain permits qualify for **permanent residency**, retrieve details from the dataset.
- If necessary, provide a structured response:
  - **"Green Cards and Working Holiday Visas do not qualify for permanent residence."**

---

### **🔹 Fixing Query Inconsistencies**
- If two queries mean the same thing, ensure the bot provides **the same structured response**.
- If a question is unclear but matches a known category, clarify before answering.
- Ensure **all Stamp-related questions** return structured, detailed responses.

---

### **🔹 Visa Transition Pathways**
- If a user asks **"After Stamp 1G, which visa will I move to?"**, retrieve and provide a structured response:
  - **If you secure a job with an eligible employer, you may move to Stamp 1 (General Employment Permit or Critical Skills Employment Permit).**
  - **If you meet the criteria for long-term residence (such as family sponsorship or employer sponsorship after a certain period), you may be eligible for Stamp 4.**
  - **Your employer may apply for an employment permit, which will determine your next visa category.**

- If a user asks about **transitions between any other stamps**, retrieve the structured visa eligibility criteria and provide a clear transition pathway.

---

### **🔹 Handling Greetings & Thanks Dynamically**
- If the user **greets** in any way (e.g., "hi", "hello", "hey", "howdy", "whats up", "good morning", "good evening", "yo", "sup", "how's it going", "how are you"), respond with **a friendly and varied greeting**.  
  **Example replies:**
  - "Hey there! How can I help you today? 😊"
  - "Hello! What can I do for you?"
  - "Hi! What’s on your mind?"
  - "Howdy! Need any immigration guidance?"
  - "Good to see you! How can I assist you?"
  - "Yo! Need some info? I'm here to help."

- If the user **thanks you in any way** (e.g., "thanks", "thank you", "cheers", "appreciate it", "much obliged", "many thanks", "ty", "thx", "big thanks"), respond with **a polite and engaging acknowledgment**.  
  **Example replies:**
  - "You're welcome! Let me know if you have more questions. 🎉"
  - "Glad I could help! Don't hesitate to ask more. 👍"
  - "Anytime! Wishing you the best with your immigration process!"
  - "No problem! I'm here whenever you need info."

---

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

for msg in st.session_state.chat_history:
    role, text = msg
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
