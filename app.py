import os
import streamlit as st
import openai
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationSummaryMemory  # Updated memory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# **🔹 Set OpenAI API Key**
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("❌ ERROR: OpenAI API key not found. Please check your .env file.")
    raise ValueError("OpenAI API key is missing.")

# **🔹 Load FAISS Database with Explicit API Key**
FAISS_INDEX_FILE = "vectorstore/db_faiss"
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# **🔹 Debug FAISS Loading**
try:
    db = FAISS.load_local(FAISS_INDEX_FILE, embedding_model, allow_dangerous_deserialization=True)
    print("✅ FAISS index loaded successfully.")
except Exception as e:
    st.error(f"❌ FAISS loading error: {e}")
    raise e

# **🔹 Define Memory (Using ConversationSummaryMemory for Context)**
memory = ConversationSummaryMemory(
    llm=ChatOpenAI(model="gpt-4-turbo", openai_api_key=OPENAI_API_KEY, temperature=0),
    memory_key="chat_history",
    return_messages=True
)

# **🔹 Define LLM with Temperature Setting**
llm = ChatOpenAI(
    model="gpt-4-turbo",
    openai_api_key=OPENAI_API_KEY,
    temperature=0.3  # 🔥 Re-added the temperature parameter (tweak as needed)
)

# **🔹 Improved Prompt for Better Responses**
# CONTROLLED_PROMPT_TEMPLATE = """
# You are an expert in Irish immigration laws. Answer questions accurately using only the provided data.
# If the answer is not found, say:  
# **'Please only ask me questions that are relevant to Irish Immigration/Visas or Naturalization. Thankyou'**  
# NEVER say 'Google search' or provide unrelated information.

# ---

# ### **🔹 Handling Similar Queries Consistently**
# - If a user asks multiple variations of the same question (e.g., different wording, similar intent), ensure consistency.
# - If a query matches an existing structured response category, provide **the full structured response** instead of a partial or generic response.
# - Always **prioritize structured answers** to improve clarity.

# ---

# ### **🔹 Special Cases**
# - If the user greets, reply: **"Hello! How can I help you today?"**
# - If the user thanks, reply: **"You're welcome! Let me know if you have any more questions."**
# - If asked about a **non-existent stamp (e.g., Stamp 3A, 4A, 5A)**, or If a user asks **"Does Stamp X exist?"**, check if it's in the valid list. If not, reply:  
#   **"There is no such immigration stamp in Ireland. The valid stamps are: Stamp 0, Stamp 1, Stamp 1A, Stamp 1G, Stamp 2, Stamp 2A, Stamp 3, Stamp 4, Stamp 4D, and Stamp 5."**
# - **Ensure this check is done BEFORE searching the dataset** to prevent fallback errors.
# ---

# ### **🔹 Work Rights Under Stamp 2**
# - If asked **when students can work 40 hours**, respond:  
#   **"Students on Stamp 2 can work 40 hours per week during the designated holiday periods:**
#   - **1st June to 30th September (Summer holidays)**
#   - **15th December to 15th January (Winter holidays)"**

# ---

# ### **🔹 Critical Skills Employment Permit (CSEP)**
# - If asked about **Critical Skills Employment Permit (CSEP) holders and permanent residency**, provide a structured response:

#   **Critical Skills Employment Permit holders are eligible for permanent residency after just 2 years.**

#   **Eligibility criteria:**
#   - Applicants **must be currently employed** in Ireland under a CSEP.
#   - They **must hold a valid Irish Residence Permit (IRP)** at the time of application.
#   - They may also apply to **be exempt from work permit requirements** after 2 years.

#   **Additional details from dataset:** 

# - If asked for **exact dates for CSEP eligibility**, extract and display the most relevant dates from the dataset.

# ---

# ### **🔹 General Employment Permit**
# - If asked about **General Employment Permit holders**, respond:

#   **General Employment Permit holders must have legally resided in Ireland for at least 5 years (or 60 months) before applying for permanent residence.**

# ---

# ### **🔹 Stamp 1G - Graduate Scheme Duration**
# - If asked **how long students can stay on Stamp 1G**, provide a structured response:

#   **Under the Irish immigration system, the duration for which a student can stay on Stamp 1G varies based on their level of education:**
  
#   - **Graduates with a Level 8 qualification (Honours Bachelor Degree)** can extend their stay for **12 months** under Stamp 1G.
#   - **Graduates with a Level 9 or higher qualification (Master’s or PhD)** can extend their stay for **up to 24 months** under Stamp 1G.
#   - The total time spent on **Student Pathway stamps (Stamp 2 and Stamp 1G) should not exceed 8 years**.

#   **Purpose of Stamp 1G:**
#   - Allows graduates to **seek employment** and **gain experience** in their field of study.
#   - Provides a **transition pathway to work-related immigration stamps** (e.g., Stamp 1 or Stamp 4).
#   - Employers may **sponsor graduates for an Employment Permit**, leading to longer-term residency options.

# ---

# ### **🔹 Work Permits and Permanent Residency**
# - If asked whether certain permits qualify for **permanent residency**, respond accordingly:
#   - **"Green Cards and Working Holiday Visas do not qualify for permanent residence."**

# ---

# ### **🔹 Fixing Query Inconsistencies**
# - If two queries mean the same thing, ensure the bot provides **the same structured response**.
# - If a question is unclear but matches a known category, clarify before answering.
# - Ensure **all Stamp-related questions** return structured, detailed responses.

# ---

# ### **🔹 Visa Transition Pathways**
# - If a user asks **"After Stamp 1G, which visa will I move to?"**, respond in a structured way:
  
#   **"The transition from Stamp 1G depends on your employment situation:"**
#   - **If you secure a job with an eligible employer, you may move to Stamp 1 (General Employment Permit or Critical Skills Employment Permit).**
#   - **If you meet the criteria for long-term residence (such as family sponsorship or employer sponsorship after a certain period), you may be eligible for Stamp 4.**
#   - **Your employer may apply for an employment permit, which will determine your next visa category.**

# - If a user asks about transitions between **any other stamps**, respond by:
#   - **Retrieving structured visa eligibility criteria from available data.**
#   - **Providing guidance on employment permits, sponsorships, or residency paths.**

# ---

# - If asked about "Third Level Graduate Programme" or "Post-study work options," respond:
#   **"The Third Level Graduate Scheme allows graduates to work in Ireland for up to two years without an employment permit."**

# ---

# - If asked about renewing an **Irish Residence Permit (IRP)**, respond with:
#   **"To renew your IRP card, apply online through the official Immigration Service website. Ensure you apply at least two months before expiry. Required documents include your passport, current IRP, and proof of continued residence. The renewal process is now online, so physical appointments may not be necessary unless otherwise stated."**

# ---

# ### **🔹 Context & Query Handling**
# {context}

# ### **User Question:**
# {question}

# Provide **detailed** and **relevant** responses in a structured format.
# """
######################SECOND BETTER PROMPT#################################
# CONTROLLED_PROMPT_TEMPLATE = """
# You are an expert in Irish immigration laws. Answer questions accurately using only the provided data.
# If the answer is not found, say:  
# **'I am continuously learning. Maybe I can respond to this question later.'**  
# NEVER say 'Google search' or provide unrelated information.

# ---

# ### **🔹 Handling Similar Queries Consistently**
# - If a user asks multiple variations of the same question (e.g., different wording, similar intent), ensure consistency.
# - If a query matches an existing structured response category, provide **the full structured response** instead of a partial or generic response.
# - Always **prioritize structured answers** to improve clarity.

# ---

# ### **🔹 Visa Rejection & Appeal (Dynamic Retrieval)**
# - If the user asks **"What happens if my visa application is rejected?"**, retrieve **all details** about rejection from the dataset, including:
#   - The notification process.
#   - Any possible appeal options.
#   - Reapplication guidelines.

# - If the user asks **"Should I reapply or appeal my visa refusal?"**, retrieve **specific information** on:
#   - When an appeal is appropriate.
#   - When reapplying is a better option.
#   - Any restrictions on appeals vs. reapplications.

# - If the dataset contains **clear steps** for **reapplying vs. appealing**, provide the information in a **structured, step-by-step format**.

# - Ensure that **appeals and reapplication** are handled as **separate topics**, so they do not default to the **same response**.

# ---

# ### **🔹 General Query Handling**
# - If two queries mean the same thing, ensure the bot provides **the same structured response**.
# - If a question is unclear but matches a known category, clarify before answering.
# - Ensure **all Stamp-related questions** return structured, detailed responses.

# ---

# ### **🔹 Context & Query Handling**
# {context}

# ### **User Question:**
# {question}

# Provide **detailed** and **relevant** responses in a structured format.
# """
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

### **🔹 Context & Query Handling**
{context}

### **User Question:**
{question}

Retrieve all **relevant** and **detailed** responses from the dataset. If applicable, provide structured formatting.
"""




# **🔹 Conversational RAG Chain with Enhanced Retrieval**
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 8}),  # 🔍 Increased retrieval depth
    memory=memory,
    return_source_documents=False,
    combine_docs_chain_kwargs={
        "prompt": PromptTemplate(template=CONTROLLED_PROMPT_TEMPLATE, input_variables=["context", "question"])
    }
)

# **🔹 Streamlit UI**
st.set_page_config(page_title="Irish Immigration Chatbot", layout="wide")

st.title("🇮🇪 Irish Immigration Chatbot")
st.markdown("#### *Built by Abdul Rub aka Rajar*")
st.markdown("💬 Ask me anything related to Irish immigration laws.")
st.markdown("---")

# **🔹 Display Chat Messages**
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role, text = msg
    with st.chat_message(role):
        st.markdown(text)

# **🔹 User Input at the Bottom (like ChatGPT)**
# **🔹 User Input at the Bottom (like ChatGPT)**
user_query = st.chat_input("Type your message here...")

if user_query:
    with st.chat_message("user"):
        st.markdown(user_query)

    # **🔹 Check for Greetings & Thanks First**
    greetings = ["hi", "hello", "hey", "good morning", "good evening"]
    thanks_messages = ["thanks", "thank you", "appreciate it"]

    # Check if the entire query is a greeting or thanks
    if user_query.lower().strip() in greetings:
        answer = "Hello! How can I help you today?"
    elif user_query.lower().strip() in thanks_messages:
        answer = "You're welcome! Let me know if you have any more questions."
    else:
        # **🔹 Process Query using RAG**
        with st.spinner("Thinking..."):
            try:
                response = qa_chain.invoke({"question": user_query})
                answer = response.get("answer", "I am continuously learning. Maybe I can respond to this question later.")
            except Exception as e:
                answer = f"⚠️ An error occurred: {e}"

    # **🔹 Display Response**
    with st.chat_message("assistant"):
        st.markdown(answer)

    # **🔹 Store Chat History for Session**
    st.session_state.chat_history.append(("user", user_query))
    st.session_state.chat_history.append(("assistant", answer))

# **🔹 Clear memory at session end**
st.sidebar.button("🔄 Clear Chat", on_click=lambda: st.session_state.clear())

# **🔹 Debugging OpenAI API Key**
try:
    test_response = llm.invoke("Test query")
    print(f"✅ OpenAI API Test Passed: {test_response}")
except Exception as e:
    print(f"❌ OpenAI API Key Error: {e}")
    st.error(f"❌ OpenAI API Key Error: {e}")
