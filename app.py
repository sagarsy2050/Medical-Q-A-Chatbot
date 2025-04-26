import streamlit as st
from datasets import load_dataset
import pandas as pd
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="Medical Q&A Chatbot", layout="wide")
st.title("🩺 Medical Q&A Chatbot using LangChain, Chroma & FLAN-T5")

# --- Session state for memory ---
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- Sidebar setup ---
st.sidebar.header("⚙️ Settings")
load_data = st.sidebar.button("🔄 Load & Initialize System")

# List of 50 medical diagnostic questions
# questions = [
#     "I have a patient with persistent cough and weight loss. How can I confirm a diagnosis of tuberculosis?",
#     "My patient presents with acute chest pain. What tests should I order to rule out myocardial infarction?",
#     # Add other questions here...
#     "I suspect a patient has osteoporosis. What test can confirm this diagnosis?",
#     "My patient has persistent fever and lymphadenopathy. How do I confirm lymphoma?"
# ]
questions = [
    "I have a patient with persistent cough and weight loss. How can I confirm a diagnosis of tuberculosis?",
    "My patient presents with acute chest pain. What tests should I order to rule out myocardial infarction?",
    "I suspect appendicitis in a patient with right lower quadrant pain. What is the best way to confirm the diagnosis?",
    "A child has a rash and high fever. How can I differentiate between measles and other viral exanthems?",
    "My patient has polyuria and polydipsia. What investigations confirm diabetes mellitus?",
    "I have a patient with jaundice. How do I confirm if it is due to hepatitis?",
    "My patient complains of sudden vision loss. What tests should I order to diagnose retinal detachment?",
    "I suspect meningitis in a febrile patient with neck stiffness. How do I confirm the diagnosis?",
    "My patient has unexplained anemia. What tests can help determine the underlying cause?",
    "I have a patient with chronic diarrhea. How can I confirm if it is due to inflammatory bowel disease?",
    "My patient is experiencing severe headaches and visual disturbances. How do I confirm a diagnosis of migraine?",
    "I suspect deep vein thrombosis in a patient with leg swelling. What diagnostic steps should I take?",
    "My patient has a persistent sore throat and enlarged tonsils. How can I confirm streptococcal pharyngitis?",
    "I have a patient with hematuria. What tests should I order to confirm a diagnosis of urinary tract infection?",
    "My patient presents with confusion and fever. How do I confirm a diagnosis of encephalitis?",
    "I suspect a patient has celiac disease. What investigations are required for confirmation?",
    "My patient complains of chest tightness and wheezing. How do I confirm a diagnosis of asthma?",
    "I have a patient with joint pain and swelling. What tests can confirm rheumatoid arthritis?",
    "My patient has a new skin lesion. How can I confirm if it is malignant melanoma?",
    "I suspect hypothyroidism in a patient with fatigue and weight gain. What tests confirm the diagnosis?",
    "My patient presents with sudden onset hemiplegia. How do I confirm a diagnosis of stroke?",
    "I have a patient with fever and a heart murmur. How can I confirm infective endocarditis?",
    "My patient has persistent vomiting and abdominal pain. What tests confirm pancreatitis?",
    "I suspect a patient has HIV infection. What is the best way to confirm the diagnosis?",
    "My patient reports hearing loss. What diagnostic tests should I order?",
    "I have a patient with chronic cough and night sweats. How do I confirm a diagnosis of lung cancer?",
    "My patient has new onset seizures. What investigations are needed to determine the cause?",
    "I suspect a urinary tract stone in a patient with flank pain. What is the best way to confirm this?",
    "My patient has persistent fever and splenomegaly. How can I confirm a diagnosis of malaria?",
    "I have a patient with unexplained bruising. What tests should I order to confirm a bleeding disorder?",
    "My patient presents with muscle weakness and double vision. How do I confirm myasthenia gravis?",
    "I suspect a patient has bacterial pneumonia. What investigations confirm the diagnosis?",
    "My patient has chronic fatigue and joint pain. How do I confirm systemic lupus erythematosus?",
    "I have a patient with unexplained weight loss. What investigations should I consider?",
    "My patient has a persistent productive cough. How do I confirm chronic obstructive pulmonary disease?",
    "I suspect a patient has acute kidney injury. What tests confirm this diagnosis?",
    "My patient presents with palpitations and anxiety. How do I confirm hyperthyroidism?",
    "I have a patient with sudden severe abdominal pain. How do I confirm a diagnosis of perforated peptic ulcer?",
    "My patient has a history of alcohol use and presents with confusion. How do I confirm hepatic encephalopathy?",
    "I suspect a patient has mononucleosis. What investigations should I order?",
    "My patient has persistent back pain. How do I confirm a diagnosis of herniated disc?",
    "I have a patient with new onset hypertension. What tests should I order to rule out secondary causes?",
    "My patient has fever and petechial rash. How do I confirm meningococcemia?",
    "I suspect a patient has osteoporosis. What test can confirm this diagnosis?",
    "My patient presents with polyuria and polydipsia. How do I confirm diabetes insipidus?",
    "I have a patient with a non-healing ulcer. What investigations should I consider for malignancy?",
    "My patient has recurrent abdominal pain and bloating. How do I confirm irritable bowel syndrome?",
    "I suspect a patient has acute cholecystitis. What is the best way to confirm the diagnosis?",
    "My patient presents with syncope. What tests should I order to determine the cause?",
    "I have a patient with persistent fever and lymphadenopathy. How do I confirm lymphoma?"
]
# --- Main pipeline setup ---
if load_data:
    st.info("Step 1: Loading and preprocessing dataset...")
    data = load_dataset("keivalya/MedQuad-MedicalQnADataset", split='train')
    df = data.to_pandas()
    df = df[:100]  # For demo

    st.success("Dataset loaded.")

    # Step 2: Document loader
    st.info("Step 2: Loading documents...")
    df_loader = DataFrameLoader(df, page_content_column="Answer")
    documents = df_loader.load()

    # Step 3: Split documents
    st.info("Step 3: Splitting text into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1250, separator="\n", chunk_overlap=100)
    split_texts = text_splitter.split_documents(documents)

    # Step 4: Generate embeddings
    st.info("Step 4: Generating embeddings using HuggingFace...")
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Step 5: Create Chroma DB
    st.info("Step 5: Creating Chroma vector store...")
    chroma_db = Chroma.from_documents(split_texts, embedder, persist_directory="chromadb")

    # Step 6: Load seq2seq model
    st.info("Step 6: Loading FLAN-T5 model...")
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-large",  # Better model for medical Q&A
        max_new_tokens=512,
        temperature=0.7
    )
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Step 7: Setup memory
    memory = ConversationBufferWindowMemory(memory_key='chat_history', k=4, return_messages=True)

    # Step 8: Create Conversational Retrieval Chain
    st.info("Step 8: Building QA Chain...")
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=chroma_db.as_retriever(),
        memory=memory
    )

    # Store in session state
    st.session_state.qa_chain = qa_chain
    st.session_state.chat_history = []

    st.success("✅ System is ready! Ask your medical question below.")

# Step 9: Ask questions and interact
if st.session_state.qa_chain:
    st.markdown("## 💬 Ask a Medical Question")
    user_query = st.text_input("Type your question here:", key="user_input")

    if st.button("Ask"):
        if user_query.strip() != "":
            with st.spinner("Generating answer..."):
                # Format the query with instruction for the model
                formatted_question = (
                    f"Act like a professional doctor. Based on the context and your knowledge, "
                    f"answer clearly:\n\nQuestion: {user_query}"
                )
                result = st.session_state.qa_chain.invoke({"question": formatted_question})
                answer = result["answer"]
                st.session_state.chat_history.append(("User", user_query))
                st.session_state.chat_history.append(("Bot", answer))

    if st.session_state.chat_history:
        st.markdown("### 📜 Chat History")
        for role, text in st.session_state.chat_history[::-1]:  # Show latest on top
            if role == "User":
                st.markdown(f"**🧑‍⚕️ You:** {text}")
            else:
                st.markdown(f"**🤖 Bot:** {text}")

    # --- List of Predefined Questions ---
    st.markdown("### 🩺 Predefined Medical Diagnostic Questions")
    for i, query in enumerate(questions, 1):
        if st.button(f"Ask Q{i}: {query}"):
            with st.spinner("Generating detailed answer..."):
                formatted_question = (
                    f"Act like a professional doctor. Based on the context and your knowledge, "
                    f"answer clearly:\n\nQuestion: {query}"
                )
                response = st.session_state.qa_chain.invoke({"question": formatted_question})
                answer = response["answer"]
                st.session_state.chat_history.append(("User", query))
                st.session_state.chat_history.append(("Bot", answer))

    # --- Debugging: Show retrieved documents ---
    if st.checkbox("Show Retrieved Context"):
        st.markdown("### 🔍 Retrieved Context")
        retriever = st.session_state.qa_chain.retriever
        docs = retriever.get_relevant_documents(user_query)
        for doc in docs[:3]:  # Display top 3 retrieved docs
            st.write(doc.page_content)

else:
    st.warning("⚠️ Please click 'Load & Initialize System' from the sidebar to begin.")
