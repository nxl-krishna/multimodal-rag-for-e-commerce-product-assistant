import streamlit as st
import json
import base64
import os

from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
# CHANGE: Using API-based embeddings to avoid Python 3.13 compatibility issues
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from langchain_classic.prompts import ChatPromptTemplate  # FIXED IMPORT

# --- CONFIGURATION ---
st.set_page_config(page_title="Multimodal RAG", layout="wide")

# 1. API Keys Setup
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    hf_token = st.secrets["HF_TOKEN"]
except FileNotFoundError:
    # Sidebar Fallback for Keys if secrets.toml is missing
    with st.sidebar:
        groq_api_key = st.text_input("Enter Groq API Key:", type="password")
        hf_token = st.text_input("Enter HuggingFace Token (HF_...):", type="password")

if not groq_api_key or not hf_token:
    st.info("⚠️ Please provide both API keys to proceed.")
    st.stop()

# 2. Initialize Models
try:
    # Vision Model
    vision_llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct", 
        api_key=groq_api_key, 
        temperature=0.0
    )
    
    # Text Model
    text_llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        api_key=groq_api_key, 
        temperature=0.3
    )
    
    # Embeddings - API BASED
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        task="feature-extraction",
        huggingfacehub_api_token=hf_token,
    )

except Exception as e:
    st.error(f"Error initializing models: {e}")
    st.stop()

# --- HELPER FUNCTIONS ---

def extract_budget(query):
    """
    Uses LLM to extract a numeric price limit from the user's text.
    Example: "Show me headphones under 100" -> Returns 100
    """
    if not query:
        return None
        
    prompt = f"""
    Analyze this search query: "{query}"
    If the user mentions a budget or price limit (e.g., "under 500", "cheaper than 100", "max $50"), 
    extract that number as an integer.
    
    If no price limit is mentioned, return exactly "0".
    Return ONLY the number. No text.
    """
    
    try:
        response = text_llm.invoke(prompt)
        # Extract digits from string (e.g., "Budget is 500" -> "500")
        budget_str = ''.join(filter(str.isdigit, response.content))
        if budget_str:
            budget = int(budget_str)
            return budget if budget > 0 else None
        return None
    except:
        return None

def encode_image(uploaded_file):
    return base64.b64encode(uploaded_file.getvalue()).decode('utf-8')

# 3. Vector Store
@st.cache_resource
def load_vector_db():
    try:
        with open("products.json", "r") as f:
            products = json.load(f)
    except FileNotFoundError:
        st.error("❌ 'products.json' not found!")
        return None

    docs = []
    for p in products:
        content = f"Product: {p['name']}\nCategory: {p['category']}\nDescription: {p['description']}\nPrice: {p['price']}"
        docs.append(Document(page_content=content, metadata=p))
    
    # This might take a few seconds as it hits the API for every product
    db = FAISS.from_documents(docs, embeddings)
    return db

vector_db = load_vector_db()

# 5. UI Layout
st.title("🛒 Visual RAG Shopping Assistant")
st.markdown("Upload a photo of a product, and I'll find similar items in our inventory.")

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Your Upload", use_column_width=True)

with col2:
    if uploaded_file and vector_db:
        st.subheader("Analysis Results")
        user_query = st.text_input("Question (Optional)", placeholder="e.g., Is this good for gaming? I need under $100.")
        
        if st.button("🔍 Analyze Product"):
            try:
                with st.spinner("👀 Analyzing image..."):
                    image_data = encode_image(uploaded_file)
                    messages = [
                        (
                            "user", 
                            [
                                {"type": "text", "text": "Describe this product image in detail for a search query. Identify the object type, color, materials, and any visible brand features."},
                                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
                            ],
                        )
                    ]
                    ai_msg = vision_llm.invoke(messages)
                    image_description = ai_msg.content
                    
                    st.success("Image Understood!")
                    with st.expander("View Description"):
                        st.write(image_description)

                with st.spinner("💾 Searching & Filtering..."):
                    # 1. Check for Budget Constraint
                    budget = extract_budget(user_query)
                    if budget:
                        st.info(f"💰 Budget Filter Active: max ${budget}")
                    
                    # 2. Semantic Search
                    search_query = f"{image_description} {user_query}"
                    raw_results = vector_db.similarity_search(search_query, k=5)
                    
                    # 3. Apply Python Filtering (Hybrid Search)
                    filtered_docs = []
                    for doc in raw_results:
                        try:
                            price_value = doc.metadata.get('price', 0)
                            product_price = float(str(price_value).replace('$', '').replace(',', ''))
                        except ValueError:
                            product_price = 0 

                        if budget:
                            if product_price <= budget:
                                filtered_docs.append(doc)
                        else:
                            filtered_docs.append(doc)
                    
                    # Fallback
                    if not filtered_docs:
                        st.warning(f"No exact matches under ${budget}. Showing closest options:")
                        final_docs = raw_results[:2]
                    else:
                        final_docs = filtered_docs[:2]

                    context_text = "\n\n".join([f"--- PRODUCT: {d.metadata['name']} (${d.metadata['price']}) ---\n{d.page_content}" for d in final_docs])

                with st.spinner("🤖 Generating Answer..."):
                    prompt = ChatPromptTemplate.from_template(
                        """
                        You are a shopping assistant.
                        USER IMAGE DESCRIPTION: {img_desc}
                        USER QUESTION: {question}
                        STORE INVENTORY MATCHES:
                        {context}
                        
                        Compare the uploaded image to our stock. Recommend the closest match.
                        """
                    )
                    chain = prompt | text_llm
                    response = chain.invoke({
                        "img_desc": image_description,
                        "question": user_query if user_query else "What is this?",
                        "context": context_text
                    })
                    
                    st.markdown("### Recommendation")
                    st.markdown(response.content)
            
            except Exception as e:
                st.error(f"Error: {e}")
