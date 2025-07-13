import streamlit as st
import os
import tempfile
from typing import List
import warnings
from dotenv import load_dotenv
warnings.filterwarnings('ignore')

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate

# Try new import first, fallback to old if needed
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

# Configuration
class RAGConfig:
    def __init__(self):
        load_dotenv()

        self.groq_api_key = os.getenv("GROQ_API_KEY")
        self.model_name = "llama3-8b-8192"  # More stable model
        self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        self.chunk_size = 1000
        self.chunk_overlap = 200
        self.top_k = 4
        self.temperature = 0.1

# Initialize session state - NO auto-processing
def initialize_session_state():
    """Initialize session state variables safely"""
    if 'config' not in st.session_state:
        st.session_state.config = RAGConfig()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    
    if 'llm' not in st.session_state:
        st.session_state.llm = None
    
    if 'embeddings' not in st.session_state:
        st.session_state.embeddings = None
    
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    
    if 'input_counter' not in st.session_state:
        st.session_state.input_counter = 0

# Helper functions
@st.cache_resource
def initialize_groq_llm():
    """Initialize Groq LLM with caching"""
    try:
        llm = ChatGroq(
            groq_api_key=st.session_state.config.groq_api_key,
            model_name=st.session_state.config.model_name,
            temperature=st.session_state.config.temperature,
            max_tokens=1024,
            timeout=30
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {e}")
        return None

@st.cache_resource
def initialize_embeddings():
    """Initialize embeddings with caching"""
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=st.session_state.config.embedding_model,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        return embeddings
    except Exception as e:
        st.error(f"Error initializing embeddings: {e}")
        return None

def load_pdf_documents(pdf_files) -> List[Document]:
    """Load PDF documents from uploaded files"""
    documents = []
    
    for pdf_file in pdf_files:
        try:
            # Save uploaded file to temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(pdf_file.read())
                tmp_file_path = tmp_file.name
            
            loader = PyPDFLoader(tmp_file_path)
            docs = loader.load()
            
            # Add metadata
            for doc in docs:
                doc.metadata['source_file'] = pdf_file.name
            
            documents.extend(docs)
            
            # Clean up temporary file
            os.unlink(tmp_file_path)
            
        except Exception as e:
            st.error(f"Error loading {pdf_file.name}: {e}")
    
    return documents

def split_documents(documents: List[Document]) -> List[Document]:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=st.session_state.config.chunk_size,
        chunk_overlap=st.session_state.config.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    splits = text_splitter.split_documents(documents)
    return splits

def create_vector_store(documents: List[Document], embeddings) -> FAISS:
    """Create vector store from documents"""
    if not documents:
        return None
    
    try:
        vector_store = FAISS.from_documents(documents, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def create_rag_chain(llm, vector_store):
    """Create conversational RAG chain with custom prompt"""
    if not vector_store:
        return None
    
    try:
        # Custom prompt template
        prompt_template = """You are a helpful AI assistant. Use the following context to answer the user's question. If you cannot find the answer in the context, say so clearly.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key='answer'
        )
        
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": st.session_state.config.top_k}
        )
        
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={'prompt': prompt}
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error creating RAG chain: {e}")
        return None

def process_pdfs(pdf_files):
    """Process uploaded PDFs - ONLY when explicitly called"""
    if not pdf_files:
        st.warning("Please upload at least one PDF file")
        return False
    
    if st.session_state.is_processing:
        st.warning("Already processing documents. Please wait.")
        return False
    
    st.session_state.is_processing = True
    
    try:
        with st.spinner("Processing PDFs..."):
            # Load documents
            documents = load_pdf_documents(pdf_files)
            
            if not documents:
                st.error("No documents loaded")
                return False
            
            # Split documents
            splits = split_documents(documents)
            
            # Create vector store
            vector_store = create_vector_store(splits, st.session_state.embeddings)
            
            if not vector_store:
                st.error("Failed to create vector store")
                return False
            
            # Create RAG chain
            qa_chain = create_rag_chain(st.session_state.llm, vector_store)
            
            if not qa_chain:
                st.error("Failed to create RAG chain")
                return False
            
            # Store in session state
            st.session_state.qa_chain = qa_chain
            st.session_state.vector_store = vector_store
            st.session_state.processed_files = [f.name for f in pdf_files]
            
            st.success(f"Successfully processed {len(pdf_files)} PDF files with {len(documents)} pages")
            return True
            
    except Exception as e:
        st.error(f"Error processing PDFs: {e}")
        return False
    finally:
        st.session_state.is_processing = False

def query_rag_system(question: str):
    """Query the RAG system"""
    if not st.session_state.qa_chain:
        return "Please upload and process PDF files first.", []
    
    try:
        result = st.session_state.qa_chain({"question": question})
        answer = result.get('answer', 'No answer generated')
        sources = result.get('source_documents', [])
        
        # Handle different response types
        if hasattr(answer, 'content'):
            answer = answer.content
        elif not isinstance(answer, str):
            answer = str(answer)
        
        return answer, sources
    except Exception as e:
        error_msg = f"Error querying RAG system: {str(e)}"
        st.error(error_msg)
        # Log the full error for debugging
        st.write(f"Debug info: {type(e).__name__}: {str(e)}")
        return "Error occurred while processing your question. Please try again.", []

def ask_question(question: str):
    """Handle question asking with proper state management"""
    if not question.strip():
        return
    
    if st.session_state.is_processing:
        st.warning("System is busy. Please wait.")
        return
    
    st.session_state.is_processing = True
    
    try:
        with st.spinner("Thinking..."):
            answer, sources = query_rag_system(question.strip())
            st.session_state.chat_history.append((question.strip(), answer, sources))
            
        # Clear input by incrementing counter
        st.session_state.input_counter += 1
        
    except Exception as e:
        st.error(f"Error processing question: {e}")
    finally:
        st.session_state.is_processing = False

# Main Streamlit app
def main():
    st.set_page_config(
        page_title="RAG Chat System with Groq",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    initialize_session_state()
    
    st.title("RAG Chat System with Groq")
    st.markdown("Upload PDFs and chat with your documents using Groq's fast AI")
    
    # Initialize models if not already done
    if st.session_state.llm is None:
        with st.spinner("Initializing Groq LLM..."):
            st.session_state.llm = initialize_groq_llm()
    
    if st.session_state.embeddings is None:
        with st.spinner("Loading embeddings..."):
            st.session_state.embeddings = initialize_embeddings()
    
    # Check if models are loaded
    if not st.session_state.llm or not st.session_state.embeddings:
        st.error("Failed to initialize models. Please refresh the page.")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("📁Document Upload")
        
        # File upload
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
        
        # Process button - EXPLICIT action required
        if st.button("Process PDFs", type="primary", disabled=st.session_state.is_processing):
            if uploaded_files:
                success = process_pdfs(uploaded_files)
                if success:
                    st.rerun()
            else:
                st.warning("Please upload PDF files first")
        
        if st.session_state.is_processing:
            st.info("Processing in progress...")
        
        # Settings
        st.subheader("Settings")
        
        # Model selection
        model_options = ["llama3-8b-8192", "mixtral-8x7b-32768", "llama3-70b-8192"]
        selected_model = st.selectbox(
            "Groq Model",
            model_options,
            index=0 if st.session_state.config.model_name == "llama3-8b-8192" else 1
        )
        
        if selected_model != st.session_state.config.model_name:
            st.session_state.config.model_name = selected_model
            st.session_state.llm = None  # Force reinitialize
            st.rerun()
        
        chunk_size = st.slider("Chunk Size", 500, 2000, st.session_state.config.chunk_size)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, st.session_state.config.chunk_overlap)
        top_k = st.slider("Top K Results", 1, 10, st.session_state.config.top_k)
        temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.config.temperature)
        
        # Update config
        st.session_state.config.chunk_size = chunk_size
        st.session_state.config.chunk_overlap = chunk_overlap
        st.session_state.config.top_k = top_k
        st.session_state.config.temperature = temperature
        
        # Processed files info
        if st.session_state.processed_files:
            st.subheader("Processed Files")
            for file in st.session_state.processed_files:
                st.write(f"• {file}")
        
        # Clear chat history
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Model info
        st.subheader("Model Info")
        st.write(f"**Model:** {st.session_state.config.model_name}")
        st.write(f"**Embeddings:** {st.session_state.config.embedding_model}")
        st.write(f"**Status:** {'Ready' if st.session_state.qa_chain else 'Upload PDFs'}")
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Chat with Documents")
        
        # Display chat history
        for i, (question, answer, sources) in enumerate(st.session_state.chat_history):
            with st.container():
                st.markdown(f"**You:** {question}")
                st.markdown(f"**AI:** {answer}")
                
                if sources:
                    with st.expander(f"Sources ({len(sources)} documents)", expanded=False):
                        for j, doc in enumerate(sources):
                            source_file = doc.metadata.get('source_file', 'Unknown')
                            page = doc.metadata.get('page', 'Unknown')
                            st.write(f"**Source {j+1}: {source_file} (Page {page})**")
                            st.write(doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content)
                            if j < len(sources) - 1:
                                st.divider()
                
                st.divider()
        
        # Chat input - NO auto-processing
        st.subheader("Ask a Question")
        
        # Use form to prevent auto-submission
        with st.form("question_form", clear_on_submit=True):
            user_question = st.text_input(
                "Your question:",
                key=f"question_input_{st.session_state.input_counter}",
                placeholder="Type your question here...",
                disabled=not st.session_state.qa_chain or st.session_state.is_processing
            )
            
            col_a, col_b = st.columns([1, 4])
            with col_a:
                submit_button = st.form_submit_button(
                    "Ask",
                    type="primary",
                    disabled=not st.session_state.qa_chain or st.session_state.is_processing
                )
            
            if submit_button and user_question:
                ask_question(user_question)
                st.rerun()
    
    with col2:
        st.subheader("Quick Actions")
        
        if st.session_state.processed_files and not st.session_state.is_processing:
            st.write("**Suggested Questions:**")
            
            if st.button("Summarize", key="summarize_btn"):
                ask_question("Please provide a comprehensive summary of all the documents.")
                st.rerun()
            
            if st.button("Key Points", key="keypoints_btn"):
                ask_question("What are the main key points and important information from these documents?")
                st.rerun()
            
            if len(st.session_state.processed_files) > 1:
                if st.button("Compare", key="compare_btn"):
                    ask_question("Compare these documents and highlight similarities and differences.")
                    st.rerun()
            
            if st.button("FAQ", key="faq_btn"):
                ask_question("Generate a list of frequently asked questions based on these documents.")
                st.rerun()
        
        elif st.session_state.is_processing:
            st.info("Processing documents...")
        else:
            st.info("Upload and process PDF files to see quick actions.")
        
        # System status
        st.subheader("System Status")
        
        if st.session_state.llm:
            st.success("Groq LLM: Connected")
        else:
            st.error("Groq LLM: Not connected")
        
        if st.session_state.embeddings:
            st.success("Embeddings: Loaded")
        else:
            st.error("Embeddings: Not loaded")
        
        if st.session_state.qa_chain:
            st.success("RAG System: Ready")
        else:
            st.warning("RAG System: Upload PDFs first")
        
        if st.session_state.is_processing:
            st.info("Processing: In progress")
        else:
            st.success("Processing: Ready")

if __name__ == "__main__":
    main()