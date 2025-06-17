import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional

import streamlit as st

from src.document_loader import SupportDocumentLoader
from src.rag_chain import SupportRAGChain
from src.vector_store import SupportVectorStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("support_rag")

# Constants
VECTOR_STORE_DIR = "vector_store"
DATA_PATH = "data"

# Initialize Streamlit state placeholders
status_placeholder = st.empty()
progress_bar = st.progress(0)

def log_error(e: Exception) -> str:
    """
    Log an error and return formatted error message.
    
    Args:
        e (Exception): The exception to log
        
    Returns:
        str: Formatted error message for display
    """
    logger.error(e, exc_info=True)
    return f"❌ Error: {str(e)}"

def get_documents():
    """
    Load support documents from the data directory.
    
    Returns:
        List[Document]: List of processed documents
    """
    try:
        status_placeholder.info("📚 Loading support documents...")
        loader = SupportDocumentLoader(DATA_PATH)
        documents = loader.create_documents()
        status_placeholder.success("✅ Support documents loaded successfully!")
        return documents
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def create_new_vector_store() -> Optional[SupportVectorStore]:
    """
    Create a new vector store from scratch.
    
    Returns:
        Optional[SupportVectorStore]: New vector store instance or None if creation fails
    """
    try:
        vector_store = SupportVectorStore(vecstore_path=VECTOR_STORE_DIR)
        
        status_placeholder.info("⚙️ Creating new vector store...")
        progress_bar.progress(40)
        
        # Create documents
        documents = get_documents()
        if not documents:
            return None
        progress_bar.progress(60)
        
        # Create embeddings and vector store
        status_placeholder.info("🔨 Generating embeddings...")
        vector_store.create_vector_store(documents)
        progress_bar.progress(80)
        
        # Save vector store
        status_placeholder.info("💾 Saving vector store...")
        progress_bar.progress(100)
        
        status_placeholder.success("✅ Vector store created and saved successfully!")
        return vector_store
        
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def load_existing_vector_store() -> Optional[SupportVectorStore]:
    """
    Load an existing vector store from disk.
    
    Returns:
        Optional[SupportVectorStore]: Loaded vector store instance or None if loading fails
    """
    try:
        status_placeholder.info("🔄 Loading existing vector store...")
        progress_bar.progress(30)
        vector_store = SupportVectorStore.load_local(VECTOR_STORE_DIR)
        progress_bar.progress(100)
        status_placeholder.success("✅ Vector store loaded successfully!")
        return vector_store
    except Exception as e:
        status_placeholder.error(log_error(e))
        return None

def initialize_rag_system() -> Optional[SupportRAGChain]:
    """
    Initialize the RAG system by loading or creating vector store.
    
    Returns:
        Optional[SupportRAGChain]: Initialized RAG chain or None if initialization fails
    """
    try:
        # Try to load existing vector store, create new one if loading fails
        vector_store = load_existing_vector_store()
        if not vector_store:
            vector_store = create_new_vector_store()
            if not vector_store:
                return None
        
        # Initialize RAG chain
        status_placeholder.info("🤖 Initializing RAG chain...")
        rag_chain = SupportRAGChain(vector_store)
        
        status_placeholder.empty()
        return rag_chain
        
    except Exception as e:
        error_msg = log_error(e)
        status_placeholder.error(error_msg)
        return None

def display_system_status():
    """Display the current system status and any required setup steps."""
    if not st.session_state.rag_chain:
        st.error("⚠️ System initialization failed")
        st.info("""
        Please ensure:
        1. The data directory contains valid support ticket files
        2. OpenAI API key is properly configured
        3. All required packages are installed
        
        Check the logs for detailed error information.
        """)
        return False
    return True

def render_search_results(query: str, rag_chain: SupportRAGChain):
    """
    Render search results and AI response for a query.
    
    Args:
        query (str): User's search query
        rag_chain (SupportRAGChain): RAG chain instance
    """
    try:
        # Show spinner for document retrieval
        with st.spinner("🔍 Searching for relevant tickets..."):
            relevant_docs = rag_chain.get_relevant_documents(query)
        
        # Show spinner for AI response generation
        with st.spinner("🤖 Generating AI response..."):
            response = asyncio.run(rag_chain.query(query))
        
        # Display AI response
        st.subheader("AI Response")
        st.write(response)
        
        # Display relevant tickets
        st.subheader("Relevant Support Tickets")
        for i, doc in enumerate(relevant_docs, 1):
            with st.expander(
                f"{i}. Ticket {doc['metadata']['ticket_id']} - {doc['metadata'].get('product', 'Unknown')}"
            ):
                st.write(f"**Tags:** {', '.join(doc['metadata'].get('tags', []))}")
                st.write(f"**Content:** {doc['content']}")
                st.write(f"**Similarity Score:** {doc['similarity']:.2f}")
                
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")

def main():
    """Main application function."""
    # Clear progress indicators
    progress_bar.empty()
    
    # Set up the main page
    st.title("Support Ticket Search & Assistant")
    st.write("""
    Welcome to the Support Ticket Assistant! Ask questions about common issues
    or search for similar support tickets to help resolve your problem.
    """)
    
    # Initialize session state
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = initialize_rag_system()
    
    # Check system status
    if not display_system_status():
        return
    
    # Product filter (optional)
    # products = ["All Products", "Product A", "Product B", "Product C"]
    # selected_product = st.selectbox("Filter by Product:", products)
    
    # Search interface
    query = st.text_input(
        "Enter your question or describe your issue:",
        placeholder="e.g., 'How do I fix the login error on Safari browser?'"
    )
    
    # Search button
    if st.button("Search") and query:
        # product = None if selected_product == "All Products" else selected_product
        render_search_results(query, st.session_state.rag_chain)

if __name__ == "__main__":
    main()