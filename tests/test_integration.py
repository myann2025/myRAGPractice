import pytest
from pathlib import Path
import tempfile
import json
import xml.etree.ElementTree as ET

from src.document_loader import SupportDocumentLoader
from src.vector_store import SupportVectorStore
from src.rag_chain import SupportRAGChain
import time

class TestSupportRAGIntegration:
    @pytest.fixture
    def test_data_directory(self, tmp_path):
        """Create test data directory with sample support tickets"""
        time.sleep(5)
        data_dir = tmp_path / "support_tickets"
        data_dir.mkdir()

        # Sample ticket data
        technical_ticket = {
            "subject": "Browser Login Issue",
            "body": "Unable to login using Safari browser version 15.0",
            "answer": "Clear browser cache and cookies, then restart browser",
            "type": "Technical",
            "queue": "Tech Support",
            "priority": "high",
            "language": "en",
            "tag_1": "Browser",
            "tag_2": "Login",
            "tag_3": "Safari",
            "Ticket ID": "tech_001"
        }

        product_ticket = {
            "subject": "Dark Mode Feature",
            "body": "Requesting dark mode theme for the dashboard",
            "answer": "Dark mode feature is planned for next release",
            "type": "Enhancement",
            "queue": "Product",
            "priority": "medium",
            "language": "en",
            "tag_1": "Feature",
            "tag_2": "UI",
            "tag_3": "Dashboard",
            "Ticket ID": "prod_001"
        }

        # Create JSON files
        with open(data_dir / "Technical Support_tickets.json", 'w') as f:
            json.dump([technical_ticket], f)
        with open(data_dir / "Product Support_tickets.json", 'w') as f:
            json.dump([product_ticket], f)

        # Create XML files
        def create_xml_ticket(ticket_data, filename):
            root = ET.Element("Tickets")
            ticket = ET.SubElement(root, "Ticket")
            for key, value in ticket_data.items():
                elem = ET.SubElement(ticket, key.replace(" ", "_"))
                elem.text = str(value)
            tree = ET.ElementTree(root)
            tree.write(data_dir / filename)

        create_xml_ticket(technical_ticket, "Technical Support_tickets.xml")
        create_xml_ticket(product_ticket, "Product Support_tickets.xml")

        return data_dir

    @pytest.mark.asyncio
    async def test_full_pipeline(self, test_data_directory, tmp_path):
        """Test complete RAG pipeline with document loading, embedding, and querying"""
        time.sleep(5)
        # Initialize components
        loader = SupportDocumentLoader(str(test_data_directory))
        documents = loader.create_documents()
        
        assert len(documents) > 0
        assert 'technical' in documents
        assert 'product' in documents

        # Create vector store
        vector_store = SupportVectorStore(vecstore_path=str(tmp_path / "vector_store"))
        vector_store.create_vector_store(documents)
        
        time.sleep(2)
        # Initialize RAG chain
        rag_chain = SupportRAGChain(vector_store)

        # Test technical support query
        technical_query = "I'm having trouble logging into the system using Safari browser, please help"
        technical_docs = rag_chain.get_relevant_documents(technical_query, support_type='technical')
        
        assert len(technical_docs) > 0
        assert any("Safari" in doc['content'] for doc in technical_docs)
        assert any("browser" in doc['content'].lower() for doc in technical_docs)

        technical_response = await rag_chain.query(technical_query, support_type='technical')
        assert isinstance(technical_response, str)
        assert len(technical_response) > 0
        assert any(term in technical_response.lower() 
                  for term in ['cache', 'cookies', 'browser', 'safari'])

        # Test product support query
        product_query = "Is there a dark mode feature available for the dashboard?"
        product_docs = rag_chain.get_relevant_documents(product_query, support_type='product')
        
        assert len(product_docs) > 0
        assert any("dark mode" in doc['content'].lower() for doc in product_docs)
        assert any("dashboard" in doc['content'].lower() for doc in product_docs)

        product_response = await rag_chain.query(product_query, support_type='product')
        assert isinstance(product_response, str)
        assert len(product_response) > 0
        assert any(term in product_response.lower() 
                  for term in ['dark mode', 'feature', 'release'])

    time.sleep(2)
    @pytest.mark.asyncio
    async def test_persistence(self, test_data_directory, tmp_path):
        """Test vector store persistence and reloading"""
        time.sleep(5)
        # Create and save vector store
        vector_store_dir = tmp_path / "vector_store"
        
        # Initial setup
        loader = SupportDocumentLoader(str(test_data_directory))
        documents = loader.create_documents()
        
        time.sleep(2)
        original_store = SupportVectorStore(vecstore_path=str(vector_store_dir))
        original_store.create_vector_store(documents)

        # Load saved vector store
        loaded_store = SupportVectorStore.load_local(str(vector_store_dir))
        
        # Test query functionality
        query = "browser login issues"
        original_results = original_store.query_similar(query, support_type='technical')
        loaded_results = loaded_store.query_similar(query, support_type='technical')
        
        assert len(original_results) == len(loaded_results)
        assert all(isinstance(r, dict) for r in loaded_results)
        assert all('content' in r for r in loaded_results)
        assert all('metadata' in r for r in loaded_results)

    def test_error_handling(self, test_data_directory, tmp_path):
        """Test error handling in the integration pipeline"""
        time.sleep(5)
        # Test with invalid data path
        with pytest.raises(FileNotFoundError):
            SupportDocumentLoader("nonexistent_path")

        # Test with empty documents
        loader = SupportDocumentLoader(str(test_data_directory))
        documents = loader.create_documents()
        
        time.sleep(2)
        vector_store = SupportVectorStore(vecstore_path=str(tmp_path / "vector_store"))
        vector_store.create_vector_store(documents)
        
        # Test query with non-existent support type
        results = vector_store.query_similar(
            "test query",
            support_type="nonexistent",
            k=3
        )
        assert len(results) == 0

        # Test query with empty string
        results = vector_store.query_similar("", support_type="technical")
        assert len(results) == 0