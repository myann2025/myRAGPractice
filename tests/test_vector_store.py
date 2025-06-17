import pytest
from langchain.schema import Document
from src.vector_store import SupportVectorStore
import time

class TestSupportVectorStore:
    @pytest.fixture
    def sample_documents(self):
        """Create sample documents for testing"""
        return {
            'technical': [
                Document(
                    page_content="""
                    Subject: Browser Login Issue
                    Description: Chrome browser login fails after recent update
                    Resolution: Clear browser cache and cookies, then restart browser
                    """,
                    metadata={
                        'ticket_id': 'tech-001',
                        'support_type': 'technical',
                        'tags': ['browser', 'login', 'chrome'],
                        'priority': 'high'
                    }
                ),
                Document(
                    page_content="""
                    Subject: Application Crash
                    Description: App crashes when processing large files
                    Resolution: Updated memory allocation in config file
                    """,
                    metadata={
                        'ticket_id': 'tech-002',
                        'support_type': 'technical',
                        'tags': ['crash', 'performance', 'config'],
                        'priority': 'high'
                    }
                )
            ],
            'product': [
                Document(
                    page_content="""
                    Subject: Feature Request
                    Description: Need dark mode option in the dashboard
                    Resolution: Dark mode will be added in next release
                    """,
                    metadata={
                        'ticket_id': 'prod-001',
                        'support_type': 'product',
                        'tags': ['feature', 'ui', 'dashboard'],
                        'priority': 'medium'
                    }
                )
            ]
        }

    time.sleep(2)
    def test_create_vector_store(self, sample_documents, tmp_path):
        """Test creating vector store from documents"""
        store = SupportVectorStore(vecstore_path=str(tmp_path))
        store.create_vector_store(sample_documents)
        
        # Verify collections were created
        assert 'technical' in store.get_support_types()
        assert 'product' in store.get_support_types()

    time.sleep(2)
    def test_query_similar(self, sample_documents, tmp_path):
        """Test similarity search functionality"""
        store = SupportVectorStore(vecstore_path=str(tmp_path))
        store.create_vector_store(sample_documents)

        # Test query for technical support
        results = store.query_similar("browser login problems", support_type='technical', k=1)
        assert len(results) == 1
        assert "browser" in results[0]['content'].lower()
        assert "login" in results[0]['content'].lower()

        # Test query for product support
        results = store.query_similar("dark mode dashboard", support_type='product', k=1)
        assert len(results) == 1
        assert "dark mode" in results[0]['content'].lower()

        # Test query across all support types
        results = store.query_similar("dashboard problems", k=2)
        assert len(results) <= 2  # Might return fewer if similarity scores are low

    time.sleep(2)
    def test_save_and_load_local(self, sample_documents, tmp_path):
        """Test saving and loading vector store locally"""
        store_path = str(tmp_path / "vector_store")
        
        # Create and save vector store
        store = SupportVectorStore(vecstore_path=store_path)
        store.create_vector_store(sample_documents)

        # Load vector store and test
        loaded_store = SupportVectorStore.load_local(store_path)

        # Test search with loaded store
        results = loaded_store.query_similar("browser login", support_type='technical', k=1)
        assert len(results) == 1
        assert "browser" in results[0]['content'].lower()
        assert "login" in results[0]['content'].lower()

    time.sleep(2)
    def test_query_nonexistent_support_type(self, sample_documents, tmp_path):
        """Test querying with non-existent support type"""
        store = SupportVectorStore(vecstore_path=str(tmp_path))
        store.create_vector_store(sample_documents)
        
        results = store.query_similar("test query", support_type='nonexistent')
        assert len(results) == 0

    time.sleep(2)
    def test_empty_query(self, sample_documents, tmp_path):
        """Test handling of empty query"""
        store = SupportVectorStore(vecstore_path=str(tmp_path))
        store.create_vector_store(sample_documents)
        
        results = store.query_similar("")
        assert isinstance(results, list)
        assert len(results) == 0
    
    def test_metadata_processing(self, tmp_path):
        """Test that metadata is processed correctly for ChromaDB compatibility"""
        store = SupportVectorStore(vecstore_path=str(tmp_path))
        
        # Create sample documents with various metadata types that need processing
        documents = {
            'technical': [
                Document(
                    page_content="Test content for metadata processing",
                    metadata={
                        'ticket_id': 'meta-001',
                        'support_type': 'technical',
                        'tags': ['test', 'metadata', 'processing'],
                        'resolved_by': None,
                        'priority': 1,
                        'is_resolved': True,
                        'complex_data': {'key': 'value'},
                    }
                )
            ]
        }
        
        # Process the documents - this shouldn't raise any errors
        try:
            store.create_vector_store(documents)
        except Exception as e:
            pytest.fail(f"Metadata processing failed: {str(e)}")
        
        # Verify the collection was created
        assert 'technical' in store.get_support_types()
        
        # Query to get the document back
        results = store.query_similar("metadata processing", support_type='technical', k=1)
        assert len(results) == 1
        
        # Verify metadata was processed correctly when returned
        metadata = results[0]['metadata']
        
        # Check that tags were converted back to a list
        assert isinstance(metadata['tags'], list)
        assert set(metadata['tags']) == set(['test', 'metadata', 'processing'])
        
        # Check that None values were handled
        assert 'resolved_by' in metadata
        assert metadata['resolved_by'] == ''
        
        # Check that numbers were preserved as strings (ChromaDB converts all to strings)
        assert 'priority' in metadata
        assert metadata['priority'] == 1
        
        # Check that booleans were preserved as strings
        assert 'is_resolved' in metadata
        assert metadata['is_resolved'] == True
        
        # Check that complex data was converted to string
        assert 'complex_data' in metadata
        assert metadata['complex_data'] == "{'key': 'value'}"