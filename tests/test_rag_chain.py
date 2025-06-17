import pytest
from unittest.mock import AsyncMock, Mock, create_autospec
import asyncio
from langchain.schema import AIMessage
import time
from src.rag_chain import SupportRAGChain

class TestSupportRAGChain:
    @pytest.fixture
    def mock_vector_store(self):
        """Create mock vector store with predefined responses"""
        time.sleep(5)
        store = Mock()
        store.query_similar.return_value = [
            {
                'content': """
                Subject: Browser Login Issue
                Description: Unable to login using Safari browser.
                Resolution: Clear browser cache and cookies.
                Type: Technical
                Queue: Tech Support
                Priority: High
                """,
                'metadata': {
                    'ticket_id': 'tech-001',
                    'support_type': 'technical',
                    'tags': ['Browser', 'Login', 'Safari'],
                    'priority': 'high'
                },
                'similarity': 0.92
            }
        ]
        return store

    @pytest.fixture
    def mock_llm_response(self):
        """Create a mock LLM response"""
        mock = AIMessage(content=(
            "To resolve the Safari browser login issue:\n"
            "1. Clear your browser cache and cookies\n"
            "2. Restart your browser\n"
            "This is a common issue that can be resolved by clearing cached data."
        ))
        return mock

    @pytest.fixture
    def rag_chain(self, mock_vector_store):
        """Create RAG chain with mocked components"""
        chain = SupportRAGChain(mock_vector_store)
        return chain

    @pytest.mark.asyncio
    async def test_basic_query(self, rag_chain, mock_llm_response):
        """Test basic query functionality"""
        time.sleep(5)
                # Create AsyncMock for LLM
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = mock_llm_response
        # rag_chain.chain.llm = mock_llm
        query = "I'm having trouble with Safari browser login, can you help me resolve this issue?"
        
        # Set up the mock response
        # rag_chain.llm.ainvoke.return_value = mock_llm_response
        
        response = await rag_chain.query(query)
        
        assert isinstance(response, str)
        assert "cache" in response.lower()
        assert "cookies" in response.lower()

    time.sleep(2)
    def test_get_relevant_documents(self, rag_chain):
        """Test document retrieval functionality"""
        time.sleep(5)
        query = "I'm experiencing login issues with my browser, need help fixing this problem"
        docs = rag_chain.get_relevant_documents(query)
        
        assert len(docs) > 0
        assert isinstance(docs[0], dict)
        assert "content" in docs[0]
        assert "metadata" in docs[0]
        assert "similarity" in docs[0]
        

    @pytest.mark.asyncio
    async def test_multiple_sequential_queries(self, rag_chain):
        """Test multiple sequential queries"""
        time.sleep(5)
        # Create different responses for each query
        responses = [
            AIMessage(content="Clear browser cache and cookies to fix login"),
            AIMessage(content="Dark mode feature will be released next month")
        ]
        
        # Set up the mock to return different responses
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = responses
        # rag_chain.llm.ainvoke.return_value = responses

        queries = [
            "Having trouble with browser login, need immediate help",
            "When will dark mode feature be available in the dashboard?"
        ]
        
        results = []
        for query in queries:
            response = await rag_chain.query(query)
            results.append(response)
        
        assert len(results) == 2
        assert "cache" in results[0].lower()
        assert "dark mode" in results[1].lower()
        # assert rag_chain.llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_concurrent_queries(self, rag_chain):
        """Test concurrent query processing"""
        time.sleep(5)
        responses = [
            AIMessage(content="Browser login solution: Clear cache"),
            AIMessage(content="Dark mode will be available next release")
        ]
        mock_llm = AsyncMock()
        mock_llm.ainvoke.return_value = responses
        # rag_chain.llm.ainvoke.side_effect = responses

        queries = [
            "Need help with browser login issues immediately",
            "When is dark mode feature being released?"
        ]
        
        tasks = [rag_chain.query(q) for q in queries]
        results = await asyncio.gather(*tasks)

        assert len(results) == 2
        assert "cache" in results[0].lower()
        assert "dark mode" in results[1].lower()
        # assert rag_chain.llm.ainvoke.call_count == 2

    @pytest.mark.asyncio
    async def test_short_query_handling(self, rag_chain):
        """Test handling of short queries"""
        time.sleep(5)
        with pytest.raises(ValueError) as exc_info:
            await rag_chain.query("help")
        assert "Query too short" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_empty_query_handling(self, rag_chain):
        """Test handling of empty queries"""
        time.sleep(5)
        with pytest.raises(ValueError) as exc_info:
            await rag_chain.query("")
        assert "Query cannot be empty" in str(exc_info.value)

        with pytest.raises(ValueError) as exc_info:
            await rag_chain.query("   ")
        assert "Query cannot be empty" in str(exc_info.value)

    def test_document_preparation(self, rag_chain):
        """Test document context preparation"""
        time.sleep(5)
        documents = [
            {
                'content': 'Browser login issue content',
                'metadata': {
                    'ticket_id': '1',
                    'support_type': 'technical',
                    'tags': ['login', 'browser']
                }
            },
            {
                'content': 'Dark mode feature request',
                'metadata': {
                    'ticket_id': '2',
                    'support_type': 'product',
                    'tags': ['feature', 'ui']
                }
            }
        ]
        
        context = rag_chain._prepare_context(documents)
        
        assert "Browser login issue" in context
        assert "Dark mode feature" in context
        assert "technical" in context
        assert "product" in context
        assert any(tag in context for tag in ['login, browser', 'browser, login'])