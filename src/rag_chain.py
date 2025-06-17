from typing import List, Dict, Any
# from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
# from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
import asyncio
import logging

from .vector_store import SupportVectorStore
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai_api = os.getenv("OPENAI_API_KEY")
logger = logging.getLogger(__name__)


class SupportRAGChain:
    """
    A class implementing the Retrieval-Augmented Generation (RAG) chain for support tickets.
    
    This class combines vector similarity search with LLM-based generation to provide
    relevant and contextual responses to support queries.
    
    IMPORTANT:
    - Empty queries MUST be rejected with the EXACT error message: "Query cannot be empty"
    - Queries shorter than 10 characters MUST be rejected with the EXACT error message: 
      "Query too short. Please provide more details."
    - Context preparation MUST follow the exact format specified in _prepare_context
    """
    
    def __init__(self, vector_store: SupportVectorStore):
        """
        Initialize the RAG chain with a vector store and LLM.
        Make sure the llm should be openAI gpt-4o
        
        Args:
            vector_store (SupportVectorStore): Vector store containing support tickets
        """
        self.vector_store = vector_store
        
        # Initialize OpenAI ChatGPT-4o model
        self.llm = ChatOpenAI(
            model="gpt-4o",
            openai_api_key=openai_api,
            temperature=0.1  # Low temperature for consistent support responses
        )
        
        # Create the prompt template for RAG
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful support assistant. Your task is to provide accurate and helpful responses to support queries based on the provided context from similar support tickets.

Guidelines:
1. Use the provided context from similar support tickets to inform your response
2. If the context contains relevant information, use it to provide a comprehensive answer
3. If the context doesn't contain enough information, acknowledge this and provide general guidance
4. Be concise but thorough in your responses
5. Always maintain a professional and helpful tone
6. If multiple solutions are available, present them clearly

Context from similar support tickets:
{context}"""),
            ("user", "{query}")
        ])
        
        logger.info("Initialized SupportRAGChain with OpenAI GPT-4o")

        

    def get_relevant_documents(
        self, 
        query: str, 
        support_type: str = None, 
        k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant support tickets for a given query.
        
        IMPORTANT:
        - Empty queries or queries shorter than 10 characters MUST be rejected with ValueError
        - The exact error message should be: "Query too short. Please provide more details."
        
        Args:
            query (str): User's support query
            support_type (str, optional): Specific support type to search for
            k (int): Number of documents to retrieve
            
        Returns:
            List[Dict[str, Any]]: List of relevant documents with metadata
            
        Raises:
            ValueError: If query is empty or too short (less than 10 characters)
        """
        if not query or not query.strip():
            raise ValueError("Query too short. Please provide more details.")
        
        # Check for queries shorter than 10 characters
        if len(query.strip()) < 10:
            raise ValueError("Query too short. Please provide more details.")
        
        try:
            # Use the vector store to find similar documents
            similar_docs = self.vector_store.query_similar(
                query=query.strip(),
                support_type=support_type,
                k=k
            )
            
            logger.info(f"Retrieved {len(similar_docs)} relevant documents for query")
            return similar_docs
            
        except Exception as e:
            logger.error(f"Error retrieving relevant documents: {e}")
            raise



    def _prepare_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Prepare retrieved documents into a formatted context string.
        
        IMPORTANT:
        - The context MUST be formatted with the EXACT format shown below
        - Each document must include: Support Type, Tags, and Content
        - When no documents are found, return "No relevant support tickets found."
        
        Args:
            documents (List[Dict[str, Any]]): Retrieved similar documents
            
        Returns:
            str: Formatted context string with the exact format:
            
            Ticket {i}:
            Support Type: {doc['metadata'].get('support_type', 'Unknown')}
            Tags: {', '.join(doc['metadata'].get('tags', []))}
            Content: {doc['content']}
        """
        if not documents:
            return "No relevant support tickets found."
        
        context_parts = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            content = doc.get('content', '')
            
            # Extract support type
            support_type = metadata.get('support_type', 'Unknown')
            
            # Extract and format tags
            tags = metadata.get('tags', [])
            tags_str = ', '.join(tags) if tags else ''
            
            # Format the ticket information
            ticket_info = f"""Ticket {i}:
Support Type: {support_type}
Tags: {tags_str}
Content: {content}"""
            
            context_parts.append(ticket_info)
        
        return '\n\n'.join(context_parts)

    async def query(
        self, 
        query: str, 
        support_type: str = None
    ) -> str:
        """
        Generate a response to a support query using RAG.
        
        IMPORTANT:
        - Empty queries MUST be rejected with the EXACT error message: "Query cannot be empty"
        - Queries with only whitespace MUST be rejected with the EXACT error message: "Query cannot be empty"
        - Queries shorter than 10 characters MUST be rejected with the EXACT error message:
          "Query too short. Please provide more details."
        
        Args:
            query (str): User's support query
            support_type (str, optional): Specific support type to search for
            
        Returns:
            str: Generated response based on relevant support tickets
            
        Raises:
            ValueError: With message "Query cannot be empty" if query is empty or whitespace only
            ValueError: With message "Query too short. Please provide more details." if query is shorter than 10 chars
            Exception: If there's an error generating the response
        """
        # Check for empty or whitespace-only queries
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # Check for queries shorter than 10 characters
        if len(query.strip()) < 10:
            raise ValueError("Query too short. Please provide more details.")
        
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Retrieve relevant documents
            relevant_docs = self.get_relevant_documents(
                query=query,
                support_type=support_type,
                k=3  # Default to 3 most relevant documents
            )
            
            # Prepare context from retrieved documents
            context = self._prepare_context(relevant_docs)
            
            # Create the prompt with context and query
            messages = self.prompt_template.format_messages(
                context=context,
                query=query.strip()
            )
            
            # Generate response using the LLM
            response = await self.llm.ainvoke(messages)
            
            logger.info("Successfully generated RAG response")
            return response.content
            
        except ValueError:
            # Re-raise ValueError with exact message
            raise
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            raise Exception(f"Error generating response: {str(e)}")
