from typing import List, Dict, Any
import os
import chromadb
from chromadb.config import Settings, DEFAULT_TENANT
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
import logging
import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

openai_api = os.getenv("OPENAI_API_KEY")

logger = logging.getLogger(__name__)

class SupportVectorStore:
    """
    A class to manage the vector store for support tickets using ChromaDB.
    
    This class handles the creation, storage, and retrieval of vector embeddings
    for technical, product, and customer support tickets in separate collections.
    
    IMPORTANT:
    - Empty queries (null or whitespace-only) must be rejected with an empty result list
    - Queries shorter than 10 characters must be rejected with an empty result list
    - All metadata must be properly processed for ChromaDB compatibility
    - Embedding model to be used should be OpenAI text-embedding-ada-002
    """
    
    def __init__(self, vecstore_path):
        """Initialize the vector store with ChromaDB client and OpenAI embeddings."""
        
         
        self.vecstore_path = vecstore_path
        
        # Initialize ChromaDB client with persistent storage
        self.client = chromadb.PersistentClient(
            path=vecstore_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize OpenAI embeddings with ada-002 model
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api
        )
        
        # Dictionary to store collection references
        self.collections = {}
        
        logger.info(f"Initialized SupportVectorStore with path: {vecstore_path}")


    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare metadata for ChromaDB by converting lists to strings and ensuring valid types.
        
        ChromaDB requires all metadata values to be primitive types (str, int, float, bool).
        Lists must be converted to comma-separated strings, and None values must be handled appropriately.
        
        Args:
            metadata (Dict[str, Any]): Original metadata dictionary
            
        Returns:
            Dict[str, Any]: Processed metadata with ChromaDB-compatible types
        """
        
        processed_metadata = {}
        
        for key, value in metadata.items():
            if value is None:
                # Convert None to empty string
                processed_metadata[key] = ""
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                processed_metadata[key] = ",".join(str(item) for item in value if item is not None)
            elif isinstance(value, (str, int, float, bool)):
                # Keep primitive types as-is
                processed_metadata[key] = value
            else:
                # Convert other types to string
                processed_metadata[key] = str(value)
        
        return processed_metadata



    def _process_metadata_for_return(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process metadata when retrieving from ChromaDB, converting string-lists back to actual lists.
        
        This function reverses the transformations done in _prepare_metadata() to ensure
        that metadata is returned in the expected format.
        
        Args:
            metadata (Dict[str, Any]): Metadata from ChromaDB
            
        Returns:
            Dict[str, Any]: Processed metadata with proper types
        """
        processed_metadata = {}
        
        # List of fields that should be converted back to lists
        list_fields = ['tags']
        
        for key, value in metadata.items():
            if key in list_fields and isinstance(value, str):
                # Convert comma-separated strings back to lists
                if value.strip():
                    processed_metadata[key] = [item.strip() for item in value.split(',') if item.strip()]
                else:
                    processed_metadata[key] = []
            else:
                # Keep other values as-is
                processed_metadata[key] = value
        
        return processed_metadata



    def create_vector_store(self, documents_by_type: Dict[str, List[Document]]) -> None:
        """
        Create vector store collections from documents, organized by support type.
        
        Args:
            documents_by_type (Dict[str, List[Document]]): Dictionary of documents organized by support type
        """
        # Create collection for each support type
        
        for support_type, documents in documents_by_type.items():
            if not documents:
                logger.warning(f"No documents found for support type: {support_type}")
                continue
            
            collection_name = f"support_{support_type}"
            logger.info(f"Creating collection: {collection_name} with {len(documents)} documents")
            
            try:
                # Create or get collection
                collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"support_type": support_type}
                )
                
                # Prepare data for ChromaDB
                texts = []
                metadatas = []
                ids = []
                
                for doc in documents:
                    # Use ticket_id as document ID, fallback to generated ID if not available
                    doc_id = doc.metadata.get('ticket_id')
                    if not doc_id:
                        import uuid
                        doc_id = str(uuid.uuid4())
                    
                    texts.append(doc.page_content)
                    metadatas.append(self._prepare_metadata(doc.metadata))
                    ids.append(doc_id)
                
                # Generate embeddings
                logger.info(f"Generating embeddings for {len(texts)} documents in {collection_name}")
                embeddings = self.embeddings.embed_documents(texts)
                
                # Add documents to collection
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids,
                    embeddings=embeddings
                )
                
                # Store collection reference
                self.collections[support_type] = collection
                
                logger.info(f"Successfully created collection {collection_name} with {len(documents)} documents")
                
            except Exception as e:
                logger.error(f"Error creating collection for {support_type}: {e}")
                raise


    @classmethod
    def load_local(cls, directory: str) -> 'SupportVectorStore':
        """
        Load a vector store from local storage.
        
        Args:
            directory (str): Directory path containing the vector store
            
        Returns:
            SupportVectorStore: Loaded vector store instance
        """
        # Create new instance with the directory
        instance = cls(directory) 
        
        # Load all collections
        try:
            # Get all collection names from ChromaDB
            collections = instance.client.list_collections()
            
            # Load collections that match our support naming pattern
            for collection_info in collections:
                collection_name = collection_info.name
                if collection_name.startswith("support_"):
                    support_type = collection_name.replace("support_", "")
                    collection = instance.client.get_collection(collection_name)
                    instance.collections[support_type] = collection
                    logger.info(f"Loaded collection: {collection_name}")
            
            logger.info(f"Successfully loaded vector store from {directory}")
            return instance
            
        except Exception as e:
            logger.error(f"Error loading vector store from {directory}: {e}")
            raise

    def query_similar(
        self, 
        query: str, 
        support_type: str = None, 
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query the vector store for similar documents.
        
        IMPORTANT:
        - Empty queries (null or whitespace-only) MUST return an empty list
        - When the query is null or whitespace-only, log a warning but DO NOT raise an exception
        - Non-existent support types MUST return an empty list with an appropriate warning
        
        Args:
            query (str): Query text to find similar documents
            support_type (str, optional): Specific support type to query. If None, queries all types
            k (int): Number of similar documents to return per collection
            
        Returns:
            List[Dict[str, Any]]: List of similar documents with their metadata, each containing:
            - 'content': Document content
            - 'metadata': Document metadata
            - 'similarity': Similarity score (1 - distance)
        """
        if not query or not query.strip():
            logger.warning("Empty or null query provided, returning empty results")
            return []
        
        # Check for queries shorter than 10 characters
        if len(query.strip()) < 10:
            logger.warning(f"Query too short ({len(query.strip())} characters), returning empty results")
            return []
        
        results = []
        
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Determine which collections to query
            collections_to_query = {}
            if support_type:
                if support_type in self.collections:
                    collections_to_query[support_type] = self.collections[support_type]
                else:
                    logger.warning(f"Support type '{support_type}' not found in vector store")
                    return []
            else:
                collections_to_query = self.collections
            
            # Query each collection
            for collection_type, collection in collections_to_query.items():
                try:
                    # Query the collection
                    query_results = collection.query(
                        query_embeddings=[query_embedding],
                        n_results=k,
                        include=['documents', 'metadatas', 'distances']
                    )
                    
                    # Process results
                    if query_results['documents'] and query_results['documents'][0]:
                        documents = query_results['documents'][0]
                        metadatas = query_results['metadatas'][0]
                        distances = query_results['distances'][0]
                        
                        for doc, metadata, distance in zip(documents, metadatas, distances):
                            # Convert distance to similarity score (1 - distance)
                            similarity = 1.0 - distance
                            
                            # Process metadata to restore original format
                            processed_metadata = self._process_metadata_for_return(metadata)
                            
                            result = {
                                'content': doc,
                                'metadata': processed_metadata,
                                'similarity': similarity
                            }
                            results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error querying collection {collection_type}: {e}")
                    continue
            
            # Sort results by similarity score (highest first)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            
            logger.info(f"Found {len(results)} similar documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Error during similarity search: {e}")
            return []


    def get_support_types(self) -> List[str]:
        """
        Get list of available support types in the vector store.
        
        Returns:
            List[str]: List of support type names
        """
        return list(self.collections.keys())
