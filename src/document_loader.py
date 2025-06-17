from typing import List, Dict, Any
from pathlib import Path
import xml.etree.ElementTree as ET
from langchain.schema import Document
from langchain_community.document_loaders import JSONLoader
import logging
import jq
from uuid import uuid4
from functools import partial

logger = logging.getLogger(__name__)

class SupportDocumentLoader:
    """
    A class to load and process support tickets from JSON and XML files using LangChain loaders.
    
    This loader uses LangChain's JSONLoader and custom XML loading to process support tickets and
    converts them into a standardized document format for the RAG system.
    
    IMPORTANT: 
    - Even when using LangChain loaders, you MUST use the custom get_json_content and 
      get_json_metadata functions to ensure consistent document formatting
    - Ensure all ticket IDs are unique across the entire dataset
    - The format of ticket IDs must follow the pattern: "{support_type}_{original_id}" for JSON
      and "{support_type}_xml_{original_id}" for XML files
    """
    
    def __init__(self, data_path: str):
        """
        Initialize the document loader with the path to data files.
        
        Args:
            data_path (str): Directory path containing support ticket files
            
        Raises:
            FileNotFoundError: If the specified data path does not exist
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path does not exist: {data_path}")
        
        logger.info(f"Initialized SupportDocumentLoader with data path: {data_path}")

    def get_json_content(self, data: Dict[str, Any]) -> str:
        """
        Format JSON data into a standardized content string.
        
        This function MUST produce content in the exact format shown below to ensure
        consistent document formatting across the system.
        
        Args:
            data (Dict[str, Any]): Raw JSON data
            
        Returns:
            str: Formatted content string with the exact format:
            
            Subject: {}
            Description: {}
            Resolution: {}
            Type: {}
            Queue: {}
            Priority: {}
        """
        subject = data.get('subject', '')
        description = data.get('body', '')
        resolution = data.get('answer', '')
        ticket_type = data.get('type', '')
        queue = data.get('queue', '')
        priority = data.get('priority', '')
        
        content = f"""Subject: {subject}
Description: {description}
Resolution: {resolution}
Type: {ticket_type}
Queue: {queue}
Priority: {priority}"""
        
        return content



    def get_json_metadata(self, record: Dict[str, Any], support_type: str = None) -> Dict[str, Any]:
        """
        Extract metadata from JSON data.
        
        This function MUST produce metadata with all the required fields shown below.
        The 'ticket_id' MUST follow the format "{support_type}_{original_id}" to ensure
        proper document identification.
        
        Args:
            record (Dict[str, Any]): Raw JSON record
            support_type (str, optional): Type of support (technical, product, customer)
            
        Returns:
            Dict[str, Any]: Extracted metadata with the exact format:
            {
                'ticket_id': "{support_type}_{original_id}",  # Unique ID
                'original_ticket_id': str,    # Original ticket ID from JSON ("Ticket ID" field)
                'support_type': str,          # Type of support (technical, product, customer)
                'type': str,                  # Type field from original data
                'queue': str,                 # Queue information
                'priority': str,              # Priority level
                'language': str,              # Content language
                'tags': List[str],            # List of tags from tag_1 through tag_8
                'source': 'json',             # Source format identifier
                'subject': str,               # Subject field for content formatting
                'body': str,                  # Body field for content formatting
                'answer': str                 # Answer field for content formatting
            }
            
        Raises:
            ValueError: If support_type is not provided
        """
        # Get the correct support type, either from parameter or from record
        if not support_type:
            # Try to infer from queue field
            queue = record.get('queue', '').lower()
            if 'technical' in queue:
                support_type = 'technical'
            elif 'product' in queue:
                support_type = 'product'
            elif 'customer' in queue:
                support_type = 'customer'
            else:
                raise ValueError("support_type must be provided")
        
        # Extract original ticket ID
        original_ticket_id = str(record.get('Ticket ID', record.get('ticket_id', '')))
        
        # Generate unique ticket ID
        ticket_id = f"{support_type}_{original_ticket_id}"
        
        # Extract tags from tag_1 through tag_8, filtering out NaN and empty values
        tags = []
        for i in range(1, 9):
            tag_key = f'tag_{i}'
            tag_value = record.get(tag_key)
            if tag_value and str(tag_value).lower() not in ['nan', 'none', '']:
                tags.append(str(tag_value))
        
        metadata = {
            'ticket_id': ticket_id,
            'original_ticket_id': original_ticket_id,
            'support_type': support_type,
            'type': record.get('type', ''),
            'queue': record.get('queue', ''),
            'priority': record.get('priority', ''),
            'language': record.get('language', 'en'),
            'tags': tags,
            'source': 'json',
            'subject': record.get('subject', ''),
            'body': record.get('body', ''),
            'answer': record.get('answer', '')
        }
        
        return metadata


    def load_xml_tickets(self, file_path: Path, support_type: str) -> List[Document]:
        """
        Load tickets from an XML file.
        
        XML tickets MUST be processed to follow the same content and metadata format
        as JSON tickets, with the only difference being the 'ticket_id' format and
        'source' field.
        
        Args:
            file_path (Path): Path to the XML file
            support_type (str): Type of support (technical, product, customer)
            
        Returns:
            List[Document]: List of Document objects with the following format:
            
            Content format:
            Subject: {}
            Description: {}
            Resolution: {}
            Type: {}
            Queue: {}
            Priority: {}
            
            Metadata format:
            {
                'ticket_id': "{support_type}_xml_{original_id}",  # Unique ID
                'original_ticket_id': str,    # Original ticket ID from XML
                'support_type': str,          # Type of support
                'type': str,                  # Type field
                'queue': str,                 # Queue information
                'priority': str,              # Priority level
                'language': str,              # Content language
                'tags': List[str],            # List of tags
                'source': 'xml'               # Source format identifier
            }
        """
        try:
            # Parse XML file
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            documents = []
            
            # Process each ticket in the XML
            for ticket_elem in root.findall('.//Ticket'):
                # Extract ticket data
                ticket_data = {}
                
                # Extract basic fields
                ticket_data['subject'] = self._get_xml_text(ticket_elem, 'subject')
                ticket_data['body'] = self._get_xml_text(ticket_elem, 'body')
                ticket_data['answer'] = self._get_xml_text(ticket_elem, 'answer')
                ticket_data['type'] = self._get_xml_text(ticket_elem, 'type')
                ticket_data['queue'] = self._get_xml_text(ticket_elem, 'queue')
                ticket_data['priority'] = self._get_xml_text(ticket_elem, 'priority')
                ticket_data['language'] = self._get_xml_text(ticket_elem, 'language')
                
                # Extract ticket ID (handle both TicketID and ticket_id formats)
                original_ticket_id = (self._get_xml_text(ticket_elem, 'TicketID') or 
                                    self._get_xml_text(ticket_elem, 'ticket_id') or 
                                    str(uuid4()))
                
                # Extract tags
                tags = []
                for i in range(1, 9):
                    tag_value = self._get_xml_text(ticket_elem, f'tag_{i}')
                    if tag_value and tag_value.lower() not in ['nan', 'none', '']:
                        tags.append(tag_value)
                
                # Create content in the required format
                content = f"""Subject: {ticket_data['subject']}
Description: {ticket_data['body']}
Resolution: {ticket_data['answer']}
Type: {ticket_data['type']}
Queue: {ticket_data['queue']}
Priority: {ticket_data['priority']}"""
                
                # Create metadata
                metadata = {
                    'ticket_id': f"{support_type}_xml_{original_ticket_id}",
                    'original_ticket_id': original_ticket_id,
                    'support_type': support_type,
                    'type': ticket_data['type'],
                    'queue': ticket_data['queue'],
                    'priority': ticket_data['priority'],
                    'language': ticket_data['language'],
                    'tags': tags,
                    'source': 'xml'
                }
                
                # Create Document object
                document = Document(page_content=content, metadata=metadata)
                documents.append(document)
            
            logger.info(f"Loaded {len(documents)} tickets from XML file: {file_path}")
            return documents
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error in {file_path}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading XML tickets from {file_path}: {e}")
            raise

    def _get_xml_text(self, element: ET.Element, tag_name: str) -> str:
        """
        Helper method to safely extract text from XML element.
        
        Args:
            element (ET.Element): XML element
            tag_name (str): Tag name to find
            
        Returns:
            str: Text content or empty string if not found
        """
        child = element.find(tag_name)
        if child is not None and child.text:
            return child.text.strip()
        return ''
        
    def load_tickets(self) -> Dict[str, List[Document]]:
        """
        Load all support tickets using LangChain loaders, organized by support type.
        
        IMPORTANT:
        - When using JSONLoader, you MUST create a custom function that properly passes
          the support_type parameter to get_json_metadata
        - Validate that all ticket IDs are unique across the entire dataset
        
        Returns:
            Dict[str, List[Document]]: Dictionary with support types as keys and lists of Documents as values
            
        Raises:
            ValueError: If duplicate ticket IDs are found
        """
        all_documents = {}
        all_ticket_ids = set()
        
        # Define support types to look for
        support_types = ['technical', 'product', 'customer']
        
        for support_type in support_types:
            documents = []
            
            # Look for JSON files
            json_pattern = f"*{support_type}*.json"
            json_files = list(self.data_path.glob(json_pattern))
            
            for json_file in json_files:
                try:
                    logger.info(f"Loading JSON file: {json_file}")
                    
                    # Create a partial function that includes the support_type
                    metadata_func = partial(self.get_json_metadata, support_type=support_type)
                    
                    # Use JSONLoader with custom content and metadata functions
                    loader = JSONLoader(
                        file_path=str(json_file),
                        jq_schema='.[:]',  # Load all items from JSON array
                        content_key=None,  # We'll use custom content function
                        metadata_func=metadata_func,
                        text_content=False
                    )
                    
                    # Load documents
                    json_docs = loader.load()
                    
                    # Process each document to apply custom content formatting
                    for doc in json_docs:
                        # The metadata should already be set by metadata_func
                        # We need to reformat the content using our custom function
                        try:
                            # Parse the original data from page_content if it's JSON string
                            if isinstance(doc.page_content, str):
                                import json
                                original_data = json.loads(doc.page_content)
                            else:
                                original_data = doc.page_content
                            
                            # Apply custom content formatting
                            formatted_content = self.get_json_content(original_data)
                            
                            # Create new document with formatted content
                            new_doc = Document(
                                page_content=formatted_content,
                                metadata=doc.metadata
                            )
                            
                            # Check for duplicate ticket IDs
                            ticket_id = new_doc.metadata.get('ticket_id')
                            if ticket_id in all_ticket_ids:
                                raise ValueError(f"Duplicate ticket ID found: {ticket_id}")
                            all_ticket_ids.add(ticket_id)
                            
                            documents.append(new_doc)
                            
                        except Exception as e:
                            logger.error(f"Error processing document: {e}")
                            continue
                    
                    logger.info(f"Loaded {len(json_docs)} documents from {json_file}")
                    
                except Exception as e:
                    logger.error(f"Error loading JSON file {json_file}: {e}")
                    continue
            
            # Look for XML files
            xml_pattern = f"*{support_type}*.xml"
            xml_files = list(self.data_path.glob(xml_pattern))
            
            for xml_file in xml_files:
                try:
                    logger.info(f"Loading XML file: {xml_file}")
                    xml_docs = self.load_xml_tickets(xml_file, support_type)
                    
                    # Check for duplicate ticket IDs
                    for doc in xml_docs:
                        ticket_id = doc.metadata.get('ticket_id')
                        if ticket_id in all_ticket_ids:
                            raise ValueError(f"Duplicate ticket ID found: {ticket_id}")
                        all_ticket_ids.add(ticket_id)
                    
                    documents.extend(xml_docs)
                    logger.info(f"Loaded {len(xml_docs)} documents from {xml_file}")
                    
                except Exception as e:
                    logger.error(f"Error loading XML file {xml_file}: {e}")
                    continue
            
            if documents:
                all_documents[support_type] = documents
                logger.info(f"Total documents for {support_type}: {len(documents)}")
        
        total_docs = sum(len(docs) for docs in all_documents.values())
        logger.info(f"Loaded total of {total_docs} documents across all support types")
        logger.info(f"Total unique ticket IDs: {len(all_ticket_ids)}")
        
        return all_documents



    def create_documents(self) -> Dict[str, List[Document]]:
        """
        Load and process all support tickets into LangChain Document objects.
        
        Returns:
            Dict[str, List[Document]]: Dictionary with support types as keys and lists of Document objects as values
        """

        return self.load_tickets()
