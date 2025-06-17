import pytest
from langchain.schema import Document
from src.document_loader import SupportDocumentLoader
import json
import xml.etree.ElementTree as ET
import time
import uuid
import re

class TestSupportDocumentLoader:
    @pytest.fixture
    def sample_json_ticket(self):
        """Sample JSON ticket data for testing"""
        return {
            "subject": "Login Error",
            "body": "Unable to login to the application using Chrome browser.",
            "answer": "Please clear browser cache and cookies, then try again.",
            "type": "Bug",
            "queue": "Technical Support",
            "priority": "high",
            "language": "en",
            "tag_1": "Browser",
            "tag_2": "Login",
            "tag_3": "Chrome",
            "tag_4": "Cache",
            "tag_5": "NaN",
            "tag_6": "NaN",
            "tag_7": "NaN",
            "tag_8": "NaN",
            "Ticket ID": "test-123"
        }

    @pytest.fixture
    def sample_xml_ticket(self):
        """Create sample XML ticket content"""
        return """<?xml version='1.0' encoding='utf-8'?>
        <Tickets>
            <Ticket>
                <subject>Login Error</subject>
                <body>Unable to login to the application using Chrome browser.</body>
                <answer>Please clear browser cache and cookies, then try again.</answer>
                <type>Bug</type>
                <queue>Technical Support</queue>
                <priority>high</priority>
                <language>en</language>
                <tag_1>Browser</tag_1>
                <tag_2>Login</tag_2>
                <tag_3>Chrome</tag_3>
                <tag_4>Cache</tag_4>
                <tag_5>nan</tag_5>
                <tag_6>nan</tag_6>
                <tag_7>nan</tag_7>
                <tag_8>nan</tag_8>
                <TicketID>test-234</TicketID>
            </Ticket>
        </Tickets>"""

    @pytest.fixture
    def test_files(self, tmp_path, sample_json_ticket, sample_xml_ticket):
        """Create temporary test files"""
        # Create directories
        data_dir = tmp_path / "support_tickets"
        data_dir.mkdir()

        # Create JSON file
        json_file = data_dir / "Technical Support_tickets.json"
        with open(json_file, 'w') as f:
            json.dump([sample_json_ticket], f)

        # Create XML file
        xml_file = data_dir / "Technical Support_tickets.xml"
        with open(xml_file, 'w') as f:
            f.write(sample_xml_ticket)

        return str(data_dir)
    

    def test_load_json_tickets(self, test_files):
        """Test loading tickets from JSON file"""
        loader = SupportDocumentLoader(test_files)
        documents = loader.load_tickets()

        # print(documents)
        assert 'technical' in documents
        assert len(documents['technical']) > 0
        
        doc = documents['technical'][0]

        assert isinstance(doc, Document)
        assert "Login Error" in doc.page_content
        assert "Chrome browser" in doc.page_content
        assert doc.metadata["original_ticket_id"] == "test-123"
        assert doc.metadata['ticket_id'] == "technical_test-123"
        assert "Browser" in doc.metadata['tags']
        assert "Login" in doc.metadata['tags']

    def test_load_xml_tickets(self, test_files):
        """Test loading tickets from XML file"""
        loader = SupportDocumentLoader(test_files)
        documents = loader.load_tickets()

        assert 'technical' in documents
        assert len(documents['technical']) > 0
        doc = documents['technical'][1]
        assert isinstance(doc, Document)
        assert "Login Error" in doc.page_content
        assert "Chrome browser" in doc.page_content
        assert doc.metadata['ticket_id'] == "technical_xml_test-234"
        assert doc.metadata['original_ticket_id'] == "test-234"
        assert "Browser" in doc.metadata['tags']
        assert "Login" in doc.metadata['tags']
    
    def test_create_documents(self, test_files):
        """Test creating documents from both JSON and XML files"""
        loader = SupportDocumentLoader(test_files)
        documents = loader.create_documents()

        # Check if documents are created for each support type
        assert 'technical' in documents
        assert isinstance(documents['technical'], list)
        
        # Verify document content and metadata
        for doc in documents['technical']:
            assert isinstance(doc, Document)
            assert 'ticket_id' in doc.metadata
            assert 'tags' in doc.metadata
            assert isinstance(doc.metadata['tags'], list)

    def test_file_not_found(self, tmp_path):
        """Test handling of non-existent directory"""
        nonexistent_path = tmp_path / "nonexistent"
        with pytest.raises(FileNotFoundError):
            SupportDocumentLoader(str(nonexistent_path))

    def test_invalid_json(self, tmp_path):
        """Test handling of invalid JSON file"""
        data_dir = tmp_path / "support_tickets"
        data_dir.mkdir()
        
        invalid_file = data_dir / "Technical Support_tickets.json"
        invalid_file.write_text("invalid json content")

        loader = SupportDocumentLoader(str(data_dir))
        documents = loader.load_tickets()
        
        # Should return empty list for technical support due to invalid file
        assert documents['technical'] == []
    

    def test_json_metadata_fields(self, test_files):
        """Test that all required metadata fields are correctly extracted from JSON"""
        loader = SupportDocumentLoader(test_files)
        documents = loader.load_tickets()
        
        assert 'technical' in documents
        assert len(documents['technical']) > 0
        
        # Find the JSON document (has source='json')
        json_docs = [doc for doc in documents['technical'] if doc.metadata.get('source') == 'json']
        assert len(json_docs) > 0
        
        doc = json_docs[0]
        
        # Check source field
        assert doc.metadata['source'] == 'json'
        
        # Check priority field
        assert doc.metadata['priority'] == 'high'
        
        # Check subject field
        assert doc.metadata['subject'] == 'Login Error'
        
        # Check other important fields
        assert doc.metadata['type'] == 'Bug'
        assert doc.metadata['queue'] == 'Technical Support'
        assert doc.metadata['language'] == 'en'
        
        # Check content fields used for formatting
        assert 'subject' in doc.metadata
        assert 'body' in doc.metadata
        assert 'answer' in doc.metadata

    def test_xml_metadata_fields(self, test_files):
        """Test that all required metadata fields are correctly extracted from XML"""
        loader = SupportDocumentLoader(test_files)
        documents = loader.load_tickets()
        
        assert 'technical' in documents
        assert len(documents['technical']) > 0
        
        # Find the XML document (has source='xml')
        xml_docs = [doc for doc in documents['technical'] if doc.metadata.get('source') == 'xml']
        assert len(xml_docs) > 0
        
        doc = xml_docs[0]
        
        # Check source field
        assert doc.metadata['source'] == 'xml'
        
        # Check priority field
        assert doc.metadata['priority'] == 'high'
        
        # Check other important fields
        assert doc.metadata['type'] == 'Bug'
        assert doc.metadata['queue'] == 'Technical Support'
        assert doc.metadata['language'] == 'en'

    def test_content_formatting(self, test_files):
        """Test that content is properly formatted for both JSON and XML documents"""
        loader = SupportDocumentLoader(test_files)
        documents = loader.load_tickets()
        
        assert 'technical' in documents
        assert len(documents['technical']) > 0
        
        for doc in documents['technical']:
            # Check that content follows the required format
            content = doc.page_content.strip()
            
            # Check required content fields
            assert "Subject:" in content
            assert "Description:" in content
            assert "Resolution:" in content
            assert "Type:" in content
            assert "Queue:" in content
            assert "Priority:" in content
            
            # Check specific content values
            assert "Login Error" in content
            assert "Chrome browser" in content
            assert "clear browser cache" in content
            
            # The source field should be correctly set
            if "json" in doc.metadata['ticket_id']:
                assert doc.metadata['source'] == 'json'
            elif "xml" in doc.metadata['ticket_id']:
                assert doc.metadata['source'] == 'xml'