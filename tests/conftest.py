import json
import os
import tempfile
from pathlib import Path
import xml.etree.ElementTree as ET
import pytest

@pytest.fixture
def sample_tickets():
    """Sample support ticket data for testing"""
    return {
        'technical': [
            {
                "subject": "Browser Login Issue",
                "body": "Unable to login using Safari browser.",
                "answer": "Clear browser cache and cookies.",
                "type": "Technical",
                "queue": "Tech Support",
                "priority": "high",
                "language": "en",
                "tag_1": "Browser",
                "tag_2": "Login",
                "tag_3": "Safari",
                "tag_4": "Technical",
                "Ticket ID": "tech-001"
            },
            {
                "subject": "Application Crash",
                "body": "Application crashes on startup.",
                "answer": "Update to latest version.",
                "type": "Technical",
                "queue": "Tech Support",
                "priority": "high",
                "language": "en",
                "tag_1": "Crash",
                "tag_2": "Startup",
                "tag_3": "Version",
                "Ticket ID": "tech-002"
            }
        ],
        'product': [
            {
                "subject": "Feature Request",
                "body": "Need dark mode feature.",
                "answer": "Dark mode is planned for next release.",
                "type": "Enhancement",
                "queue": "Product",
                "priority": "medium",
                "language": "en",
                "tag_1": "Feature",
                "tag_2": "UI",
                "tag_3": "Enhancement",
                "Ticket ID": "prod-001"
            }
        ],
        'customer': [
            {
                "subject": "Billing Question",
                "body": "Need clarification on subscription pricing.",
                "answer": "Subscription details provided with pricing tiers.",
                "type": "Inquiry",
                "queue": "Customer Service",
                "priority": "low",
                "language": "en",
                "tag_1": "Billing",
                "tag_2": "Subscription",
                "tag_3": "Pricing",
                "Ticket ID": "cust-001"
            }
        ]
    }

def create_json_file(tickets: dict, file_path: Path):
    """Create a JSON file with given tickets"""
    with open(file_path, 'w') as f:
        json.dump(tickets, f, indent=2)

def create_xml_file(tickets: list, file_path: Path):
    """Create an XML file with given tickets"""
    root = ET.Element("Tickets")
    for ticket in tickets:
        ticket_elem = ET.SubElement(root, "Ticket")
        for key, value in ticket.items():
            elem = ET.SubElement(ticket_elem, key.replace(" ", "_"))
            elem.text = str(value)
    
    tree = ET.ElementTree(root)
    tree.write(file_path)

@pytest.fixture
def test_environment(sample_tickets, tmp_path):
    """Create test environment with support ticket data files"""
    # Create directories
    data_dir = tmp_path / "support_tickets"
    vector_store_dir = tmp_path / "vector_store"
    data_dir.mkdir()
    vector_store_dir.mkdir()

    # Create files for each support type
    for support_type, tickets in sample_tickets.items():
        # Create JSON file
        json_file = data_dir / f"{support_type}_support.json"
        create_json_file(tickets, json_file)

        # Create XML file
        xml_file = data_dir / f"{support_type}_support.xml"
        create_xml_file(tickets, xml_file)

    return {
        "tmp_dir": str(tmp_path),
        "data_dir": str(data_dir),
        "vector_store_dir": str(vector_store_dir)
    }

@pytest.fixture
def mock_openai_env(monkeypatch):
    """Mock OpenAI environment variables"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-key")

@pytest.fixture
def mock_embeddings():
    """Mock embeddings for testing"""
    return [
        [0.1, 0.2, 0.3],  # Simplified 3D embeddings for testing
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ]

@pytest.fixture
def mock_llm_response():
    """Mock LLM response for testing"""
    return {
        "technical": "To fix the browser login issue:\n1. Clear cache and cookies\n2. Restart browser",
        "product": "Dark mode feature is planned for the next release cycle in Q2",
        "customer": "Our subscription pricing tiers are as follows:\n1. Basic: $10/mo\n2. Pro: $25/mo"
    }