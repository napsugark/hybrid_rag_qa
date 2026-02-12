#!/usr/bin/env python3
"""
Create payload indexes in Qdrant for metadata filtering
This enables efficient filtering on metadata fields like year, month, day, etc.
"""

import sys
import logging
from pathlib import Path
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import PayloadSchemaType

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config

# Setup logging
log_file = config.LOGS_DIR / f"create_indexes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
config.LOGS_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def create_metadata_indexes():
    """Create payload indexes for all metadata fields used in filtering"""
    
    logger.info("="*80)
    logger.info("CREATING QDRANT PAYLOAD INDEXES")
    logger.info("="*80)
    logger.info("")
    
    # Connect to Qdrant
    logger.info(f"Connecting to Qdrant...")
    logger.info(f"  URL: {config.QDRANT_URL}")
    logger.info(f"  Collection: {config.QDRANT_COLLECTION}")
    logger.info("")
    
    client = QdrantClient(
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
    )
    
    # Define metadata fields that need indexes for filtering
    # Haystack stores metadata in 'meta' object, so we need to index 'meta.field_name'
    metadata_indexes = {
        # From MetadataEnricher (extracted from document content)
        "meta.company": PayloadSchemaType.KEYWORD,         # FILTERABLE company name
        "meta.client": PayloadSchemaType.KEYWORD,          # FILTERABLE client name
        "meta.year": PayloadSchemaType.INTEGER,            # FILTERABLE year
        "meta.month": PayloadSchemaType.INTEGER,           # FILTERABLE month
        "meta.day": PayloadSchemaType.INTEGER,             # FILTERABLE day
        "meta.date": PayloadSchemaType.KEYWORD,            # FILTERABLE full date string
        "meta.document_type": PayloadSchemaType.KEYWORD,   # FILTERABLE doc type (from metadata)
        "meta.invoice_number": PayloadSchemaType.KEYWORD,  # FILTERABLE invoice ID
        "meta.amount": PayloadSchemaType.FLOAT,            # FILTERABLE monetary value
        "meta.currency": PayloadSchemaType.KEYWORD,        # FILTERABLE currency
        
        # From DocumentTypeDetector (production-grade component)
        "meta.detected_document_type": PayloadSchemaType.KEYWORD,  # Auto-detected type
        
        # From SemanticDocumentChunker (production-grade component)
        "meta.section_type": PayloadSchemaType.KEYWORD,    # Chunk section type
        
        # From BoilerplateFilter (production-grade component)
        "meta.boilerplate_score": PayloadSchemaType.INTEGER,  # Boilerplate detection score
        
        # Mayan EDMS integration fields (for permission filtering and deduplication)
        "meta.document_id": PayloadSchemaType.INTEGER,           # Mayan document ID
        "meta.document_version_id": PayloadSchemaType.INTEGER,   # Mayan document version ID
        "meta.allowed_users": PayloadSchemaType.INTEGER,         # User IDs with access (array)
        "meta.content_hash": PayloadSchemaType.KEYWORD,          # SHA-256 hash for content dedup
    }
    
    logger.info("Creating payload indexes for metadata fields:")
    logger.info("")
    
    for field_name, schema_type in metadata_indexes.items():
        try:
            logger.info(f"  Creating index for '{field_name}' ({schema_type})...")
            
            client.create_payload_index(
                collection_name=config.QDRANT_COLLECTION,
                field_name=field_name,
                field_schema=schema_type,
            )
            
            logger.info("    [OK] Created")
            
        except Exception as e:
            # Check if index already exists
            if "already exists" in str(e).lower():
                logger.info("    [OK] Already exists")
            else:
                logger.error(f"    [FAILED] {e}")
    
    logger.info("")
    logger.info("="*80)
    logger.info("PAYLOAD INDEXES CREATED SUCCESSFULLY")
    logger.info("="*80)
    logger.info("")
    logger.info("You can now use metadata filtering in queries!")
    logger.info("Example: query_app.py 'facturi din 18.01.2025'")
    logger.info("")


if __name__ == "__main__":
    create_metadata_indexes()
