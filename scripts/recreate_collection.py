#!/usr/bin/env python3
"""
Recreate Qdrant collection from scratch
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from src.app import HybridRAGApplication

# Setup logging
log_file = config.LOGS_DIR / f"recreate_collection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

logger.info("="*80)
logger.info("RECREATING QDRANT COLLECTION")
logger.info("="*80)
logger.info("")
logger.info(f"Collection: {config.QDRANT_COLLECTION}")
logger.info("")

# Initialize app (this will create the collection if it doesn't exist)
logger.info("Initializing RAG application...")
app = HybridRAGApplication()

# Delete and recreate collection
logger.info("Deleting old collection...")
app.recreate_collection()
logger.info("[OK] Collection recreated")
logger.info("")

# Re-create indexes
logger.info("Creating payload indexes...")
from scripts.create_payload_indexes import create_metadata_indexes
create_metadata_indexes()
logger.info("")

# Re-index documents
logger.info("Re-indexing documents...")
docs = app.load_documents_from_folder(config.DOCUMENTS_DIR)
logger.info(f"Loaded {len(docs)} documents")
logger.info("")

final_count = app.index_documents(docs, skip_duplicates=False)
logger.info(f"[OK] Indexed {final_count} chunks")
logger.info("")

logger.info("="*80)
logger.info("COLLECTION RECREATED SUCCESSFULLY!")
logger.info("="*80)
logger.info("")
logger.info("Now you can run queries:")
logger.info("  python query_app.py")
logger.info("")
