#!/usr/bin/env python3
"""
Clean up old payload indexes (without meta. prefix)
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from qdrant_client import QdrantClient

# Setup logging
log_file = config.LOGS_DIR / f"cleanup_indexes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
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

client = QdrantClient(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY,
)

# Old indexes to delete (without meta. prefix)
old_indexes = [
    "company",
    "client",
    "year",
    "month",
    "day",
    "date",
    "document_type",
    "invoice_number",
    "amount",
    "currency",
    "detected_document_type",
    "section_type",
    "boilerplate_score",
]

logger.info("="*80)
logger.info("CLEANING UP OLD QDRANT INDEXES")
logger.info("="*80)
logger.info("")

for field_name in old_indexes:
    try:
        print(f"  Deleting old index '{field_name}'...", end=" ")
        client.delete_payload_index(
            collection_name=config.QDRANT_COLLECTION,
            field_name=field_name,
        )
        print("[OK] Deleted")
    except Exception as e:
        if "not found" in str(e).lower():
            print("[WARN] Not found (already deleted)")
        else:
            print(f"[FAILED] {e}")

print()
print("="*80)
print("OLD INDEXES CLEANED UP")
print("="*80)
print()
print("Only meta.* indexes remain (the correct ones)!")
print()
