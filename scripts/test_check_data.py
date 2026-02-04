#!/usr/bin/env python3
"""Quick check of indexed data"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src import config
from qdrant_client import QdrantClient

client = QdrantClient(
    url=config.QDRANT_URL,
    api_key=config.QDRANT_API_KEY,
)

# Get a few sample documents
print("Fetching sample documents...")
result = client.scroll(
    collection_name=config.QDRANT_COLLECTION,
    limit=5,
)

points = result[0]
print(f"\nFound {len(points)} sample documents\n")

for i, point in enumerate(points, 1):
    payload = point.payload
    print(f"Document {i}:")
    print(f"  ID: {point.id}")
    
    # Haystack stores metadata in 'meta' field or directly in payload
    # Check both locations
    meta = payload.get('meta', {})
    
    # Try direct payload first, then meta
    print(f"  File: {payload.get('file_path') or meta.get('file_path', 'N/A')}")
    print(f"  Date: {payload.get('date') or meta.get('date', 'N/A')}")
    print(f"  Year: {payload.get('year') or meta.get('year', 'N/A')}")
    print(f"  Month: {payload.get('month') or meta.get('month', 'N/A')}")
    print(f"  Day: {payload.get('day') or meta.get('day', 'N/A')}")
    print(f"  Document Type: {payload.get('document_type') or meta.get('document_type', 'N/A')}")
    print(f"  Section Type: {payload.get('section_type') or meta.get('section_type', 'N/A')}")
    print(f"  Company: {payload.get('company') or meta.get('company', 'N/A')}")
    
    # Debug: show ALL keys in payload
    print(f"  All payload keys: {list(payload.keys())}")
    if meta:
        print(f"  All meta keys: {list(meta.keys())}")
    print()

# Check collection info
print("\nCollection Info:")
info = client.get_collection(config.QDRANT_COLLECTION)
print(f"  Total points: {info.points_count}")
print(f"  Indexed payload fields:")
if info.payload_schema:
    for field, schema in info.payload_schema.items():
        print(f"    - {field}: {schema}")
else:
    print("    ⚠️  NO INDEXES FOUND!")
