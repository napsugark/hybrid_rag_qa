"""Debug section_type field values in Qdrant"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchAny
from src import config

client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

# Get sample documents
result = client.scroll(
    collection_name=config.QDRANT_COLLECTION, 
    limit=20, 
    with_payload=['meta']
)

print("Sample documents section_type values:")
section_types_found = set()
docs_without_section_type = 0

for p in result[0]:
    meta = p.payload.get("meta", {})
    st = meta.get("section_type")
    dt = meta.get("document_type")
    if st:
        section_types_found.add(st)
    else:
        docs_without_section_type += 1
    print(f"  {p.id}: section_type={st}, doc_type={dt}")

print(f"\nUnique section_type values found: {section_types_found}")
print(f"Documents without section_type: {docs_without_section_type}")

# Test filtering with MatchAny
print("\n\nTesting MatchAny filter on section_type...")
try:
    result = client.scroll(
        collection_name=config.QDRANT_COLLECTION,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="meta.section_type",
                    match=MatchAny(any=["totals", "line_items", "header"])
                )
            ]
        ),
        limit=5,
        with_payload=['meta.section_type']
    )
    print(f"Filter works! Found {len(result[0])} documents")
except Exception as e:
    print(f"Filter failed: {e}")
