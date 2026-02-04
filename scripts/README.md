# Scripts Directory

Utility scripts for managing the RAG system.

## Scripts

### Document Indexing
- **index_documents.py** - Index documents from data/documents_ro/ to Qdrant
  ```bash
  poetry run python scripts/index_documents.py
  ```

### Qdrant Management
- **create_payload_indexes.py** - Create metadata indexes for filtering
  ```bash
  poetry run python scripts/create_payload_indexes.py
  ```

- **recreate_collection.py** - Delete and recreate entire Qdrant collection
  ```bash
  poetry run python scripts/recreate_collection.py
  ```

- **cleanup_old_indexes.py** - Clean up old payload indexes (maintenance)
  ```bash
  poetry run python scripts/cleanup_old_indexes.py
  ```

### Debugging
- **debug_section_type.py** - Check section_type values in indexed documents
  ```bash
  poetry run python scripts/debug_section_type.py
  ```

- **test_check_data.py** - Quick check of indexed data
  ```bash
  poetry run python scripts/test_check_data.py
  ```

## Typical Workflow

1. **First time setup:**
   ```bash
   poetry run python scripts/create_payload_indexes.py
   poetry run python scripts/index_documents.py
   ```

2. **Adding new documents:**
   ```bash
   # Just add files to data/documents_ro/ and run:
   poetry run python scripts/index_documents.py
   ```

3. **Full reset (if needed):**
   ```bash
   poetry run python scripts/recreate_collection.py
   ```
