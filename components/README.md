# Components Directory

Custom Haystack components for the RAG pipeline.

## Components

### Core Processing
- **semantic_chunker.py** - Splits documents by semantic sections (header, totals, line_items, simple_chunk)
- **metadata_enricher.py** - Extracts metadata from documents (dates, amounts, companies, etc.)
- **document_type_detector.py** - Auto-detects document type (invoice, contract, report)

### Query Processing
- **query_metadata_extractor.py** - Extracts metadata filters from user queries in Romanian

### Quality Filters
- **boilerplate_filter.py** - Filters out boilerplate/template sections
- **summarizer.py** - Generates document summaries

## Component Pipeline

```
Documents
    ↓
DocumentTypeDetector → Detect type (invoice/contract/report)
    ↓
SemanticChunker → Split by sections
    ↓
MetadataEnricher → Extract metadata (dates, amounts, etc.)
    ↓
BoilerplateFilter → Remove templates
    ↓
Summarizer → Generate summaries
    ↓
QdrantDocumentWriter → Store in Qdrant
```

## Query Pipeline

```
User Query
    ↓
QueryMetadataExtractor → Extract filters from Romanian query
    ↓
QdrantHybridRetriever → Retrieve with filters + hybrid search
    ↓
Reranker → Re-rank results
    ↓
Generator → Generate answer
```

## Development

Each component:
- Extends Haystack base classes
- Has clear input/output contracts
- Includes error handling
- Uses LLM where needed (via Azure OpenAI)
