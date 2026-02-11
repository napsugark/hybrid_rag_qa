"""
Document Type Detector Component
Identifies document type (invoice, contract, receipt) for appropriate processing
"""

import re
import logging
from typing import List, Dict, Any
from haystack import component, Document

logger = logging.getLogger(__name__)


@component
class DocumentTypeDetector:
    """
    Detects document type from content and sets appropriate metadata
    
    Supported types:
    - factura (invoice)
    - chitanta (receipt)
    - contract (contract)
    - oferta (offer/quote)
    - raport (report)
    - email
    - other
    
    This enables type-specific processing rules downstream
    """
    
    # Romanian document type patterns
    INVOICE_PATTERNS = [
        r"(?i)\bfactur[aă]\b",
        r"(?i)\binvoice\b",
        r"(?i)(?:nr|numar|număr)[.\s]*factur",
        r"(?i)total\s+de\s+plat[ăa]",
        r"(?i)cod\s+de\s+\u00eencasare",
    ]
    
    RECEIPT_PATTERNS = [
        r"(?i)\bchitan[țt][aă]\b",
        r"(?i)\breceipt\b",
        r"(?i)bon\s+fiscal",
        r"(?i)bon\s+de\s+plat[aă]",
    ]
    
    CONTRACT_PATTERNS = [
        r"(?i)\bcontract\b",
        r"(?i)acord\s+cadru",
        r"(?i)conven[țt]ie",
        r"(?i){\s*parts\s*contrac",
    ]
    
    OFFER_PATTERNS = [
        r"(?i)\bofer[țt][aă]\b",
        r"(?i)\bquote\b",
        r"(?i)propunere\s+comercial[aă]",
        r"(?i)ofert[aă]\s+de\s+pre[țt]",
    ]
    
    def __init__(self):
        """Initialize document type detector"""
        self.compiled_patterns = {
            "factura": [re.compile(p) for p in self.INVOICE_PATTERNS],
            "chitanta": [re.compile(p) for p in self.RECEIPT_PATTERNS],
            "contract": [re.compile(p) for p in self.CONTRACT_PATTERNS],
            "oferta": [re.compile(p) for p in self.OFFER_PATTERNS],
        }
    
    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Detect document type for each document
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            Dictionary with documents annotated with document_type
        """
        if not documents:
            return {"documents": []}
        
        annotated_docs = []
        for doc in documents:
            doc_type = self._detect_type(doc.content)
            doc.meta["detected_document_type"] = doc_type
            
            # Also set processing hints based on type
            doc.meta["requires_semantic_chunking"] = doc_type in ["factura", "contract"]
            doc.meta["requires_boilerplate_filtering"] = doc_type in ["factura", "contract"]
            
            logger.info(f"Detected type '{doc_type}' for document: {doc.meta.get('file_path', 'unknown')}")
            annotated_docs.append(doc)
        
        return {"documents": annotated_docs}
    
    def _detect_type(self, content: str) -> str:
        """
        Detect document type from content
        
        Args:
            content: Document text
            
        Returns:
            Document type string
        """
        # Use first 2000 characters for detection (header area)
        sample = content[:2000]
        
        # Score each type
        scores = {}
        for doc_type, patterns in self.compiled_patterns.items():
            score = sum(1 for pattern in patterns if pattern.search(sample))
            scores[doc_type] = score
        
        # Return type with highest score
        if max(scores.values()) > 0:
            detected_type = max(scores, key=scores.get)
            logger.debug(f"Type scores: {scores} → {detected_type}")
            return detected_type
        
        return "other"


if __name__ == "__main__":
    """Test document type detector"""
    logging.basicConfig(level=logging.DEBUG)
    
    detector = DocumentTypeDetector()
    
    # Test invoice
    invoice_doc = Document(
        content="FACTURA FISCALA\nNr. factura: 12345\nTotal de plată: 1234.56 Lei",
        meta={"file_path": "test_invoice.pdf"}
    )
    
    # Test contract
    contract_doc = Document(
        content="CONTRACT DE PRESTĂRI SERVICII\nÎntre părțile contractante...",
        meta={"file_path": "test_contract.pdf"}
    )
    
    # Test receipt
    receipt_doc = Document(
        content="CHITANȚĂ\nAm primit de la...",
        meta={"file_path": "test_receipt.pdf"}
    )
    
    result = detector.run(documents=[invoice_doc, contract_doc, receipt_doc])
    
    for doc in result["documents"]:
        print(f"\nFile: {doc.meta['file_path']}")
        print(f"Detected type: {doc.meta['detected_document_type']}")
        print(f"Requires semantic chunking: {doc.meta['requires_semantic_chunking']}")
        print(f"Requires boilerplate filtering: {doc.meta['requires_boilerplate_filtering']}")
