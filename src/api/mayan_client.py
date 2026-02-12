"""
Mayan EDMS API Client

Handles authentication and document retrieval from Mayan EDMS
"""

import os
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import httpx
from pathlib import Path
import tempfile

logger = logging.getLogger("HybridRAG.MayanClient")


@dataclass
class MayanDocument:
    """Represents a document fetched from Mayan EDMS"""
    id: int
    label: str
    description: str
    document_type: str
    language: str
    datetime_created: str
    file_content: Optional[bytes] = None
    ocr_content: Optional[str] = None
    metadata: Dict[str, Any] = None
    filename: Optional[str] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def file_extension(self) -> str:
        """Get lowercase file extension from filename"""
        if self.filename:
            return Path(self.filename).suffix.lower()
        return Path(self.label).suffix.lower()
    
    @property
    def is_image(self) -> bool:
        """Check if the document is an image file"""
        return self.file_extension in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


class MayanClient:
    """
    Client for Mayan EDMS REST API v4
    
    Supports:
    - Token-based authentication
    - Document metadata retrieval
    - Document file download
    - OCR content retrieval
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        api_token: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or os.getenv("MAYAN_URL", "")).rstrip("/")
        self.api_token = api_token or os.getenv("MAYAN_API_TOKEN")
        self.username = username or os.getenv("MAYAN_USERNAME")
        self.password = password or os.getenv("MAYAN_PASSWORD")
        self.timeout = timeout
        
        if not self.base_url:
            raise ValueError(
                "Mayan URL not configured. Set MAYAN_URL environment variable "
                "or pass base_url parameter."
            )
        
        self._client = httpx.Client(timeout=self.timeout)
        self._headers = {}
        
        # Set up authentication
        if self.api_token:
            self._headers["Authorization"] = f"Token {self.api_token}"
            logger.info("Using API token authentication")
        elif self.username and self.password:
            # Will use basic auth per request
            logger.info("Using username/password authentication")
        else:
            raise ValueError(
                "Mayan credentials not configured. Set MAYAN_API_TOKEN or "
                "MAYAN_USERNAME + MAYAN_PASSWORD environment variables."
            )
    
    @property
    def auth(self) -> Optional[tuple]:
        """Return auth tuple for basic auth if no token"""
        if not self.api_token and self.username and self.password:
            return (self.username, self.password)
        return None
    
    def _get(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make GET request to Mayan API"""
        url = f"{self.base_url}/api/v4{endpoint}"
        response = self._client.get(
            url, 
            headers=self._headers, 
            auth=self.auth,
            **kwargs
        )
        response.raise_for_status()
        return response
    
    def _post(self, endpoint: str, **kwargs) -> httpx.Response:
        """Make POST request to Mayan API"""
        url = f"{self.base_url}/api/v4{endpoint}"
        response = self._client.post(
            url, 
            headers=self._headers, 
            auth=self.auth,
            **kwargs
        )
        response.raise_for_status()
        return response
    
    def get_api_token(self) -> str:
        """Obtain API token using username/password"""
        if not self.username or not self.password:
            raise ValueError("Username and password required for token generation")
        
        url = f"{self.base_url}/api/v4/auth/token/obtain/?format=json"
        response = self._client.post(
            url, 
            data={"username": self.username, "password": self.password}
        )
        response.raise_for_status()
        token = response.json().get("token")
        
        # Update headers with new token
        self.api_token = token
        self._headers["Authorization"] = f"Token {token}"
        logger.info("Successfully obtained API token")
        return token
    
    def list_documents(
        self, 
        page: int = 1, 
        page_size: int = 50,
        document_type_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        List documents in Mayan
        
        Returns paginated list with 'count', 'next', 'previous', 'results'
        """
        params = {"page": page, "page_size": page_size}
        if document_type_id:
            params["document_type_id"] = document_type_id
        
        response = self._get("/documents/", params=params)
        return response.json()
    
    def get_document(self, document_id: int) -> Dict[str, Any]:
        """Get document metadata by ID"""
        response = self._get(f"/documents/{document_id}/")
        return response.json()
    
    def get_document_versions(self, document_id: int) -> List[Dict[str, Any]]:
        """Get all versions of a document"""
        response = self._get(f"/documents/{document_id}/versions/")
        return response.json().get("results", [])
    
    def get_active_version(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get the active (current) version of a document"""
        doc = self.get_document(document_id)
        return doc.get("version_active")
    
    def get_version_pages(self, document_id: int, version_id: int) -> List[Dict[str, Any]]:
        """Get pages of a document version"""
        response = self._get(f"/documents/{document_id}/versions/{version_id}/pages/")
        return response.json().get("results", [])
    
    def get_page_ocr(self, document_id: int, version_id: int, page_id: int) -> str:
        """Get OCR text for a specific page"""
        response = self._get(
            f"/documents/{document_id}/versions/{version_id}/pages/{page_id}/ocr/"
        )
        data = response.json()
        # OCR content can be in 'content' or 'results' depending on version
        if isinstance(data, dict):
            return data.get("content", data.get("text", ""))
        return str(data)
    
    def get_document_ocr_content(self, document_id: int) -> str:
        """
        Get full OCR content for a document (all pages combined)
        """
        doc = self.get_document(document_id)
        active_version = doc.get("version_active")
        
        if not active_version:
            logger.warning(f"Document {document_id} has no active version")
            return ""
        
        version_id = active_version.get("id")
        pages = self.get_version_pages(document_id, version_id)
        
        ocr_texts = []
        for page in pages:
            page_id = page.get("id")
            try:
                ocr_text = self.get_page_ocr(document_id, version_id, page_id)
                if ocr_text:
                    ocr_texts.append(ocr_text)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    logger.debug(f"No OCR for page {page_id}")
                else:
                    raise
        
        return "\n\n".join(ocr_texts)
    
    def download_document_file(
        self, 
        document_id: int, 
        output_path: Optional[Path] = None
    ) -> Path:
        """
        Download document file to local path
        
        If output_path is None, saves to temp directory
        """
        doc = self.get_document(document_id)
        active_version = doc.get("version_active")
        
        if not active_version:
            raise ValueError(f"Document {document_id} has no active version")
        
        # Get file download URL
        files_url = doc.get("file_list_url")
        if not files_url:
            raise ValueError(f"Document {document_id} has no files")
        
        # Get file info
        files_response = self._get(f"/documents/{document_id}/files/")
        files = files_response.json().get("results", [])
        
        if not files:
            raise ValueError(f"Document {document_id} has no files")
        
        # Get latest file
        latest_file = files[-1]
        file_id = latest_file.get("id")
        filename = latest_file.get("filename", f"document_{document_id}")
        
        # Download file content
        download_response = self._get(
            f"/documents/{document_id}/files/{file_id}/download/"
        )
        
        # Determine output path
        if output_path is None:
            temp_dir = Path(tempfile.mkdtemp())
            output_path = temp_dir / filename
        
        output_path.write_bytes(download_response.content)
        logger.info(f"Downloaded document {document_id} to {output_path}")
        return output_path
    
    def get_document_metadata(self, document_id: int) -> Dict[str, Any]:
        """Get custom metadata attached to document"""
        try:
            response = self._get(f"/documents/{document_id}/metadata/")
            metadata_list = response.json().get("results", [])
            
            # Convert to dict
            metadata = {}
            for item in metadata_list:
                key = item.get("metadata_type", {}).get("name", item.get("metadata_type_id"))
                value = item.get("value")
                if key and value:
                    metadata[key] = value
            
            return metadata
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {}
            raise
    
    def get_document_tags(self, document_id: int) -> List[str]:
        """Get tags attached to document"""
        try:
            response = self._get(f"/documents/{document_id}/tags/")
            tags = response.json().get("results", [])
            return [tag.get("label", "") for tag in tags if tag.get("label")]
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return []
            raise
    
    def download_document_bytes(self, document_id: int) -> tuple:
        """
        Download document file as bytes.
        
        Returns:
            (file_bytes, filename) tuple
        """
        # Get file listing
        files_response = self._get(f"/documents/{document_id}/files/")
        files = files_response.json().get("results", [])
        
        if not files:
            raise ValueError(f"Document {document_id} has no files")
        
        # Get latest file
        latest_file = files[-1]
        file_id = latest_file.get("id")
        filename = latest_file.get("filename", f"document_{document_id}")
        
        # Download file content
        download_response = self._get(
            f"/documents/{document_id}/files/{file_id}/download/"
        )
        
        logger.info(f"Downloaded {filename} ({len(download_response.content)} bytes)")
        return download_response.content, filename

    def fetch_document_for_indexing(self, document_id: int) -> MayanDocument:
        """
        Fetch all document data needed for RAG indexing
        
        Returns MayanDocument with:
        - Basic metadata (label, type, language)
        - For images: OCR content from Mayan
        - For PDFs/DOCX/TXT: raw file bytes (processed by Haystack)
        - Custom metadata
        - Tags
        """
        logger.info(f"Fetching document {document_id} from Mayan EDMS")
        
        # Get basic document info
        doc_data = self.get_document(document_id)
        
        # Get document type label
        doc_type = doc_data.get("document_type", {})
        doc_type_label = doc_type.get("label", "Unknown") if isinstance(doc_type, dict) else str(doc_type)
        
        # Get custom metadata
        custom_metadata = self.get_document_metadata(document_id)
        
        # Get tags
        tags = self.get_document_tags(document_id)
        if tags:
            custom_metadata["tags"] = tags
        
        label = doc_data.get("label", f"Document {document_id}")
        file_ext = Path(label).suffix.lower()
        is_image = file_ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
        
        ocr_content = None
        file_content = None
        filename = None
        
        if is_image:
            # Images: use Mayan's OCR content
            ocr_content = self.get_document_ocr_content(document_id)
            logger.info(f"Image file — using OCR content ({len(ocr_content)} chars)")
        else:
            # PDFs, DOCX, TXT: download the actual file for Haystack processing
            try:
                file_content, filename = self.download_document_bytes(document_id)
                logger.info(f"Document file — downloaded for Haystack processing ({len(file_content)} bytes)")
            except Exception as e:
                # Fallback to OCR if file download fails
                logger.warning(f"File download failed, falling back to OCR: {e}")
                ocr_content = self.get_document_ocr_content(document_id)
        
        mayan_doc = MayanDocument(
            id=document_id,
            label=label,
            description=doc_data.get("description", ""),
            document_type=doc_type_label,
            language=doc_data.get("language", "eng"),
            datetime_created=doc_data.get("datetime_created", ""),
            file_content=file_content,
            ocr_content=ocr_content,
            metadata=custom_metadata,
            filename=filename or label,
        )
        
        content_info = f"OCR: {len(ocr_content)} chars" if ocr_content else f"file: {len(file_content)} bytes"
        logger.info(
            f"Fetched document: {mayan_doc.label} "
            f"(type: {mayan_doc.document_type}, {content_info})"
        )
        
        return mayan_doc
    
    def close(self):
        """Close HTTP client"""
        self._client.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
