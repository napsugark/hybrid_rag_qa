"""
Custom Haystack component for document summarization using local LLM
"""

from typing import List, Dict, Any
import logging

from haystack import component, Document
from .. import config
from haystack_integrations.components.generators.ollama import OllamaGenerator
from haystack.components.generators.chat import AzureOpenAIChatGenerator
from haystack.utils import Secret
from haystack.dataclasses import ChatMessage


logger = logging.getLogger(__name__)
@component
class DocumentSummarizer:
    """
    Generates concise summaries for documents using a local LLM.
    Adds summaries to document metadata for improved retrieval context.
    """

    def __init__(
        self,
        llm_type: str = None,
        llm_model: str = None,
        llm_url: str = None,
        llm_api_key: str = None,
        llm_api_version: str = None,
        max_summary_length: int = 150,
        summary_style: str = "concise",
        timeout: int = 600,
    ):
        """
        Initialize document summarizer
        
        Args:
            ollama_url: URL of Ollama server
            ollama_model: Ollama model to use
            max_summary_length: Maximum words in summary
            summary_style: Style of summary ('concise', 'detailed', 'bullet_points')
            timeout: Request timeout in seconds
        """
        self.llm_type = llm_type or getattr(config, "LLM_TYPE", "OLLAMA")
        self.llm_model = llm_model or getattr(config, "LLM_MODEL", "llama3.1:8b")
        self.llm_url = llm_url or getattr(config, "LLM_URL", "http://127.0.0.1:11435")
        self.llm_api_key = llm_api_key or getattr(config, "LLM_API_KEY", None)
        self.llm_api_version = llm_api_version or getattr(config, "LLM_API_VERSION", None)
        self.max_summary_length = max_summary_length
        self.summary_style = summary_style
        self.timeout = timeout
        
        # Style-specific prompt adjustments
        style_instructions = {
            "concise": "Create a brief, single-paragraph summary focusing on the main point.",
            "detailed": "Create a comprehensive summary covering all key points in detail.",
            "bullet_points": "Create a bullet-point summary of the main ideas (3-5 points)."
        }

        self.style_instruction = style_instructions.get(
            summary_style,
            style_instructions["concise"]
        )

        # Load prompt template from external file, fall back to inline default
        self.prompt_template = self._load_prompt_template()
        logger.info(f"Summarization prompt loaded ({len(self.prompt_template)} chars)")

        # Initialize the correct generator based on config.LLM_TYPE
        if self.llm_type == "OLLAMA":
            if OllamaGenerator is None:
                raise ImportError("OllamaGenerator not available")
            self.generator = OllamaGenerator(
                model=self.llm_model,
                url=self.llm_url,
                timeout=self.timeout,
                generation_kwargs={
                    "temperature": 0.5,
                    "num_predict": 300,
                }
            )
            logger.info(f"DocumentSummarizer initialized with Ollama model {self.llm_model}")
        elif self.llm_type == "AZURE_OPENAI":
            if AzureOpenAIChatGenerator is None:
                raise ImportError("AzureOpenAIChatGenerator not available")
            self.generator = AzureOpenAIChatGenerator(
                azure_deployment=self.llm_model,
                azure_endpoint=self.llm_url,
                api_key=Secret.from_token(self.llm_api_key),
                api_version=self.llm_api_version,
                timeout=self.timeout,
                generation_kwargs={
                    "temperature": 0.5,
                    "max_tokens": 300,
                },
            )
            logger.info(f"DocumentSummarizer initialized with Azure OpenAI deployment {self.llm_model}")
        else:
            raise ValueError(f"Unsupported LLM_TYPE: {self.llm_type}")

    @component.output_types(documents=List[Document])
    def run(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Generate summaries for documents
        
        Args:
            documents: List of documents to summarize
            
        Returns:
            Dictionary with documents containing summaries in metadata
        """
        if not documents:
            return {"documents": []}
        
        summarized_docs = []
        logger.info(f"Generating summaries for {len(documents)} documents")
        
        for idx, doc in enumerate(documents):
            try:
                # Skip very short documents
                if not doc.content or len(doc.content.strip()) < 100:
                    logger.debug(f"Document {idx} too short for summarization")
                    doc.meta["summary"] = doc.content[:100] if doc.content else ""
                    summarized_docs.append(doc)
                    continue
                
                # Limit content length for summarization
                content_to_summarize = doc.content[:3000]
                
                # Prepare prompt
                prompt = self.prompt_template.format(
                    max_length=self.max_summary_length,
                    style_instruction=self.style_instruction,
                    content=content_to_summarize
                )
                
                # Generate summary
                result = self.generator.run(messages=[ChatMessage.from_user(prompt)])
                if result and "replies" in result and result["replies"]:
                    chat_msg = result["replies"][0]         
                    # ChatMessage object
                    summary = chat_msg.text.strip()  
                    # Clean up summary
                    summary = self._clean_summary(summary)
                    
                    # Add to metadata
                    doc.meta["summary"] = summary
                    logger.debug(f"Generated summary for doc {idx} ({len(summary)} chars)")
                else:
                    logger.error(f"No response from LLM for document {idx}. Result: {result}")
                    doc.meta["summary"] = content_to_summarize[:200] + "..."
                
                summarized_docs.append(doc)
                
            except Exception as e:
                error_type = type(e).__name__
                error_msg = str(e)
                
                # Handle Azure content filter gracefully
                if "content_filter" in error_msg or "content management policy" in error_msg:
                    logger.warning(
                        f"Document {idx}: Azure content filter triggered — "
                        f"using fallback summary (content_filter_result in response)"
                    )
                    doc.meta["summary"] = doc.content[:200] if doc.content else ""
                    doc.meta["summary_error"] = "content_filter"
                else:
                    logger.error(f"Error summarizing document {idx}: {error_type}: {error_msg}")
                    logger.exception(f"Full traceback for document {idx}:")
                    doc.meta["summary"] = doc.content[:200] if doc.content else ""
                
                summarized_docs.append(doc)
        
        logger.info(f"Generated summaries for {len(summarized_docs)} documents")
        return {"documents": summarized_docs}
    
    def _load_prompt_template(self) -> str:
        """Load the summarization prompt from the external file, with inline fallback."""
        prompt_file = getattr(config, "SUMMARIZATION_PROMPT_FILE", None)
        if prompt_file and prompt_file.exists():
            logger.info(f"Loading summarization prompt from {prompt_file}")
            return prompt_file.read_text(encoding="utf-8")

        logger.warning("External summarization prompt not found — using inline fallback")
        return (
            "You are a document summarizer. Read the following text and write a "
            "{max_length}-word summary.\n"
            "{style_instruction}\n"
            "Preserve important information such as names, dates, and amounts.\n\n"
            "Text:\n{content}\n\nSummary:\n"
        )

    def _clean_summary(self, summary: str) -> str:
        """
        Clean and format summary text
        
        Args:
            summary: Raw summary from LLM
            
        Returns:
            Cleaned summary
        """
        # Remove common prefixes (case-insensitive)
        prefixes_to_remove = [
            "here is a summary of the text in 150 words or less:",
            "here is a summary of the text:",
            "here is a summary:",
            "here's a summary:",
            "summary:",
            "the summary is:",
            "here is the summary:",
            "this is a summary:",
        ]
        
        summary_lower = summary.lower()
        for prefix in prefixes_to_remove:
            if summary_lower.startswith(prefix):
                summary = summary[len(prefix):].strip()
                summary_lower = summary.lower()  # Update for potential multiple prefixes
        
        # Remove quotes if wrapped
        if summary.startswith('"') and summary.endswith('"'):
            summary = summary[1:-1]
        if summary.startswith("'") and summary.endswith("'"):
            summary = summary[1:-1]
        
        return summary.strip()
