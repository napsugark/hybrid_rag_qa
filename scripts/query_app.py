

#!/usr/bin/env python3
"""
Interactive query interface for Hybrid RAG application
"""

import sys
import logging
import time
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.app import HybridRAGApplication
from src import config

console = Console()
logger = logging.getLogger("HybridRAG")


def display_welcome():
    welcome_text = """
# Advanced Hybrid RAG Query Interface

Ask questions about your documents using hybrid search!

**Features:**
- Sparse + Dense retrieval
- Metadata-aware search
- Summary-enhanced context
- Local LLM (no API costs!)
    """
    console.print(Panel(Markdown(welcome_text), border_style="blue"))


def display_documents(documents):
    """Display retrieved documents"""
    if not documents:
        console.print("[yellow]No documents retrieved[/yellow]")
        return

    for idx, doc in enumerate(documents, 1):
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Field", style="bold cyan", width=15)
        table.add_column("Value", style="white")

        source = Path(doc.meta.get("file_path", "Unknown")).name
        table.add_row("Source", source)
        score = getattr(doc, "score", None)
        if score is not None:
            table.add_row("Relevance", f"{score:.3f}")
        if config.ENABLE_METADATA_EXTRACTION:
            if topics := doc.meta.get("topics"):
                table.add_row("Topics", ", ".join(topics[:3]))
            if keywords := doc.meta.get("keywords"):
                table.add_row("Keywords", ", ".join(keywords[:5]))
        if doc.meta.get("summary"):
            summary = doc.meta["summary"]
            if len(summary) > 200:
                summary = summary[:200] + "..."
            table.add_row("Summary", summary)

        content = doc.content.strip()
        if len(content) > config.DISPLAY_CONFIG["max_content_preview"]:
            content = content[: config.DISPLAY_CONFIG["max_content_preview"]] + "..."
        console.print(
            Panel(table, title=f"[bold]Document {idx}[/bold]", border_style="green")
        )
        console.print(f"[dim]{content}[/dim]\n")


def interactive_query(app: HybridRAGApplication):
    console.print(
        "\n[bold green]Enter your questions (type 'quit' or 'exit' to stop)[/bold green]\n"
    )
    query_count = 0

    while True:
        try:
            query = console.input("[bold cyan]❓ Question:[/bold cyan] ").strip()
            if not query:
                continue
            if query.lower() in ["quit", "exit", "q"]:
                console.print("\n[yellow]Goodbye! 👋[/yellow]")
                logger.info(f"Query session ended. Total queries: {query_count}")
                break

            if query.lower() == "stats":
                stats = app.get_statistics()
                console.print(f"\n[bold]Document Statistics:[/bold]")
                console.print(f"  Total chunks: {stats['total_documents']}")
                console.print(f"  With summaries: {stats['with_summaries']}")
                console.print(f"  With metadata: {stats['with_metadata']}")
                console.print(f"  Unique sources: {len(stats['sources'])}")
                logger.info("Statistics command executed")
                continue

            query_count += 1
            logger.info("=" * 80)
            logger.info(f"QUERY #{query_count}: {query}")
            logger.info("=" * 80)

            console.print("\n[dim]Searching with hybrid retrieval...[/dim]")
            query_start_time = time.time()
            result = app.query(query)
            query_elapsed = time.time() - query_start_time

            # Extract documents - app.query() returns them under "retriever" key
            documents = result.get("retriever", {}).get("documents", [])

            display_documents(documents)

            # Display answer
            replies = result.get("generator", {}).get("replies", [])
            if replies:
                answer = replies[0].text
                console.print(
                    Panel(
                        Markdown(f"**Answer:**\n\n{answer}"),
                        title="[bold green]AI Response[/bold green]",
                        border_style="green",
                    )
                )
                logger.info(f"ANSWER #{query_count}: {answer}")
                console.print(f"[dim]Response generated in {query_elapsed:.2f}s[/dim]")

        except KeyboardInterrupt:
            console.print("\n\n[yellow]Interrupted. Goodbye! 👋[/yellow]")
            logger.info(f"Query session interrupted. Total queries: {query_count}")
            break
        except Exception as e:
            console.print(f"\n[red]Error: {e}[/red]\n")
            logger.error(f"Error processing query: {e}", exc_info=True)


def main():
    display_welcome()
    try:
        console.print("[dim]Initializing Hybrid RAG application...[/dim]")
        start_time = time.time()
        app = HybridRAGApplication()
        init_time = time.time() - start_time
        console.print(f"[dim]Initialized in {init_time:.2f}s[/dim]")

        doc_count = app.get_document_count()
        if doc_count == 0:
            console.print(
                "\n[yellow]WARNING: No documents found in the database![/yellow]\n"
                "Please run the indexing script first.\n"
            )
            logger.warning("No documents found in database")
            return

        console.print(
            f"\n[green]Ready! {doc_count} document chunks available for search.[/green]"
        )
        interactive_query(app)

        if app.langfuse:
            try:
                app.langfuse.flush()
                logger.info("Langfuse data flushed")
            except Exception as e:
                logger.debug(f"Langfuse flush failed: {e}")

    except Exception as e:
        console.print(f"\n[red]Error initializing application: {e}[/red]")
        logger.error(f"Failed to initialize application: {e}", exc_info=True)
    finally:
        if "app" in locals() and app.langfuse:
            try:
                app.langfuse.shutdown()
                logger.info("Langfuse shutdown complete")
            except Exception as e:
                logger.debug(f"Langfuse shutdown failed: {e}")


if __name__ == "__main__":
    main()
