#!/usr/bin/env python3
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator, List, Optional, Sequence

from fastmcp import FastMCP
from fastmcp.server.providers import Provider
from fastmcp.tools import Tool

from adapter_agent.library.async_rust_doc_analyzer import (
    AsyncRustDocAnalyzer,
    SearchResult,
)

class RustDocProvider(Provider):
    """
    Provides tools and resources for analyzing Rust documentation.
    Uses AsyncRustDocAnalyzer with an Elasticsearch backend.
    """

    def __init__(self, lib_dir: Path, es_host: str):
        super().__init__()
        self.lib_dir = lib_dir
        self.es_host = es_host
        self.analyzer: Optional[AsyncRustDocAnalyzer] = None

    @asynccontextmanager
    async def lifespan(self) -> AsyncIterator[None]:
        """Manage the lifecycle of the AsyncRustDocAnalyzer."""
        print(f"Initializing AsyncRustDocAnalyzer with lib_dir={self.lib_dir}, es_host={self.es_host}...")
        try:
            self.analyzer = await AsyncRustDocAnalyzer.create_from_libdir(
                host_lib_dir=self.lib_dir, host=self.es_host, skip_init=False
            )
            print("AsyncRustDocAnalyzer initialized successfully.")
            yield
        except Exception as e:
            print(f"Failed to initialize AsyncRustDocAnalyzer: {e}")
            yield
        finally:
            if self.analyzer:
                await self.analyzer.close()
                print("AsyncRustDocAnalyzer closed.")

    async def _list_tools(self) -> Sequence[Tool]:
        """Return the tools offered by this provider."""
        
        async def search(query: str, limit: int = 10) -> str:
            """
            Search for Rust symbols and documentation within the indexed crate.

            Args:
                query: Semantic or keyword search query (e.g., 'matrix multiplication', 'Vector struct').
                limit: Maximum number of results to return (default: 10).
            """
            if not self.analyzer:
                return "Error: RustDoc analyzer is not initialized."

            try:
                results = await self.analyzer.search(query, limit=limit)
                return self._format_search_results(results)
            except Exception as e:
                return f"Error during search execution: {e}"

        return [Tool.from_function(search, name="search")]

    def _format_search_results(self, results: List[SearchResult]) -> str:
        """Format the SearchResult list into a readable Markdown string for LLM consumption."""
        if not results:
            return "No results found for the given query."

        output = []
        for i, res in enumerate(results, 1):
            output.append(f"### {i}. {res.name}")
            output.append(f"- **Kind**: `{res.kind}`")
            if res.signature:
                output.append(f"- **Signature**:\n```rust\n{res.signature}\n```")
            if res.associated_methods:
                output.append(
                    f"- **Associated Methods**: {', '.join(f'`{m}`' for m in res.associated_methods)}"
                )
            if res.docs:
                docs_snippet = res.docs.strip()
                output.append(f"- **Documentation**:\n{docs_snippet}")
            output.append("\n---\n")

        return "\n".join(output)

# Configuration from environment variables
lib_dir_path = Path(os.getenv("RUSTDOC_LIB_DIR", "repositories/numrs"))
elasticsearch_url = os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")

# Initialize FastMCP with the custom provider at module level
mcp = FastMCP(
    "RustDoc",
    providers=[
        RustDocProvider(lib_dir=lib_dir_path, es_host=elasticsearch_url)
    ]
)

if __name__ == "__main__":
    # Run the server (default mode is stdio)
    mcp.run()
