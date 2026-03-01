from typing import List

import pydantic
from agents import RunContextWrapper, function_tool

from adapter_agent.library.rust_doc_analyzer import RustDocAnalyzer, SearchResult


class WithRustDocAnalyzer(pydantic.BaseModel):
    """
    Context mixin that requires a 'rust_doc_analyzer' attribute.
    Since RustDocAnalyzer is complex and not a Pydantic model, we allow arbitrary types.
    """

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)
    rust_doc_analyzer: RustDocAnalyzer


@function_tool
def search_docs(
    wrapper: RunContextWrapper[WithRustDocAnalyzer], query: str, limit: int = 10
) -> List[SearchResult]:
    """
    Search the Rust documentation (docstrings) for a specific query.

    This function searches the text content of the documentation for all items in the crate.
    It is useful for finding functionality based on descriptions, concepts, or keywords
    that might appear in the explanation of the code, rather than just the names of types or functions.

    Args:
        wrapper: The execution context containing the RustDocAnalyzer instance.
        query: The string to search for within the documentation text. Case-insensitive.
        limit: The maximum number of results to return. Defaults to 10.

    Returns:
        List[SearchResult]: A list of matching items, including their name, kind, relevance score, and a text snippet.

    Example Usage:
        # To find functions or structs related to matrix multiplication described in text
        results = search_docs(context, "matrix multiplication")

        # To find where "eigenvectors" are mentioned
        results = search_docs(context, "eigenvectors")

    Example Output:
        [
            SearchResult(
                name="numrs2::matrix::Matrix::dot",
                kind="fn",
                score=3,
                snippet="...computes the matrix multiplication of two matrices..."
            ),
            ...
        ]
    """
    return wrapper.context.rust_doc_analyzer.search_docs(query, limit)


@function_tool
def search_symbol(
    wrapper: RunContextWrapper[WithRustDocAnalyzer], query: str, limit: int = 10
) -> List[SearchResult]:
    """
    Search for Rust symbols (Structs, Enums, Functions, Modules, etc.) by their name.

    This function strictly searches the *name* of the item, ignoring the full path.
    It uses a scoring system where exact matches are ranked highest, followed by prefix matches.
    Use this tool when you know the name (or part of the name) of the type or function you are looking for.

    Args:
        wrapper: The execution context containing the RustDocAnalyzer instance.
        query: The name (or partial name) of the symbol to search for. Case-insensitive.
        limit: The maximum number of results to return. Defaults to 10.

    Returns:
        List[SearchResult]: A list of symbols matching the name query.

    Example Usage:
        # To find the 'Matrix' struct
        results = search_symbol(context, "Matrix")

        # To find functions starting with 'new'
        results = search_symbol(context, "new_")

    Example Output:
        [
            SearchResult(
                name="numrs2::matrix::Matrix",
                kind="struct",
                score=100,
                snippet="A generic N-dimensional matrix."
            ),
            SearchResult(
                name="numrs2::matrix::MatrixDecomposition",
                kind="trait",
                score=50,
                snippet="..."
            )
        ]
    """
    return wrapper.context.rust_doc_analyzer.search_symbol(query, limit)


@function_tool
def search(
    wrapper: RunContextWrapper[WithRustDocAnalyzer], query: str, limit: int = 10
) -> List[SearchResult]:
    """
    Search the Rust documentation for both symbols and concepts.

    This function searches BOTH the name of the item and its explanation text (docstrings).
    It is the recommended tool to find functionality based on descriptions, concepts,
    keywords, or the names of types/functions. Matches in the item name are prioritized
    over matches in the description.

    Args:
        wrapper: The execution context containing the RustDocAnalyzer instance.
        query: The string to search for within the documentation text or symbol name. Case-insensitive.
        limit: The maximum number of results to return. Defaults to 10.

    Returns:
        List[SearchResult]: A list of matching items, including their name, kind, relevance score, and a text snippet.

    Example Usage:
        # To find functions or structs related to sorting either by name or description
        results = search(context, "sort")
    """
    return wrapper.context.rust_doc_analyzer.search(query, limit)


@function_tool
def get_crate_overview(wrapper: RunContextWrapper[WithRustDocAnalyzer]) -> str:
    """
    Get a comprehensive textual overview of the crate configuration and structure.

    This returns a markdown-formatted string that includes:
    1.  **Crate Description**: The high-level documentation of the root crate.
    2.  **Module Hierarchy**: A tree view of the crate's modules and significant public items (structs, functions, etc.).

    This is extremely useful as a first step to understand the layout of the library,
    what top-level modules are available, and where to look for specific functionality.

    Args:
        wrapper: The execution context containing the RustDocAnalyzer instance.

    Returns:
        str: A markdown string containing the crate description and module tree.

    Example Usage:
        overview = get_crate_overview(context)
        print(overview)
    """
    return wrapper.context.rust_doc_analyzer.get_overview()
