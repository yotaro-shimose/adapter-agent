from typing import Self
import logging
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ItemInner(BaseModel):
    """Represents the 'inner' dictionary of a Rustdoc item."""

    module: Optional[Dict[str, Any]] = None
    struct: Optional[Dict[str, Any]] = None
    function: Optional[Dict[str, Any]] = None
    enum: Optional[Dict[str, Any]] = None
    trait: Optional[Dict[str, Any]] = None
    impl: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class Item(BaseModel):
    """Represents a single item in the Rustdoc index."""

    id: Union[str, int]
    crate_id: int
    name: Optional[str] = None
    docs: Optional[str] = None
    inner: Dict[str, Any] = Field(default_factory=dict)
    visibility: Union[str, Dict[str, Any]] = "default"


class PathEntry(BaseModel):
    """Entry in the 'paths' section."""

    crate_id: int
    path: List[str]
    kind: str


class CrateData(BaseModel):
    """Root model for the rustdoc JSON."""

    root: str
    crate_version: Optional[str] = None
    format_version: Optional[int] = None
    includes_private: bool = False
    index: Dict[str, Item]
    paths: Dict[str, PathEntry] = Field(default_factory=dict)
    external_crates: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    name: str
    kind: str
    score: int
    snippet: str
    signature: Optional[str] = None


class RustDocAnalyzer:
    """
    Analyzer for Rustdoc JSON output.
    Uses Pydantic for data validation and pathlib for paths.
    """

    def __init__(
        self,
        data: CrateData,
        parent_map: Dict[str, str],
        impl_to_type_map: Dict[str, str],
        pubapi_path: Optional[Path] = None,
    ):
        self.data = data
        self.parent_map = parent_map
        self.impl_to_type_map = impl_to_type_map
        self._memo_paths: Dict[str, str] = {}
        self.pubapi_map: Dict[str, List[str]] = defaultdict(list)
        if pubapi_path:
            self._load_pubapi(pubapi_path)

    @classmethod
    def from_json(cls, path: Path, pubapi_path: Optional[Path] = None) -> Self:
        """
        Factory method to load and parse the JSON file.
        Performs all IO and heavy processing before returning an instance.
        """
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found at {path}")

        logger.info(f"Loading JSON from {path}...")
        raw_data = json.loads(path.read_text(encoding="utf-8"))

        if "root" in raw_data:
            raw_data["root"] = str(raw_data["root"])

        logger.info("Parsing data into Pydantic models...")
        crate_data = CrateData(**raw_data)

        logger.info("Building index maps...")
        parent_map: Dict[str, str] = {}
        impl_to_type_map: Dict[str, str] = {}

        for i_id, item in crate_data.index.items():
            inner = item.inner

            if "module" in inner:
                for child_id in inner["module"].get("items", []):
                    parent_map[str(child_id)] = i_id
            elif "impl" in inner:
                impl_body = inner["impl"]
                for child_id in impl_body.get("items", []):
                    parent_map[str(child_id)] = i_id

                for_obj = impl_body.get("for", {})
                resolved_path = for_obj.get("resolved_path", {})
                type_id = resolved_path.get("id")
                if type_id:
                    impl_to_type_map[i_id] = str(type_id)
            elif "enum" in inner:
                for variant_id in inner["enum"].get("variants", []):
                    parent_map[str(variant_id)] = i_id

        return cls(crate_data, parent_map, impl_to_type_map, pubapi_path)

    def _load_pubapi(self, path: Path) -> None:
        """Load and parse pubapi.txt."""
        if not path.exists():
            logger.warning(f"pubapi file not found at {path}")
            return

        logger.info(f"Loading pubapi from {path}...")
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("pub "):
                        continue

                    # Extract definition for normalization
                    # Heuristic: split by space, find first part with '::' that might be the item
                    # e.g. "pub fn numrs2::array::Array<f32>::simd_fma(...)"

                    parts = line.split(" ")
                    name_part = None
                    for p in parts:
                        if "::" in p:
                            # Taking the part before '(' if it's a function
                            if "(" in p:
                                name_part = p.split("(")[0]
                            else:
                                name_part = p
                            break

                    if name_part:
                        normalized = self._normalize_name(name_part)
                        self.pubapi_map[normalized].append(line)
        except Exception as e:
            logger.error(f"Failed to load pubapi: {e}")

    def _normalize_name(self, name: str) -> str:
        """
        Normalize a name by removing generic parameters.
        numrs2::array::Array<f32>::simd_fma -> numrs2::array::Array::simd_fma
        """
        prev = name
        while True:
            # Non-greedy match for <...>
            new = re.sub(r"<[^<>]*>", "", prev)
            if new == prev:
                break
            prev = new
        return prev

    def resolve_full_name(self, item_id: str, depth: int = 0) -> str:
        """Resolve the fully qualified name of an item recursively."""
        if depth > 10:
            return "..."

        item_id = str(item_id)

        if item_id in self._memo_paths:
            return self._memo_paths[item_id]

        if item_id in self.data.paths:
            p_entry = self.data.paths[item_id]
            full = "::".join(p_entry.path)
            self._memo_paths[item_id] = full
            return full

        item = self.data.index.get(item_id)
        if not item:
            return f"<Unknown {item_id}>"

        my_name = item.name

        if item_id in self.impl_to_type_map:
            type_id = self.impl_to_type_map[item_id]
            return self.resolve_full_name(type_id, depth + 1)

        parent_id = self.parent_map.get(item_id)
        if parent_id:
            if parent_id in self.impl_to_type_map:
                type_id = self.impl_to_type_map[parent_id]
                type_name = self.resolve_full_name(type_id, depth + 1)
                full = f"{type_name}::{my_name}" if my_name else type_name
                self._memo_paths[item_id] = full
                return full
            else:
                parent_name = self.resolve_full_name(parent_id, depth + 1)
                full = f"{parent_name}::{my_name}" if my_name else parent_name
                self._memo_paths[item_id] = full
                return full

        if my_name:
            self._memo_paths[item_id] = my_name
            return my_name

        return f"<Anonymous {item_id}>"

    def _render_type(self, type_obj: Dict[str, Any]) -> str:
        """Render a type dictionary into a string."""
        if not type_obj:
            return "_"

        if "primitive" in type_obj:
            return type_obj["primitive"]

        if "resolved_path" in type_obj:
            path_obj = type_obj["resolved_path"]
            name = path_obj.get("name", "")
            if not name and "path" in path_obj:
                name = path_obj[
                    "path"
                ]  # Older versions might use 'path' string? Newer seems to have name/id?
                # The logged example showed: "resolved_path": { "path": "Array", ... }

            args = path_obj.get("args")
            if args:
                if "angle_bracketed" in args:
                    ab = args["angle_bracketed"]
                    arg_list = []
                    for arg in ab.get("args", []):
                        if "type" in arg:
                            arg_list.append(self._render_type(arg["type"]))
                        elif "const" in arg:
                            arg_list.append(
                                f"const {arg.get('const', {}).get('expr', '_')}"
                            )

                    if arg_list:
                        return f"{name}<{', '.join(arg_list)}>"
            return name

        if "borrowed_ref" in type_obj:
            ref = type_obj["borrowed_ref"]
            mutable = "mut " if ref.get("is_mutable") else ""
            inner = self._render_type(ref.get("type", {}))
            return f"&{mutable}{inner}"

        if "generic" in type_obj:
            return type_obj["generic"]

        if "tuple" in type_obj:
            types = [self._render_type(t) for t in type_obj["tuple"]]
            return f"({', '.join(types)})"

        if "slice" in type_obj:
            return f"[{self._render_type(type_obj['slice'])}]"

        return "_"

    def _render_signature(self, item: Item) -> Optional[str]:
        """Render function signature if available."""
        # Try pubapi first
        if self.pubapi_map:
            full_name = self.resolve_full_name(str(item.id))
            if full_name:
                matches = self.pubapi_map.get(full_name)
                if matches:
                    return "\n".join(matches)

        if not item.inner or "function" not in item.inner:
            return None

        fn_obj = item.inner["function"]
        sig = fn_obj.get("sig", {})

        inputs = []
        for arg in sig.get("inputs", []):
            # arg is [name, type]
            if len(arg) == 2:
                name, type_val = arg
                type_str = self._render_type(type_val)
                inputs.append(f"{name}: {type_str}")

        output_str = ""
        output = sig.get("output")
        if output:
            output_str = f" -> {self._render_type(output)}"

        fn_name = item.name or "_"
        return f"fn {fn_name}({', '.join(inputs)}){output_str}"

    def search_docs(self, query: str, limit: int = 10) -> List[SearchResult]:
        """Search documentation (docstrings) for the given query."""
        query_lower = query.lower()
        matches = []

        for item_id, item in self.data.index.items():
            if not item.docs:
                continue

            docs_lower = item.docs.lower()
            if query_lower in docs_lower:
                fullname = self.resolve_full_name(item_id)
                kind = "unknown"
                if item.inner:
                    kind = next(iter(item.inner))

                score = docs_lower.count(query_lower)

                idx = docs_lower.find(query_lower)
                start = max(0, idx - 40)
                end = min(len(item.docs), idx + 100)
                snippet = item.docs[start:end].replace("\n", " ")
                if start > 0:
                    snippet = "..." + snippet
                if end < len(item.docs):
                    snippet = snippet + "..."

                signature = self._render_signature(item)

                matches.append(
                    SearchResult(
                        name=fullname,
                        kind=kind,
                        score=score,
                        snippet=snippet,
                        signature=signature,
                    )
                )

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:limit]

    def search_symbol(self, query: str, limit: int = 10) -> List[SearchResult]:
        """
        Search for symbols by name.
        Only checks the symbol's own name, ignoring the full path.
        """
        query_lower = query.lower()
        matches = []

        for item_id, item in self.data.index.items():
            if not item.name:
                continue

            name_lower = item.name.lower()
            if query_lower not in name_lower:
                continue

            fullname = self.resolve_full_name(item_id)
            kind = "unknown"
            if item.inner:
                kind = next(iter(item.inner))

            if name_lower == query_lower:
                score = 100
            elif name_lower.startswith(query_lower):
                score = 50
            else:
                score = 10

            snippet = ""
            if item.docs:
                snippet = item.docs.strip().split("\n")[0][:100]
                if len(snippet) < len(item.docs):
                    snippet += "..."

            signature = self._render_signature(item)

            matches.append(
                SearchResult(
                    name=fullname,
                    kind=kind,
                    score=score,
                    snippet=snippet,
                    signature=signature,
                )
            )

        matches.sort(key=lambda x: x.score, reverse=True)
        return matches[:limit]

    def get_overview(self) -> str:
        """Generate a markdown overview of the crate."""
        lines = []
        root_item = self.data.index.get(self.data.root)
        if not root_item:
            return "Root item not found"

        name = root_item.name or "Unknown"
        version = self.data.crate_version or "N/A"
        lines.append(f"# Crate: {name} ({version})")

        if root_item.docs:
            lines.append("\n## Description")
            summary_lines = root_item.docs.strip().split("\n")
            count = 0
            for line in summary_lines:
                lines.append(f"{line}")
                count += 1

        lines.append("\n## Module Hierarchy")

        def get_public_children_ids(item: Item) -> List[str]:
            if "module" in item.inner:
                return [str(i) for i in item.inner["module"].get("items", [])]
            return []

        def recurse_tree(item_id: str, depth: int = 0, prefix: str = ""):
            item = self.data.index.get(item_id)
            if not item:
                return

            item_name = item.name or "<anon>"
            children_ids = get_public_children_ids(item)

            modules = []
            stats = defaultdict(int)

            for cid in children_ids:
                child = self.data.index.get(cid)
                if not child:
                    continue

                inner_keys = list(child.inner.keys())
                if "module" in inner_keys:
                    modules.append(cid)
                elif inner_keys:
                    kind = inner_keys[0]
                    if kind == "function":
                        k = "fn"
                    else:
                        k = kind
                    stats[k] += 1
                else:
                    stats["other"] += 1

            stat_parts = []
            for k in ["struct", "enum", "trait", "fn"]:
                if stats[k] > 0:
                    stat_parts.append(f"{stats[k]} {k}s")

            other_count = sum(
                v
                for k, v in stats.items()
                if k not in ["struct", "enum", "trait", "fn"]
            )
            if other_count:
                stat_parts.append(f"{other_count} other")

            stat_str = f" ({', '.join(stat_parts)})" if stat_parts else ""
            lines.append(f"{prefix}📦 {item_name}{stat_str}")

            for i, mid in enumerate(modules):
                is_last = i == len(modules) - 1
                connector = "└── " if is_last else "├── "
                recurse_tree(
                    mid, depth + 1, prefix + ("    " if is_last else "│   ") + connector
                )

        def print_node(
            node_id: str, prefix: str = "", is_last: bool = True, is_root: bool = False
        ):
            item = self.data.index.get(node_id)
            if not item:
                return

            node_name = item.name or "<anon>"
            children_ids = get_public_children_ids(item)

            modules = []
            stats = defaultdict(int)
            for cid in children_ids:
                child = self.data.index.get(cid)
                if not child:
                    continue

                keys = list(child.inner.keys())
                if "module" in keys:
                    modules.append(cid)
                elif keys:
                    k = keys[0]
                    if k == "function":
                        k = "fn"
                    stats[k] += 1
                else:
                    stats["unknown"] += 1

            stat_parts = []
            for k in ["struct", "enum", "trait", "fn"]:
                if stats[k]:
                    stat_parts.append(f"{stats[k]} {k}")
            other = sum(
                v
                for k, v in stats.items()
                if k not in ["struct", "enum", "trait", "fn"]
            )
            if other:
                stat_parts.append(f"{other} other")

            stat_str = f" ({', '.join(stat_parts)})" if stat_parts else ""

            if is_root:
                lines.append(f"📦 {node_name}{stat_str}")
                my_prefix = ""
            else:
                connector = "└── " if is_last else "├── "
                lines.append(f"{prefix}{connector}📦 {node_name}{stat_str}")
                my_prefix = prefix + ("    " if is_last else "│   ")

            for i, mid in enumerate(modules):
                print_node(mid, my_prefix, i == len(modules) - 1, False)

        print_node(self.data.root, is_root=True)
        return "\n".join(lines)

    @classmethod
    def from_libdir(cls, host_lib_dir: Path) -> Self:
        doc_path = host_lib_dir / "target" / "doc"
        pubapi_path = host_lib_dir / "pubapi.txt"
        json_path = None
        if doc_path.exists():
            if (doc_path / "numrs2.json").exists():
                json_path = doc_path / "numrs2.json"

        if json_path and json_path.exists():
            return RustDocAnalyzer.from_json(json_path, pubapi_path=pubapi_path)
        else:
            # Fallback or error
            # For debug script, we might want to try finding it or just fail
            raise FileNotFoundError(
                f"Could not find rustdoc json in {doc_path}. Ensure you are in the workspace root or path is correct."
            )
