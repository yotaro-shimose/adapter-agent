import hashlib
import json
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Self, Union

from elasticsearch import AsyncElasticsearch, helpers
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# --- Data Models ---
class ItemInner(BaseModel):
    module: Optional[Dict[str, Any]] = None
    struct: Optional[Dict[str, Any]] = None
    function: Optional[Dict[str, Any]] = None
    enum: Optional[Dict[str, Any]] = None
    trait: Optional[Dict[str, Any]] = None
    impl: Optional[Dict[str, Any]] = None
    struct_field: Optional[Dict[str, Any]] = None
    variant: Optional[Dict[str, Any]] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

class Item(BaseModel):
    id: Union[str, int]
    crate_id: int
    name: Optional[str] = None
    docs: Optional[str] = None
    inner: Dict[str, Any] = Field(default_factory=dict)
    visibility: Union[str, Dict[str, Any]] = "default"

class PathEntry(BaseModel):
    crate_id: int
    path: List[str]
    kind: str

class CrateData(BaseModel):
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
    docs: str
    signature: Optional[str] = None
    associated_methods: Optional[List[str]] = None


# --- Analyzer ---
class AsyncRustDocAnalyzer:
    """
    Elasticsearch-backed analyzer for Rustdoc JSON.
    Replaces local memory searches with ES index queries.
    """
    def __init__(
        self,
        es_client: AsyncElasticsearch,
        index_name: str,
        data: CrateData,
        parent_map: Dict[str, str],
        impl_to_type_map: Dict[str, str],
        pubapi_path: Optional[Path] = None,
    ):
        self.es_client = es_client
        self.index_name = index_name
        self.data = data
        self.parent_map = parent_map
        self.impl_to_type_map = impl_to_type_map
        self._memo_paths: Dict[str, str] = {}
        self.pubapi_map: Dict[str, List[str]] = defaultdict(list)
        if pubapi_path:
            self._load_pubapi(pubapi_path)

    @classmethod
    async def _init_elasticsearch(cls, es_client: AsyncElasticsearch, index_name: str, instance: Self) -> None:
        exists = await es_client.indices.exists(index=index_name)
        if exists:
            count_res = await es_client.count(index=index_name)
            if count_res["count"] > 0:
                logger.info(f"Index '{index_name}' already exists with {count_res['count']} documents. Skipping bulk insert.")
                return
            else:
                logger.info(f"Index '{index_name}' exists but is empty. Recreating...")
                await es_client.indices.delete(index=index_name)

        logger.info(f"Creating Elasticsearch index '{index_name}'...")
        await es_client.indices.create(
            index=index_name,
            body={
                "settings": {
                    "analysis": {
                        "analyzer": {
                            "default": {
                                "type": "standard"
                            }
                        }
                    }
                },
                "mappings": {
                    "properties": {
                        "id": {"type": "keyword"},
                        "name": {"type": "text", "similarity": "BM25", "fields": {"keyword": {"type": "keyword"}}},
                        "full_name": {"type": "text", "similarity": "BM25"},
                        "docs": {"type": "text", "similarity": "BM25"},
                        "kind": {"type": "keyword"},
                    }
                }
            }
        )

        async def generate_actions():
            for item_id, item in instance.data.index.items():
                if not item.name and not item.docs:
                    continue
                fullname = instance.resolve_full_name(item_id)
                kind = next(iter(item.inner)) if item.inner else "unknown"
                yield {
                    "_index": index_name,
                    "_id": str(item_id),
                    "_source": {
                        "id": str(item_id),
                        "name": item.name or "",
                        "full_name": fullname,
                        "docs": item.docs or "",
                        "kind": kind,
                    }
                }

        logger.info(f"Indexing RustDoc items into '{index_name}'...")
        await helpers.async_bulk(es_client, generate_actions())
        await es_client.indices.refresh(index=index_name)
        logger.info("Indexing completed successfully.")

    @classmethod
    async def create_from_json(
        cls, 
        path: Path, 
        host: str = "http://localhost:9200", 
        pubapi_path: Optional[Path] = None, 
        index_name: Optional[str] = None
    ) -> Self:
        if not path.exists():
            raise FileNotFoundError(f"JSON file not found at {path}")

        file_content = path.read_bytes()
        file_hash = hashlib.md5(file_content).hexdigest()[:8]
        raw_data = json.loads(file_content.decode("utf-8"))

        if "root" in raw_data:
            raw_data["root"] = str(raw_data["root"])

        crate_data = CrateData(**raw_data)
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
        
        root_name = crate_data.index[crate_data.root].name or "unknown"
        version = crate_data.crate_version or "unknown"
        final_index_name = index_name or f"rustdoc_{root_name}_{version}_{file_hash}".replace(".", "_").lower()

        es_client = AsyncElasticsearch(hosts=[host])
        
        async_instance = cls(
            es_client=es_client, 
            index_name=final_index_name,
            data=crate_data,
            parent_map=parent_map,
            impl_to_type_map=impl_to_type_map,
            pubapi_path=pubapi_path
        )
        
        await cls._init_elasticsearch(es_client, final_index_name, async_instance)
        return async_instance

    @classmethod
    async def create_from_libdir(cls, host_lib_dir: Path, host: str = "http://localhost:9200") -> Self:
        doc_path = host_lib_dir / "target" / "doc"
        pubapi_path = host_lib_dir / "pubapi.txt"
        json_path = None
        if doc_path.exists():
            for possible_json in doc_path.glob("*.json"):
                json_path = possible_json
                break

        if json_path and json_path.exists():
            return await cls.create_from_json(json_path, host=host, pubapi_path=pubapi_path)
        else:
            raise FileNotFoundError(f"Could not find rustdoc json in {doc_path}.")

    def _load_pubapi(self, path: Path) -> None:
        if not path.exists():
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line.startswith("pub "):
                        continue

                    parts = line.split(" ")
                    name_part = None
                    for p in parts:
                        if "::" in p:
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
        prev = name
        while True:
            new = re.sub(r"<[^<>]*>", "", prev)
            if new == prev:
                break
            prev = new
        return prev

    def resolve_full_name(self, item_id: str, depth: int = 0) -> str:
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
        if not type_obj:
            return "_"
        if "primitive" in type_obj:
            return type_obj["primitive"]
        if "resolved_path" in type_obj:
            path_obj = type_obj["resolved_path"]
            name = path_obj.get("name", "")
            if not name and "path" in path_obj:
                name = path_obj["path"]
            args = path_obj.get("args")
            if args:
                if "angle_bracketed" in args:
                    ab = args["angle_bracketed"]
                    arg_list = []
                    for arg in ab.get("args", []):
                        if "type" in arg:
                            arg_list.append(self._render_type(arg["type"]))
                        elif "const" in arg:
                            arg_list.append(f"const {arg.get('const', {}).get('expr', '_')}")

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
        """Convert AST definitions into a readable structured text (functions, structs, enums)."""
        if self.pubapi_map:
            full_name = self.resolve_full_name(str(item.id))
            if full_name:
                matches = self.pubapi_map.get(full_name)
                if matches:
                    return "\n".join(list(set(matches)))

        if not item.inner:
            return None

        kind = next(iter(item.inner))
        if kind == "function":
            fn_obj = item.inner["function"]
            sig = fn_obj.get("sig", {})

            inputs = []
            for arg in sig.get("inputs", []):
                if len(arg) == 2:
                    name, type_val = arg
                    type_str = self._render_type(type_val)
                    inputs.append(f"{name}: {type_str}")

            output_str = ""
            output = sig.get("output")
            if output:
                output_str = f" -> {self._render_type(output)}"

            fn_name = item.name or "_"
            return f"pub fn {fn_name}({', '.join(inputs)}){output_str}"

        elif kind == "struct":
            struct_obj = item.inner["struct"]
            fields = []
            kind_inner = struct_obj.get("kind", {})
            if isinstance(kind_inner, dict) and "plain" in kind_inner:
                plain_fields = kind_inner["plain"].get("fields", [])
                for f_id in plain_fields:
                    f_item = self.data.index.get(str(f_id))
                    if f_item and f_item.inner and "struct_field" in f_item.inner:
                        f_type = self._render_type(f_item.inner["struct_field"])
                        vis = f_item.visibility
                        vis_str = "pub " if vis == "public" or vis == "default" else "" 
                        fields.append(f"{vis_str}{f_item.name}: {f_type}")
            
            if not fields:
                return f"pub struct {item.name or '_'} {{ /* No public fields or opaque */ }}"

            fields_str = ",\n    ".join(fields)
            return f"pub struct {item.name or '_'} {{\n    {fields_str}\n}}"

        elif kind == "enum":
            enum_obj = item.inner["enum"]
            variants = []
            for v_id in enum_obj.get("variants", []):
                v_item = self.data.index.get(str(v_id))
                if v_item:
                    variants.append(v_item.name or "_")
            if variants:
                return f"pub enum {item.name or '_'} {{\n    " + ",\n    ".join(variants) + "\n}}"
            else:
                return f"pub enum {item.name or '_'} {{}}"

        return None

    def _get_associated_methods(self, item: Item) -> Optional[List[str]]:
        """Strictly follow JSON AST to extract associated inherent methods for structs and enums."""
        methods = []
        if not item.inner:
            return None

        kind = next(iter(item.inner))
        if kind not in ["struct", "enum"]:
            return None

        impls = item.inner[kind].get("impls", [])
        for impl_id in impls:
            impl_item = self.data.index.get(str(impl_id))
            if not impl_item or not impl_item.inner or "impl" not in impl_item.inner:
                continue
            
            # Skip trait impls, we primarily want inherent associated functions/methods
            if impl_item.inner["impl"].get("trait") is not None:
                continue

            for method_id in impl_item.inner["impl"].get("items", []):
                method_item = self.data.index.get(str(method_id))
                if method_item and method_item.name:
                    methods.append(method_item.name)
        
        # Deduplicate and sort
        if not methods:
            return None
        return sorted(list(set(methods)))

    def _extract_safe_docs(self, full_docs: str) -> str:
        """Return the documentation safely without arbitrarily cutting inside syntax."""
        if not full_docs:
            return ""
        
        # Heuristic: LLM usually only needs the top summary logic.
        # We grab the first few paragraphs strictly respecting double newlines (markdown blocks).
        parts = full_docs.split("\n\n")
        
        safe_lines = []
        total_len = 0
        for p in parts:
            if total_len > 400: # Soft limit to prevent blowing up context with massive doc pages
                safe_lines.append("... (Content truncated)")
                break
            safe_lines.append(p)
            total_len += len(p)
            
        return "\n\n".join(safe_lines)

    async def _handle_es_response(self, response: Any) -> List[SearchResult]:
        matches = []
        for hit in response["hits"]["hits"]:
            item_id = hit["_id"]
            score = hit["_score"]
            
            item = self.data.index.get(item_id)
            if not item:
                continue

            fullname = hit["_source"].get("full_name", "")
            kind = hit["_source"].get("kind", "unknown")

            docs = self._extract_safe_docs(item.docs or "")
            signature = self._render_signature(item)
            methods = self._get_associated_methods(item)

            matches.append(
                SearchResult(
                    name=fullname,
                    kind=kind,
                    score=int(score * 10),
                    docs=docs,
                    signature=signature,
                    associated_methods=methods
                )
            )
        return matches

    async def search_docs(self, query: str, limit: int = 10) -> List[SearchResult]:
        response = await self.es_client.search(
            index=self.index_name,
            body={
                "query": {
                    "match": {
                        "docs": query
                    }
                },
                "size": limit
            }
        )
        return await self._handle_es_response(response)

    async def search_symbol(self, query: str, limit: int = 10) -> List[SearchResult]:
        response = await self.es_client.search(
            index=self.index_name,
            body={
                "query": {
                    "bool": {
                        "should": [
                            {"term": {"name.keyword": {"value": query, "boost": 100.0}}},
                            {"prefix": {"name.keyword": {"value": query.lower(), "boost": 50.0}}},
                            {"match": {"name": {"query": query, "boost": 10.0}}}
                        ],
                        "minimum_should_match": 1
                    }
                },
                "size": limit
            }
        )
        return await self._handle_es_response(response)

    async def search(self, query: str, limit: int = 10) -> List[SearchResult]:
        response = await self.es_client.search(
            index=self.index_name,
            body={
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["name.keyword^100", "name^50", "full_name^10", "docs^1"]
                    }
                },
                "size": limit
            }
        )
        return await self._handle_es_response(response)
        
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
            # Reuse the safe extraction but maybe a bit more for the root overview
            lines.append(self._extract_safe_docs(root_item.docs))

        lines.append("\n## Module Hierarchy")

        def get_public_children_ids(item: Item) -> List[str]:
            if "module" in item.inner:
                return [str(i) for i in item.inner["module"].get("items", [])]
            return []

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

    async def close(self):
        await self.es_client.close()
