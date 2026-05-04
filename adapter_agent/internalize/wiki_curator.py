import logging
import re

from coder_mcp.runtime import Runtime
from oai_utils.agent import AgentsSDKModel, AgentWrapper

from adapter_agent.library.wiki_manager import WikiManager

logger = logging.getLogger(__name__)


MAX_TURNS = 30

TAG_NAMES = ("read_file", "write_file", "str_replace", "run_cargo", "finish")
TAG_RE = re.compile(
    r"<(" + "|".join(TAG_NAMES) + r")\s*(?:/>|>(.*?)</\1>)",
    re.DOTALL,
)
SUBFIELD_RE = re.compile(r"<(\w+)>(.*?)</\1>", re.DOTALL)


def _parse_subfields(body: str) -> dict[str, str]:
    return {m.group(1): m.group(2).strip() for m in SUBFIELD_RE.finditer(body)}


# Style anchors. Three hypothetical libraries from very different domains
# (HTTP routing, graph traversal, markdown parsing) — chosen to be unrelated
# to each other and to any numerical / array library, so the curator absorbs
# *structural* style rather than copying domain-specific patterns.
STYLE_EXAMPLES = '''### Style examples (3 anchors from unrelated domains)

These examples document three completely different hypothetical libraries:
an HTTP router, a graph traversal library, and a markdown parser. They have
*different* article structures — picked to fit each content, not from a
fixed template. Use them as inspiration for shape and depth, not as a
rigid format. Numbers of subsections, names of subsections, and presence
of a "Why" section all vary by article.

#### Example 1 — Multi-pattern article (multiple ways to do one thing)

When the Final Answer demonstrates several related patterns under a single
umbrella concept, group them into one article with multiple `## Pattern:`
subsections.

Final Answer fragment (hypothetical HTTP router `routera`):

```rust
use routera::Router;
use routera::http::{get, post};

let app = Router::new()
    .route("/users", get(list_users))
    .route("/users", post(create_user))
    .route("/users/:id", get(get_user));

app.serve("0.0.0.0:8080");
```

Article — `route_registration.md`:

```markdown
# Route Registration in routera

This article covers how to register HTTP routes on a `routera::Router`.

## Pattern: Single-Method Routes

Use `.route(path, method_handler)` to bind one HTTP method to one path.

```rust
use routera::Router;
use routera::http::get;

let app = Router::new().route("/users", get(list_users));
```

The second argument is a *method handler* — built with `get()`, `post()`, `put()`, `delete()` from `routera::http`. Passing the handler function directly without wrapping it (e.g. `.route("/users", list_users)`) is a compile error: the type does not tell the router which HTTP method to expect.

## Pattern: Multiple Methods, Same Path

Chain `.route()` calls to bind several methods to the same path. The router merges them internally — no special syntax needed.

```rust
let app = Router::new()
    .route("/users", get(list_users))
    .route("/users", post(create_user));
```

## Pattern: Path Parameters

Path segments prefixed with `:` are captured as parameters and made available to the handler:

```rust
let app = Router::new().route("/users/:id", get(get_user));
```
```

MoC entry:

```
- [route_registration.md] Bind HTTP routes to handlers on a `routera::Router` (`.route(path, get(handler))` family, plus path parameters with `:name`). Use this when wiring up an HTTP server or recovering from "expected MethodHandler, found fn(...)" type errors.
```

#### Example 2 — Single-API deep dive (one function, exhaustively)

When a single API has subtle constraints worth documenting in detail, focus
the article on that one function. No `## Pattern:` headers — instead use
sections that name the specific concerns (Return Type, Why, etc.).

Final Answer fragment (hypothetical graph library `graphlet`):

```rust
use graphlet::Graph;
use graphlet::traverse::bfs;

let mut g: Graph<&str> = Graph::new();
g.add_edge("a", "b");
g.add_edge("b", "c");

let path = bfs(&g, "a", "c").expect("no path");
```

Article — `bfs_traversal.md`:

```markdown
# BFS Traversal in graphlet

`graphlet::traverse::bfs` performs breadth-first search and returns the shortest path between two nodes by edge count. It does not live as a method on `Graph` — it must be imported from `graphlet::traverse`.

## Usage

```rust
use graphlet::Graph;
use graphlet::traverse::bfs;

let mut g: Graph<&str> = Graph::new();
g.add_edge("a", "b");
g.add_edge("b", "c");

let path: Vec<&str> = bfs(&g, "a", "c").expect("no path");
```

## Return Type

`bfs` returns `Option<Vec<N>>`, where `N` is the node label type:

- `Some(path)` — `path[0]` is the start node, `path.last()` is the destination.
- `None` — the destination is unreachable from the start.

This is `Option`, **not** `Result`. There is no error type; either a path exists or it does not. Calling `.unwrap_err()` on the return value is a type error.

## Why Option, not Result

A graph traversal can have exactly one of two outcomes — found or not found — and "not found" is an expected, normal case for many call sites. `Option` matches this shape and avoids forcing every caller through error-handling boilerplate for a routine condition.
```

MoC entry:

```
- [bfs_traversal.md] Find shortest path by edge count via `graphlet::traverse::bfs`. Use this when traversing a graph and you need a path; covers the `Option<Vec<N>>` return shape and why it is not `Result`.
```

#### Example 3 — Sibling APIs with a shared idiom

When the Final Answer touches two or more closely related functions that share
a calling convention, document them together — emphasize the shared idiom
once, then list the differences.

Final Answer fragment (hypothetical markdown parser `mdline`):

```rust
use mdline::{parse_block, parse_inline};
use mdline::ast::Node;

let block_ast: Node = parse_block("# Title\\n\\nA paragraph.");
let inline_ast: Node = parse_inline("Some **bold** text.");
```

Article — `parsing_entry_points.md`:

```markdown
# Markdown Parsing Entry Points in mdline

`mdline` exposes two top-level parsing functions, `parse_block` and `parse_inline`, which differ in what fragment of Markdown they accept. Both return an `mdline::ast::Node` rooted at a fresh `Document`.

## Shared Idiom

Both functions take a `&str` and return `Node` — no `Result`, no `Option`. Malformed input produces an AST with `Node::Error` leaves rather than a parse failure.

```rust
use mdline::{parse_block, parse_inline};
use mdline::ast::Node;

let block_ast: Node = parse_block("# Title\\n\\nA paragraph.");
let inline_ast: Node = parse_inline("Some **bold** text.");
```

## parse_block vs parse_inline

- `parse_block` accepts the **full Markdown grammar** — headings, lists, code fences, paragraphs. Use this for complete documents or document fragments.
- `parse_inline` accepts only **inline-level constructs** — emphasis, links, code spans, plain text. Block-level syntax in the input (a `#` heading, for example) is treated as literal text, not as a heading.

Pick `parse_block` when in doubt; pick `parse_inline` when you have a single line you know contains no block-level structure (e.g. a table cell's contents).

## Why both functions

A heading inside an inline context (say, the body of a list item) should not start a new heading — it is just a literal `#` followed by text. The two-function split lets the parser commit to one or the other interpretation up front, avoiding ambiguity.
```

MoC entry:

```
- [parsing_entry_points.md] Parse Markdown via `mdline::parse_block` (full grammar) and `mdline::parse_inline` (inline only). Use this when choosing the right entry point, or when a `#` in inline-parsed input is unexpectedly literal.
```

#### What the three examples teach

- Article structure adapts to the content: multi-pattern grouping, single-API deep dive, sibling-API comparison are all valid shapes. Pick the one that fits the Final Answer in front of you.
- Sections like "Type Constraints", "Return Type", "Why this matters" appear *only when there is something non-obvious to say* — not as required boilerplate.
- MoC entries name the specific APIs and trigger phrases ("Use this when …") and call out the error codes or pitfalls the article helps recover from.
- Code examples are minimal but complete enough to be runnable.
- These examples are from unrelated domains (routing / graphs / parsing). Do not import their idioms; absorb the *style of decomposition*, then apply it to whatever the Final Answer is actually about.

'''


class WikiCurator:
    """One-stage agent that mines a Task + Final Answer for library knowledge
    and commits it to the Wiki in a single autonomous run.

    Replaces the Reflector → WikiIntegrator pipeline: the act of deciding
    what is worth recording IS the integration step, so they share one LLM
    run instead of being split via a Reflection intermediate representation.
    """

    def __init__(
        self,
        wiki_manager: WikiManager,
        model: AgentsSDKModel,
        library_name: str,
        qwen_no_think: bool = False,
    ):
        self.wiki_manager = wiki_manager
        self.model = model
        self.library_name = library_name
        self.qwen_no_think = qwen_no_think

    async def curate(
        self,
        task_instruction: str,
        final_answer: str,
        runtime: Runtime,
    ) -> None:
        """
        Runs a single autonomous loop that mines the Final Answer for
        non-obvious {library_name} knowledge and integrates it into the Wiki.
        """
        instructions = (
            f"You are the Wiki Knowledge Curator for the **{self.library_name}** library. Your job is to mine the agent's verified-working solution for non-obvious {self.library_name} knowledge and commit it to a versioned Wiki — in a single autonomous run.\n\n"
            "### Source Material\n"
            "<Task>\n"
            f"{task_instruction}\n"
            "</Task>\n\n"
            "<FinalAnswer>\n"
            f"{final_answer}\n"
            "</FinalAnswer>\n\n"
            f"The Final Answer is the agent's verified-working solution. It is your authoritative source of correct `{self.library_name}` API usage. You may adapt examples from it (simplify, rename, change literal values, drop unrelated boilerplate) — you do not have to quote it verbatim. The hard constraint is on the APIs themselves: do NOT introduce APIs, function names, or module paths that are absent from the Final Answer's imports and call sites. If you must check a variation, use `<run_cargo>`.\n\n"
            "### Wiki Structure\n"
            "The Wiki has two layers, both consumed by future LLM agents (not humans):\n"
            "- **Articles**: Each article is a self-contained how-to that a future agent can read once and apply. Title states the topic precisely; body has a brief context (1–2 sentences), at least one runnable code example (adapted from the Final Answer; using only APIs that appear there), and any non-obvious notes (imports, return types, argument shapes). The body must be plain markdown — do NOT include YAML frontmatter (`---\\ntitle: ...\\n---`), HTML metadata, or any other front-of-file metadata block.\n"
            "- **MOC.md** (Map of Content): The sole navigation hub. Future agents read this first and pick which articles to consult. Each entry has a one-line summary and a usage trigger (\"use this when …\"). Library overviews, table-of-contents pages, and any other index-style writing live here, never as standalone articles.\n\n"
            "### Principles\n"
            f"1. **Library focus, traced through imports**: Every article must document `{self.library_name}` itself — APIs reached via `use {self.library_name}::...`, or methods on types that came from `{self.library_name}`. Imports in the Final Answer are the boundary of what counts as `{self.library_name}` knowledge. Functions or types defined inside the Final Answer body (e.g. a local `fn calculate_strides(...) {{ ... }}`) are the agent's own scaffolding, NOT library knowledge — never document them, even when the Task hints at the concept they compute. Do not promote alternative crates as substitutes. General Rust patterns are acceptable only when they appear in service of using `{self.library_name}`.\n"
            "2. **How-to framing**: Articles describe how to do something correctly — the working API, the right module path, the idiomatic pattern. Do not write troubleshooting entries or cautionary tales. Mention pitfalls only briefly inside a how-to.\n"
            "3. **Atomic by reuse**: An article covers one task-shaped knowledge unit — the granularity at which a future agent scanning the MOC says \"this is what I need.\" Group items that together solve one task; do not split into one article per struct or function unless each is independently the knowledge unit.\n"
            "4. **Title–content alignment**: An article's title must describe exactly what its body covers. A title broader than the content misleads readers.\n"
            "5. **Flat naming**: All articles live at the wiki root with descriptive, task-shaped filenames like `array_construction.md`, `concatenate_along_axis.md`. Do NOT create folders or nested paths. The consumer is an LLM agent that picks articles from MOC.md, not a human who browses folders — folders only add navigation cost. Each filename must end with `.md`.\n"
            "6. **Uniqueness**: Audit before writing. If a concept is already well-covered, do nothing. If new knowledge is a variant of an existing article (a different approach to the same task, an alternative API, an additional argument shape), update that article rather than creating a parallel one — pick whichever editing tool (`<write_file>` overwrite or `<str_replace>` surgical edit) is easier for the change at hand.\n"
            "7. **Splitting**: If an existing article has drifted into covering multiple unrelated tasks, split it into atomic articles as part of this curation.\n"
            "8. **Concrete code examples**: Every article that documents an API or pattern includes at least one runnable code example. Examples may be adapted from the Final Answer — simplify, rename variables, change literal values, drop unrelated boilerplate. The constraint that does survive is Principle 1: the APIs (function names, methods, types, module paths) you use in the example must be ones that actually appear in the Final Answer. Do not introduce new API names by analogy.\n\n"
            + STYLE_EXAMPLES
            + "### Mining: what to extract from the Final Answer\n"
            f"A future agent who has *completely forgotten* {self.library_name} should be able to reproduce code like the Final Answer by reading 1–3 articles you author. So look for the bits of knowledge that don't follow from general Rust skill alone:\n"
            "- **Where things live**: module paths, prelude membership, error/Result type location. Imports are a strong signal — they tell you what a forgetful future agent must `use`.\n"
            "- **Call shapes**: free function vs method, argument types (slice of refs vs single ref, axis as int, etc.), return-type idioms (Result, Option, plain).\n"
            "- **Surprising names**: non-standard constructor names (`from_vec` vs `new`), conversion helpers, etc.\n"
            "- **Idiomatic clusters**: when several APIs naturally co-occur to solve one task, treat them as one concept (one article).\n"
            f"Skip: generic Rust patterns (e.g., `?` propagation, `Result<()>` in main) that aren't specific to {self.library_name}; helper functions defined locally in the Final Answer body (these are scaffolding, not library knowledge); restatements of things already in the Wiki.\n\n"
            "### When to do nothing\n"
            "Producing zero articles is a correct outcome — not a failure. Emit `<finish />` immediately when any of these signals hold:\n"
            f"- The bulk of the Final Answer is a locally-defined helper function (`fn ...() {{ ... }}` the agent wrote itself). Per Principle 1, helpers are NOT documentable.\n"
            f"- The agent's solution implements logic by hand because `{self.library_name}` did not expose the needed feature. There is nothing about `{self.library_name}` to record here.\n"
            f"- Every API actually called from the `use {self.library_name}::...` imports in the Final Answer is already covered by an existing article in the wiki state shown to you.\n"
            f"- The only `{self.library_name}` content is import statements of items that already have articles.\n\n"
            "If you find yourself drafting an article whose code example is mostly a hand-rolled function rather than a library API, stop and `<finish />`. That's the agent's scaffolding, not library knowledge — capturing it pollutes the knowledge base.\n\n"
            "### Tool Tags\n"
            "You drive this task by emitting exactly ONE XML tool tag per response. Free-form text outside the tag (analysis, plans, comments) is ignored — only the tag is executed.\n"
            "Available tags:\n"
            "- `<read_file>path/to/article.md</read_file>` — read an article from the listing in your first message.\n"
            "- `<write_file><path>filename.md</path><content>...full markdown body...</content></write_file>` — create or overwrite an article. Use this freely; if you need to revise an existing article, you may either rewrite it via `<write_file>` or do a surgical edit via `<str_replace>`.\n"
            "- `<str_replace><path>path/to/article.md</path><old_str>...</old_str><new_str>...</new_str></str_replace>` — atomic in-place edit.\n"
            "- `<run_cargo>...full Rust source for src/main.rs...</run_cargo>` — write the code to `src/main.rs` and run `cargo run`. Use this to verify a variation before recording it.\n"
            "- `<finish />` — end the run.\n"
            "Rules:\n"
            "- Emit only ONE tag per response. Multiple tags will be rejected and you will be asked to retry.\n"
            "- Multi-argument tags (`write_file`, `str_replace`) require their named sub-tags. Do NOT use JSON, attributes, or any other format.\n"
            "- Tag bodies are taken verbatim. Inside `<run_cargo>` / `<old_str>` / `<new_str>` / `<content>`, write text DIRECTLY — actual newlines (press Enter), actual quotes, actual backslashes. Do not write the two-character sequence `\\n`, do not escape quotes as `\\\"`, do not wrap in JSON.\n"
            "- Bodies must not contain the closing tag of the enclosing tool (e.g., `<run_cargo>` body cannot contain the literal `</run_cargo>`).\n\n"
            "### Using <str_replace> correctly\n"
            "`<old_str>` must match exactly one location in the file character-for-character — including indentation, surrounding blank lines, and any punctuation. Pick an `<old_str>` 2–3 lines long: short enough to be unambiguous, long enough to be unique.\n\n"
            "Example. Suppose `greeting.md` currently contains:\n\n"
            "    # Greeting\n\n"
            "    The default greeting is:\n\n"
            "    > Hello, world!\n\n"
            "To append a multilingual note after the quote, the correct call is:\n\n"
            "<str_replace>\n"
            "<path>greeting.md</path>\n"
            "<old_str>The default greeting is:\n\n"
            "> Hello, world!</old_str>\n"
            "<new_str>The default greeting is:\n\n"
            "> Hello, world!\n\n"
            "In other languages it is \"Bonjour\" (FR) or \"Hola\" (ES).</new_str>\n"
            "</str_replace>\n\n"
            "Note how every newline in `<old_str>` and `<new_str>` is an actual newline character — written by pressing Enter. The text `\\n` (a literal backslash followed by `n`) does not match a newline in the file and will produce `Target string not found`. The same applies to quotes and backslashes: write them bare, not escaped.\n"
            "If `Target string not found` is returned, do NOT retry the same `<old_str>`. Re-`<read_file>` the article, copy a different unique passage exactly, and try again.\n\n"
            "### Workflow\n"
            "1. **Survey**: your first user message contains the full list of existing articles and the current MOC.md. Skim them. `<read_file>` any article that might already cover what you'd extract.\n"
            f"2. **Mine**: identify the non-obvious `{self.library_name}` knowledge in the Final Answer. Many trajectories yield 1–3 articles; many yield zero. Both are correct outcomes — quality matters more than count. If you cannot point to a `{self.library_name}` API in the Final Answer that isn't already covered (see the “When to do nothing” signals above), jump straight to `<finish />`.\n"
            "3. **Verify**: code lifted directly from the Final Answer is already verified — use it as-is. Use `<run_cargo>` only for variations you adapt or invent.\n"
            "4. **Synthesize**: write or update articles. Use `<write_file>` for new articles or full rewrites; use `<str_replace>` for surgical edits to existing ones. Each article must include at least one runnable code example (adapted from the Final Answer; only APIs that appear there).\n"
            "5. **Update MOC.md**: when you create or significantly revise an article, update its MOC.md entry (one-line summary + \"use this when …\" trigger).\n"
            "6. **Finish**: `<finish />` when the wiki accurately reflects the new knowledge — or immediately if there's nothing worth adding.\n"
        )

        agent = AgentWrapper[str].create(
            name="WikiCurator",
            instructions=instructions,
            model=self.model,
        )

        snapshot = await self._render_initial_snapshot()
        first_user = (
            "Curate the Wiki using the Task and Final Answer above as your source material.\n\n"
            "### Current Wiki State\n"
            f"{snapshot}"
        )
        if self.qwen_no_think:
            first_user = "/no_think " + first_user

        history: list[dict] = [{"role": "user", "content": first_user}]

        logger.info(
            f"Starting WikiCurator run for task: {task_instruction[:80]}..."
        )
        logger.info("[curator first user]\n%s", first_user)
        for turn_idx in range(MAX_TURNS):
            result = await agent.run(history)
            assistant_text = result.final_output()
            history.append({"role": "assistant", "content": assistant_text})

            logger.info("[curator turn %d / assistant]\n%s", turn_idx + 1, assistant_text)

            matches = list(TAG_RE.finditer(assistant_text))

            if not matches:
                logger.warning("[curator turn %d] NO_TOOL_TAG", turn_idx + 1)
                history.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM ERROR] NO_TOOL_TAG\n"
                        "No recognized tool tag was found in your response. Emit exactly one of: "
                        "<ls />, <ls>folder/</ls>, <read_file>...</read_file>, <write_file>...</write_file>, "
                        "<str_replace>...</str_replace>, <run_cargo>...</run_cargo>, <finish />."
                    ),
                })
                continue

            if len(matches) > 1:
                names = [m.group(1) for m in matches]
                logger.warning(
                    "[curator turn %d] MULTIPLE_TOOL_TAGS: %s", turn_idx + 1, names
                )
                history.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM ERROR] MULTIPLE_TOOL_TAGS\n"
                        f"Detected: {names}. Emit only ONE tool tag per response. Re-think and re-send a single action."
                    ),
                })
                continue

            match = matches[0]
            tag_name = match.group(1)
            body = (match.group(2) or "").strip()

            logger.info(
                "[curator turn %d] tool=%s body[:200]=%r",
                turn_idx + 1,
                tag_name,
                body[:200],
            )

            if tag_name == "finish":
                logger.info("WikiCurator finished.")
                return

            tool_result = await self._dispatch(tag_name, body, runtime)
            logger.info(
                "[curator turn %d / tool result]\n%s",
                turn_idx + 1,
                tool_result,
            )
            history.append({"role": "user", "content": tool_result})

        logger.warning(
            f"WikiCurator hit max_turns ({MAX_TURNS}) without <finish />."
        )

    async def _render_initial_snapshot(self) -> str:
        titles = await self.wiki_manager.ls(None) or []
        listing = "\n".join(sorted(titles)) if titles else "(empty — no articles yet)"
        moc = await self.wiki_manager.read("MOC.md")
        moc_text = moc if moc is not None else "(MOC.md does not exist yet — create it as you write articles)"
        return (
            f"<articles>\n{listing}\n</articles>\n\n"
            f"<MOC>\n{moc_text}\n</MOC>"
        )

    async def _dispatch(self, tag_name: str, body: str, runtime: Runtime) -> str:
        if tag_name == "read_file":
            return await self._tool_read_file(body)
        if tag_name == "write_file":
            fields = _parse_subfields(body)
            return await self._tool_write_file(
                fields.get("path", ""), fields.get("content", "")
            )
        if tag_name == "str_replace":
            fields = _parse_subfields(body)
            return await self._tool_str_replace(
                fields.get("path", ""),
                fields.get("old_str", ""),
                fields.get("new_str", ""),
            )
        if tag_name == "run_cargo":
            return await self._tool_run_cargo(body, runtime)
        return f"<error>Unknown tool: {tag_name}</error>"

    async def _tool_read_file(self, path: str) -> str:
        if not path:
            return "<read_error>Missing path. Use <read_file>path/to/article.md</read_file>.</read_error>"
        content = await self.wiki_manager.read(path)
        if content is None:
            return f"<read_error>Article '{path}' not found.</read_error>"
        return f"<file path=\"{path}\">\n{content}\n</file>"

    async def _tool_write_file(self, path: str, content: str) -> str:
        if not path:
            return "<write_error>Missing <path> sub-tag.</write_error>"
        await self.wiki_manager.write(path, content)
        return f"<write_result>Wrote '{path}'.</write_result>"

    async def _tool_str_replace(self, path: str, old_str: str, new_str: str) -> str:
        if not path or not old_str:
            return "<str_replace_error>Missing required sub-tag (<path> and <old_str>).</str_replace_error>"
        success = await self.wiki_manager.str_replace(path, old_str, new_str)
        if success:
            return f"<str_replace_result>Updated '{path}' atomically.</str_replace_result>"
        return f"<str_replace_error>Target string not found in '{path}'.</str_replace_error>"

    async def _tool_run_cargo(self, code: str, runtime: Runtime) -> str:
        if not code:
            return "<cargo_error>Empty body. Place the full Rust source inside <run_cargo>...</run_cargo>.</cargo_error>"
        await runtime.set_content("src/main.rs", code)
        run_ret, run_success = await runtime.run_cargo()
        return (
            f"<cargo_result success=\"{str(run_success).lower()}\">\n"
            f"{run_ret}\n"
            f"</cargo_result>"
        )
