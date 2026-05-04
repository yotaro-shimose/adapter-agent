from datetime import datetime, timezone

import polars as pl


def main():
    base_dir = "data/createsio/2026-05-04-020032/data"
    crates_path = f"{base_dir}/crates.csv"
    downloads_path = f"{base_dir}/crate_downloads.csv"
    dependencies_path = f"{base_dir}/dependencies.csv"
    versions_path = f"{base_dir}/versions.csv"

    print("Loading dataframes...")
    df_crates = pl.read_csv(crates_path, ignore_errors=True)
    df_downloads = pl.read_csv(downloads_path)
    df_deps = pl.read_csv(dependencies_path, ignore_errors=True)
    df_versions = pl.read_csv(versions_path, ignore_errors=True)

    # Date parsing
    def parse_date(x):
        try:
            return datetime.fromisoformat(x)
        except (ValueError, TypeError, AttributeError):
            try:
                if isinstance(x, str):
                    return datetime.fromisoformat(x.replace("Z", "+00:00"))
            except Exception:
                return None
            return None

    print("Parsing dates...")
    df_crates = df_crates.with_columns(
        pl.col("created_at")
        .map_elements(parse_date, return_dtype=pl.Datetime("us", "UTC"))
        .alias("created_at")
    )

    # 1. Repository Age Filter
    print("Filtering old repositories...")
    repo_start_dates = (
        df_crates.lazy()
        .filter(pl.col("repository").is_not_null())
        .group_by("repository")
        .agg(pl.min("created_at").alias("repo_start_date"))
        .collect()
    )
    old_repo_cutoff = datetime(2024, 7, 1, tzinfo=timezone.utc)
    new_repos = repo_start_dates.filter(pl.col("repo_start_date") >= old_repo_cutoff)

    # Filter crates to only those in new repos
    df_crates_new_repos = df_crates.join(new_repos, on="repository", how="inner")

    # 2. Dependency Analysis
    print("Analyzing dependencies...")
    # Map version_id -> crate_id
    version_to_crate = df_versions.select(["id", "crate_id"]).rename(
        {"id": "version_id", "crate_id": "dependant_id"}
    )

    # Enrich dependencies with dependant_id
    df_deps_enriched = df_deps.lazy().join(
        version_to_crate.lazy(), on="version_id", how="inner"
    )

    # Calculate direct dependant count (unique crates depending on this crate)
    direct_dependants = (
        df_deps_enriched.group_by("crate_id")
        .agg(pl.col("dependant_id").n_unique().alias("direct_dependant_count"))
        .collect()
    )

    # Identify "Frontend" crates:
    # 1. Join dependencies with repos for both sides
    repo_mapping = df_crates.select(["id", "repository"]).rename(
        {"id": "crate_id", "repository": "repo"}
    )

    internal_deps = (
        df_deps_enriched.join(
            repo_mapping.lazy(), on="crate_id", how="inner"
        )  # target crate repo
        .rename({"repo": "target_repo"})
        .join(
            repo_mapping.lazy(),
            left_on="dependant_id",
            right_on="crate_id",
            how="inner",
        )  # dependant crate repo
        .rename({"repo": "dependant_repo"})
        .filter(pl.col("target_repo") == pl.col("dependant_repo"))
        .select(["crate_id"])
        .unique()
        .collect()
    )

    # Crates that ARE dependencies within their own repo
    internal_dep_ids = internal_deps["crate_id"].to_list()

    # 3. Quality Filters & Frontend filtering
    print("Applying quality and frontend filters...")

    # Generic suffix/keyword patterns that indicate "infrastructure" crates
    # (binding-side, macros, sys, etc.) rather than end-developer-facing libs.
    name_exclude_pattern = (
        r"(?i)"
        r"(^|[-_])("
        r"core|utils?|util|common|commons|shared|internal|internals|"
        r"impl|imp|sys|ffi|abi|"
        r"derive|macro|macros|proc-macro|proc_macro|"
        r"types|trait|traits|"
        r"builder|builders|helper|helpers|"
        r"codec|codecs|"
        r"tutorial"
        r")([-_]|$)"
    )

    # Specific crate-name prefixes/exact names known to be monorepo siblings
    # or org-internal infrastructure observed in the result set.
    name_blacklist_pattern = (
        r"(?i)^("
        # monorepo / ecosystem prefixes
        r"commonware[-_].*|"
        r"oak[-_].*|"
        r"nargo[-_].*|"
        r"greentic[-_].*|"
        r"pimalaya[-_].*|"
        r"qubit[-_].*|"
        r"zarrs[-_].*|"
        r"casper[-_].*|"
        r"rustcrypto[-_].*|"
        r"tg[-_].*|"
        r"molten[-_].*|"
        r"oxidized[-_].*|"
        r"gpui[-_].*|"
        r"wp-connector[-_].*|"
        # exact infra-ish names
        r"kspin|lazyinit|axio|axcpu|axruntime|percpu|page_table_multiarch|"
        r"deno_error|idna_adapter|"
        r"pyo3-async-runtimes|variadics_please|"
        r"astral-tokio-tar|frame-decode|"
        r"oci-client|wae-request|hopper-runtime|"
        r"glyphs|lacquer|"
        r"io-stream|"
        # network / HTTP / TLS clients
        r"http-kit|reqx|tokio-boring2|four-word-networking|yt-dlp|"
        # proc-macro infra
        r"mokuya|kenzu|"
        # blockchain / zk
        r"tycho-simulation|atlas-metrics|spacedb|ff-arcium-fork|"
        r"jubjub-plus|uselesskey|"
        # crypto wrappers (end-dev but off-theme)
        r"libsodium-rs|saorsa-pqc|elliptic-curve-tools|"
        # OS / fs / assembler internals
        r"rsext4|x86_64-assembler|userspace_build|bladvak|"
        # narrow / template / asset
        r"propagators-chirho|error-envelope|"
        # web frameworks
        r"rkt|rocket-community|tower-sessions-ext|"
        # network/IPC clients & wrappers
        r"wp-mini|getifs|gntp|otlp2records|"
        # blockchain / on-chain instrumentation
        r"bellscoin|trezoa-svm-measure|crankx|"
        # OS-internal / Windows API tricks
        r"dinvk|"
        # AI-related internal toolkit
        r"apcore-toolkit|"
        # alacritty-internal fork
        r"unicode-width-16|"
        # AI / ML backends
        r"burn-mlx|"
        # OTel / mocks / encodings infra
        r"mock-collector|xdp|"
        # narrow utility helpers / single-concept (insufficient API depth)
        r"id_newtype|default_is_triple_underscore|umbra|among|asbytes|"
        r"varing|conflate|composable|accepts|state-machines|"
        r"arcshift|clear-cache|keybinds|clap-sort|"
        # loggers (single purpose)
        r"urlogger|fancy-log|rat_logger|treelog|"
        # path / fs / test helpers
        r"soft-canonicalize|workspace_tools|app-path|random-dir|"
        r"base32-fs|extended-notify|"
        r"snapshot-testing|test-fork|libtest-mimic-collect|"
        # narrow encodings / data formats
        r"trit-vsa|oct|minstd|woofwoof|jsonrepair|"
        # narrow runtime helpers
        r"typed-env|ansic|smol-timeout2|fork_union|"
        r"fast-down|deko|ewf|inquire-clack|frontmatter-gen|tx2-link|"
        # narrow CLI / config helpers
        r"facet-args|figue|"
        # numerical computing duplicates with numrs2
        r"hisab|sciforge|oxifft|glamx|fastnum|approxim|inner-space|spirix|rmt|"
        # ML / clustering / stats / optim (already covered also via desc, but be sure)
        r"clump|kuji|textprep|logp|wass|dendritic|graphops|postings|sbits|"
        # audio / image / signal binary-output
        r"opus-decoder|zaft|fast-ssim2|maolan-widgets|plotters-iced2|"
        r"chess-corners|"
        # rejected earlier
        r"fluent-i18n|octofhir-ucum|rumdl|ez-ffmpeg"
        r")$"
    )

    # Repository / org URL blacklist — catches monorepo siblings whose names
    # vary but live under the same GitHub org or repo.
    repo_blacklist_pattern = (
        r"(?i)github\.com/("
        r"arceos-org|"
        r"commonwarexyz|"
        r"ygg-lang|"
        r"greenticai|"
        r"moltenlabs|"
        r"pimalaya|"
        r"doki-land|"
        r"qubit-ltd|"
        r"rustcrypto|"
        r"rcore-os|"
        r"casper-network|"
        r"zarrs|"
        r"denoland|"
        r"pyo3|"
        r"bevyengine|"
        r"oras-project|"
        r"astral-sh|"
        r"paritytech|"
        r"oovm|"
        r"oxidized-mc|"
        r"bluefootlabs|"
        # blockchain / zk orgs
        r"propeller-heads|"
        r"atlas-chain|"
        r"spacesprotocol|"
        r"arcium-hq|"
        r"lit-protocol|"
        r"zkcrypto|"
        # OS / kernel / compiler internals
        r"starry-os|"
        r"nyar-vm|"
        # additional blockchain / sdk-only orgs
        r"anoma|"
        r"miraland-labs|"
        r"exowarexyz|"
        r"pubky|"
        r"daa-hq|"
        r"wp-labs|"
        r"turingworks|"
        r"eth-cscs|"
        # solo author with many small domain-sim crates (quality concern)
        r"maccracken"
        r")/"
    )

    desc_exclude_pattern = (
        r"(?i)("
        r"\b("
        # AI / LLM
        r"ai|llm|llms|gpt|chatgpt|openai|anthropic|claude|gemini|mistral|llama|"
        r"agent|agentic|mcp|model context protocol|"
        r"embedding|embeddings|rag|prompt|inference|"
        # blockchain / consensus / p2p / zk
        r"blockchain|consensus|byzantine|substrate|polkadot|ethereum|solana|"
        r"p2p|peer-to-peer|zk|zk-snark|zero-knowledge|post-quantum|"
        # low-level OS
        r"kernel|syscall|firmware|bootloader|hypervisor|"
        # network / transport
        r"http|https|tls|ssl|websocket|grpc|"
        # numerical / math (overlap with numrs2)
        r"matrix|fft|fourier|"
        # audio / image / signal (binary outputs)
        r"audio|synthesis|oscillator|midi|dsp|waveform|"
        r"image|pixel|raster|color|"
        # ML / clustering / optimization
        r"clustering|regression|classification|"
        r"optimizer|optimization|gradient|"
        # physics / domain simulations
        r"simulation|thermodynamic|fluid|optics|chemistry|acoustic|"
        # other infra signals
        r"rcore"
        r")\b"
        r"|"
        # multi-word phrases (no \b around the alternation)
        r"procedural macro|proc[- ]macro|derive macro|"
        r"rest api|http client|api wrapper|web framework|"
        r"on-chain|smart contract|proof-of-work|proof of work|"
        r"deep learning|neural network|"
        r"linear algebra|scientific computing|color management|"
        r"information theory|signal processing|image processing|"
        r"random number"
        r")"
    )

    crate_created_cutoff = datetime(2025, 2, 1, tzinfo=timezone.utc)

    df_quality = df_crates_new_repos.filter(
        pl.col("created_at") >= crate_created_cutoff,
        pl.col("description").str.len_chars() > 10,
        (pl.col("documentation").is_not_null()) & (pl.col("homepage").is_not_null()),
        pl.col("readme").str.len_chars() > 100,
        ~pl.col("id").is_in(
            internal_dep_ids
        ),  # Filter out internal dependencies -> Keep "Frontends"
        ~pl.col("name").str.contains(name_exclude_pattern),
        ~pl.col("name").str.contains(name_blacklist_pattern),
        ~pl.col("repository").str.contains(repo_blacklist_pattern),
        ~pl.col("description").str.contains(desc_exclude_pattern),
    )

    # 4. Join with Downloads and Dependant counts
    print("Joining metrics...")
    q_joined = (
        df_quality.lazy()
        .join(df_downloads.lazy(), left_on="id", right_on="crate_id", how="left")
        .join(direct_dependants.lazy(), left_on="id", right_on="crate_id", how="left")
        .fill_null(0)
    )

    # 5. Result Selection and Sorting
    print("Finalizing results...")
    q_final = (
        q_joined.sort("direct_dependant_count", descending=True)
        .select(
            [
                "name",
                "created_at",
                "direct_dependant_count",
                "downloads",
                "description",
                "repository",
                "homepage",
                "documentation",
            ]
        )
        .head(100)
    )

    print("--- Top 100 Quality Frontend New Crates (Sorted by Direct Dependants) ---")
    with pl.Config(fmt_str_lengths=120, tbl_rows=110, tbl_width_chars=240):
        print(q_final.collect())


if __name__ == "__main__":
    main()
