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
    df_crates_new_repos = df_crates.join(new_repos, on="repository", how="inner")

    print("Analyzing dependencies...")
    version_to_crate = df_versions.select(["id", "crate_id"]).rename(
        {"id": "version_id", "crate_id": "dependant_id"}
    )

    df_deps_enriched = df_deps.lazy().join(
        version_to_crate.lazy(), on="version_id", how="inner"
    )

    direct_dependants = (
        df_deps_enriched.group_by("crate_id")
        .agg(pl.col("dependant_id").n_unique().alias("direct_dependant_count"))
        .collect()
    )

    repo_mapping = df_crates.select(["id", "repository"]).rename(
        {"id": "crate_id", "repository": "repo"}
    )

    internal_deps = (
        df_deps_enriched.join(repo_mapping.lazy(), on="crate_id", how="inner")
        .rename({"repo": "target_repo"})
        .join(
            repo_mapping.lazy(),
            left_on="dependant_id",
            right_on="crate_id",
            how="inner",
        )
        .rename({"repo": "dependant_repo"})
        .filter(pl.col("target_repo") == pl.col("dependant_repo"))
        .select(["crate_id"])
        .unique()
        .collect()
    )

    internal_dep_ids = internal_deps["crate_id"].to_list()

    print("Applying filters (numerical-focused)...")

    # 共通の infrastructure 除外 (sys / derive / macros 等) は維持
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

    # 数値計算ポジティブフィルター: description / readme に numpy 系キーワードを含むもの
    # 数値配列, 行列, テンソル, 線形代数, FFT, 統計, 最適化, 数値計算, 多次元配列 etc.
    numerical_include_pattern = (
        r"(?i)"
        r"(\b("
        r"numpy|ndarray|tensor|tensors|matrix|matrices|vector|vectors|"
        r"linear algebra|linalg|"
        r"numerical|numerics|scientific computing|"
        r"fft|fourier|dft|"
        r"statistics|statistical|stats|"
        r"probability|probabilistic|"
        r"interpolation|interpolate|"
        r"differential equation|ode|pde|"
        r"integration|quadrature|"
        r"optimization|optimizer|optimize|"
        r"regression|"
        r"polynomial|polynomials|"
        r"eigen|svd|qr decomposition|cholesky|lu decomposition|"
        r"sparse matrix|"
        r"convolution|convolve|"
        r"random number|prng|rng|"
        r"distribution|distributions|"
        r"signal processing|"
        r"calculus|"
        r"arbitrary precision|big number|bignum|"
        r"complex number|complex numbers|"
        r"unit conversion|units|"
        r"geometry|geometric|"
        r"computational"
        r")\b)"
    )

    # 数値計算系で意味のないものを軽く除外 (LLM/blockchain/network/etc.)
    desc_exclude_pattern = (
        r"(?i)("
        r"\b("
        r"ai|llm|llms|gpt|chatgpt|openai|anthropic|claude|gemini|mistral|llama|"
        r"agent|agentic|mcp|"
        r"embedding|embeddings|rag|prompt|inference|"
        r"blockchain|consensus|byzantine|substrate|polkadot|ethereum|solana|"
        r"p2p|peer-to-peer|zk|zk-snark|zero-knowledge|"
        r"kernel|syscall|firmware|bootloader|hypervisor|"
        r"http|https|tls|ssl|websocket|grpc|"
        r"audio|midi|"
        r"image|pixel|raster"
        r")\b"
        r"|"
        r"procedural macro|proc[- ]macro|derive macro|"
        r"on-chain|smart contract|"
        r"deep learning|neural network"
        r")"
    )

    crate_created_cutoff = datetime(2025, 2, 1, tzinfo=timezone.utc)

    df_quality = df_crates_new_repos.filter(
        pl.col("created_at") >= crate_created_cutoff,
        pl.col("description").str.len_chars() > 10,
        (pl.col("documentation").is_not_null()) & (pl.col("homepage").is_not_null()),
        pl.col("readme").str.len_chars() > 100,
        ~pl.col("id").is_in(internal_dep_ids),
        ~pl.col("name").str.contains(name_exclude_pattern),
        ~pl.col("description").str.contains(desc_exclude_pattern),
        # numpy ライクな numeric キーワードを description に含む
        pl.col("description").str.contains(numerical_include_pattern),
        # numrs2 自体は除外
        pl.col("name") != "numrs2",
    )

    print("Joining metrics...")
    q_joined = (
        df_quality.lazy()
        .join(df_downloads.lazy(), left_on="id", right_on="crate_id", how="left")
        .join(direct_dependants.lazy(), left_on="id", right_on="crate_id", how="left")
        .fill_null(0)
    )

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
        .head(80)
    )

    print("--- Top 80 Numerical-Computing Crates (Sorted by Direct Dependants) ---")
    with pl.Config(fmt_str_lengths=160, tbl_rows=90, tbl_width_chars=260):
        print(q_final.collect())


if __name__ == "__main__":
    main()
