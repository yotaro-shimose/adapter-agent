import subprocess
from pathlib import Path
from adapter_agent.exam.repository import GitRepository


def setup_test_repos(base_dir: Path) -> tuple[GitRepository, GitRepository]:
    """Create dummy project and library repos for testing."""
    base_dir.mkdir(parents=True, exist_ok=True)

    # Project Repo (Main Exam Subject)
    project_dir = base_dir / "project"
    project_dir.mkdir()
    subprocess.run(["git", "init"], cwd=project_dir, check=True)
    subprocess.run(
        ["git", "config", "user.name", "Test User"], cwd=project_dir, check=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=project_dir, check=True
    )

    (project_dir / "Cargo.toml").write_text(
        '[package]\nname = "test_project"\nversion = "0.1.0"\n[dependencies]\n'
    )
    (project_dir / "src").mkdir()
    (project_dir / "src/main.rs").write_text('fn main() { println!("Hello"); }')

    repo = GitRepository(local_dir=project_dir)
    # Allow pushing to the current branch (needed for tests pushing back to this non-bare repo)
    repo.run_git(["config", "receive.denyCurrentBranch", "ignore"])
    repo.add(".")
    repo.commit("Initial commit")

    # Library Repo (Dependency)
    lib_dir = base_dir / "library"
    lib_dir.mkdir()
    subprocess.run(["git", "init"], cwd=lib_dir, check=True)
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=lib_dir, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=lib_dir, check=True
    )

    (lib_dir / "lib.rs").write_text(
        'pub fn get_greeting_suffix() -> String { "World".to_string() }'
    )

    lib_repo = GitRepository(local_dir=lib_dir)
    lib_repo.add(".")
    lib_repo.commit("Initial lib commit")

    return repo, lib_repo
