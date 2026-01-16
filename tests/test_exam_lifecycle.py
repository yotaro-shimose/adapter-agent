from oai_utils.agent import AgentsSDKModel
from pathlib import Path
import pytest
from adapter_agent.exam.creator import create_exam
from adapter_agent.exam.runner import solve_exam, evaluate_exam
from adapter_agent.exam.topic import Topic


from tests.utils import setup_test_repos


@pytest.mark.asyncio
async def test_create_exam_live(model: AgentsSDKModel, tmp_path: Path):
    """Live test of create_exam with real agent execution."""
    project_repo, lib_repo = setup_test_repos(tmp_path / "test_repos")

    # A simple topic that requires using the library
    topic = Topic(
        title="Return True",
        description=(
            "Modify the function `solution` in `src/solution.rs` to return `true`.\n"
            "The function signature should be `pub fn solution() -> bool`."
        ),
    )

    # Execute create_exam (this runs the real agent)
    # We expect this to take some time and consume tokens
    exam = await create_exam(model, project_repo, lib_repo, topic)

    # Basic Validations
    print(f"Exam ID: {exam.id}")
    print(f"Question: {exam.question}")

    assert "Return True" in exam.question or "true" in exam.question.lower()
    assert len(exam.eval_rubric) > 10
    assert exam.problem_commit != exam.solution_commit

    # Verify Git History Relationship
    repo = exam.project
    # Solution commit should be an ancestor of Problem commit (Base -> Solution -> Problem)
    try:
        repo.run_git(
            ["merge-base", "--is-ancestor", exam.solution_commit, exam.problem_commit]
        )
    except Exception:
        pytest.fail("Solution commit is NOT an ancestor of Problem commit")

    # Verify Content in Problem State (HEAD)
    # HEAD should be the Problem Commit (most recent)
    head_files = repo.run_git(["ls-tree", "-r", "HEAD", "--name-only"]).splitlines()
    assert "src/solution.rs" in head_files
    # Check content contains stubs (Problem State)
    prob_content = (repo.local_dir / "src/solution.rs").read_text()
    # Should be stubbed (returning false) or empty (todo!)
    # We check for presence of a failing condition rather than absence of "true"
    # because comments might contain "true".
    print(f"Problem Content:\n{prob_content}")
    assert "false" in prob_content or "todo!" in prob_content

    # Verify Content in Solution State
    # Checkout solution commit and check files
    repo.run_git(["checkout", exam.solution_commit])
    sol_content = (repo.local_dir / "src/solution.rs").read_text()
    assert "todo!" not in sol_content
    assert "pub fn solution()" in sol_content

    # Verify Push to Original Repo
    branch_name = f"exam-{exam.id}"
    branches = project_repo.run_git(["branch", "--list", branch_name])
    assert branch_name in branches

    # Verify we can checkout the commits in the original repo
    project_repo.run_git(["checkout", branch_name])
    head_commit = project_repo.run_git(["rev-parse", "HEAD"])
    # The branch tip should be the Problem Commit
    assert head_commit.strip() == exam.problem_commit.strip()

    # --- Test solve_exam ---
    print("\nTesting solve_exam...")
    solution_path = await solve_exam(model, exam)

    # Check if solution exists in the new workspace
    print(f"Solution Path: {solution_path}")
    new_sol_content = (solution_path / "src/solution.rs").read_text()
    print(f"Solution Content:\n{new_sol_content}")

    # Assert that the agent actually wrote the solution again
    assert "true" in new_sol_content

    # --- Test evaluate_exam ---
    print("\nTesting evaluate_exam...")
    report = await evaluate_exam(model, exam, solution_path)
    print(f"Evaluation Report:\n{report}")

    # Check for score report
    assert "TOTAL USER SCORE" in report or "points" in report.lower()
