"""Student-region marker protocol.

A test-gated task is one Rust source file that compiles + runs end-to-end
with assertions. The region the student is asked to implement is delimited
by two comment markers:

    // === STUDENT REGION START ===
    fn solve(...) -> ... { ... }   // student-replaceable body
    // === STUDENT REGION END ===

Generation-time agents author the full file (including the reference body
inside the markers) and submit it as `full_program`. Rollout-time, the
trained student is shown the file with the region content blanked out and
asked to fill the gap; the harness above and below the markers is the
fixed evaluation surface.

This module owns the marker contract and a tiny set of helpers that both
sides of the pipeline call.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Plain Rust comments — invisible to rustc, unique enough that no real
# source file is ever going to contain them by accident. We match on the
# stripped-leading-whitespace form so agents that indent the marker by a
# few spaces still parse cleanly.
STUDENT_REGION_START_MARKER = "// === STUDENT REGION START ==="
STUDENT_REGION_END_MARKER = "// === STUDENT REGION END ==="


class StudentRegionParseError(ValueError):
    """`full_program` did not contain exactly one well-formed pair of
    student-region markers."""


@dataclass(frozen=True)
class StudentRegion:
    """A successfully-parsed split of a full Rust program around the
    student region markers.

    Concatenating `prefix + region_body + suffix` reproduces the original
    `full_program` exactly. Replacing `region_body` with student-supplied
    code yields the program the harness will compile and run.
    """

    full_program: str  # original program, markers included
    prefix: str        # everything up to and including the START marker line
    region_body: str   # what's between the markers (the reference solution
                       # at generation time; the student's code at rollout time)
    suffix: str        # everything from the END marker line onward


def parse_student_region(full_program: str) -> StudentRegion:
    """Split `full_program` around exactly one START/END marker pair.

    Raises `StudentRegionParseError` on any malformedness — multiple
    pairs, missing pair, END before START, etc.
    """
    starts = [
        i
        for i, line in enumerate(full_program.splitlines(keepends=True))
        if line.lstrip().startswith(STUDENT_REGION_START_MARKER)
    ]
    ends = [
        i
        for i, line in enumerate(full_program.splitlines(keepends=True))
        if line.lstrip().startswith(STUDENT_REGION_END_MARKER)
    ]
    if len(starts) != 1 or len(ends) != 1:
        raise StudentRegionParseError(
            f"expected exactly one START and one END marker, "
            f"found starts={starts} ends={ends}."
        )
    s, e = starts[0], ends[0]
    if e <= s:
        raise StudentRegionParseError(
            f"END marker (line {e}) precedes START marker (line {s})."
        )

    lines = full_program.splitlines(keepends=True)
    prefix = "".join(lines[: s + 1])         # includes START line
    region_body = "".join(lines[s + 1 : e])  # everything between markers
    suffix = "".join(lines[e:])              # END marker line onward

    return StudentRegion(
        full_program=full_program,
        prefix=prefix,
        region_body=region_body,
        suffix=suffix,
    )


def compose_with_student_code(region: StudentRegion, student_code: str) -> str:
    """Replace the region body with student-supplied code. The result is
    the program the harness will run."""
    body = student_code.rstrip() + "\n" if student_code.strip() else ""
    return region.prefix + body + region.suffix


def blank_region_for_student_view(region: StudentRegion) -> str:
    """Return the program with the region body replaced by a placeholder
    comment, as shown to the student in the prompt."""
    placeholder = "    // <YOUR IMPLEMENTATION HERE>\n"
    return region.prefix + placeholder + region.suffix
