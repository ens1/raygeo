"""Tests for AggregateGroup LinkMode.

- LinkMode::None (default): inputs are concatenated verbatim.
- LinkMode::Sequential: travel moves (retract -> XY travel -> plunge)
  are emitted between consecutive inputs, plus a final lift after
  the last input.
"""

from conftest import (
    aggregate_result,
    collect_completions,
    make_contour_compute,
    make_square_part,
)

from raygeo.cnc.execution.specs import (
    AggregateGroup,
    AggregateInput,
    AggregateSpec,
    LinkMode,
    MachineParams,
    Marker,
)
from raygeo.ops import Ops
from raygeo.ops.types import CommandCategory, CommandType, SectionType
from raygeo.pipeline.request import NodeRequest
from raygeo.pipeline.stage import StageSpec

IDENTITY = [
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]

TRANSLATE_20 = [
    [1.0, 0.0, 0.0, 20.0],
    [0.0, 1.0, 0.0, 20.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _by_key(completed):
    return {c.key: c for c in completed}


def _count_category(ops: Ops, category: CommandCategory) -> int:
    return sum(ops.category(i) == category for i in range(ops.len()))


def _assert_vector_sections(ops: Ops, expected_uids: list[str]) -> None:
    depth = 0
    section_uids = []
    for i in range(ops.len()):
        command = ops.command_type(i)
        if command == CommandType.OPS_SECTION_START:
            assert depth == 0
            section_type, uid, raster_mode = ops.section_params(i)
            assert section_type == SectionType.VECTOR_OUTLINE
            assert raster_mode is None
            section_uids.append(uid)
            depth = 1
        elif command == CommandType.OPS_SECTION_END:
            assert depth == 1
            depth = 0
    assert depth == 0
    assert section_uids == expected_uids


def _moves_outside_sections(ops: Ops) -> list[tuple[float, float, float]]:
    depth = 0
    moves = []
    for i in range(ops.len()):
        command = ops.command_type(i)
        if command == CommandType.OPS_SECTION_START:
            depth += 1
        elif command == CommandType.OPS_SECTION_END:
            depth -= 1
        elif depth == 0 and ops.category(i) == CommandCategory.MOVING:
            moves.append(ops.endpoint(i))
    assert depth == 0
    return moves


def _make_aggregate(
    key: str,
    inputs: list,
    link_mode=None,
    wrap_start=None,
    wrap_end=None,
    machine=None,
    start_markers=None,
    end_markers=None,
) -> NodeRequest:
    return NodeRequest(
        key=key,
        generation_id=1,
        stage=StageSpec.Aggregate(
            spec=AggregateSpec(
                wrap_start=wrap_start or [],
                groups=[
                    AggregateGroup(
                        start_markers=start_markers or [],
                        inputs=inputs,
                        end_markers=end_markers or [],
                        link_mode=link_mode or LinkMode.none(),
                    )
                ],
                wrap_end=wrap_end or [],
                machine=machine or MachineParams(),
            )
        ),
    )


# ── LinkMode.none (default, no linking) ──────────────────────────


def test_link_mode_none_concatenates_without_travel():
    """No travel moves when link_mode is None."""
    a = make_contour_compute("a", workpiece_uid="a")
    b = make_contour_compute("b", workpiece_uid="b")
    agg = _make_aggregate(
        "agg",
        [
            AggregateInput(
                source_key="a",
                placement_matrix=IDENTITY,
                uid="a",
                target_dimensions=(0.0, 0.0),
            ),
            AggregateInput(
                source_key="b",
                placement_matrix=IDENTITY,
                uid="b",
                target_dimensions=(0.0, 0.0),
            ),
        ],
        link_mode=LinkMode.none(),
    )
    completed, _ = collect_completions([a, b, agg])
    out = aggregate_result(_by_key(completed)["agg"])
    assert _count_category(out.ops, CommandCategory.MOVING) == 10
    assert _count_category(out.ops, CommandCategory.STATE) == 2
    assert _count_category(out.ops, CommandCategory.MARKER) == 4
    _assert_vector_sections(out.ops, ["a", "b"])
    assert _moves_outside_sections(out.ops) == []


# ── LinkMode.sequential — basic ──────────────────────────────────


def test_link_mode_sequential_adds_travel_between_inputs():
    """Travel moves (retract + plunge) appear between two identical
    contour inputs."""
    a = make_contour_compute("a", workpiece_uid="a")
    b = make_contour_compute("b", workpiece_uid="b")
    agg = _make_aggregate(
        "agg",
        [
            AggregateInput(
                source_key="a",
                placement_matrix=IDENTITY,
                uid="a",
                target_dimensions=(0.0, 0.0),
            ),
            AggregateInput(
                source_key="b",
                placement_matrix=IDENTITY,
                uid="b",
                target_dimensions=(0.0, 0.0),
            ),
        ],
        link_mode=LinkMode.sequential(safe_z=2.0),
    )
    completed, _ = collect_completions([a, b, agg])
    out = aggregate_result(_by_key(completed)["agg"])
    assert _count_category(out.ops, CommandCategory.MOVING) == 13
    assert _count_category(out.ops, CommandCategory.STATE) == 2
    assert _count_category(out.ops, CommandCategory.MARKER) == 4
    _assert_vector_sections(out.ops, ["a", "b"])
    assert _moves_outside_sections(out.ops) == [
        (0.0, 0.0, 2.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
    ]


def test_link_mode_sequential_lifts_after_single_input():
    """Even with one input, sequential mode emits a final lift if
    the tool ends below safe_z."""
    a = make_contour_compute("a", workpiece_uid="a")
    agg = _make_aggregate(
        "agg",
        [
            AggregateInput(
                source_key="a",
                placement_matrix=IDENTITY,
                uid="a",
                target_dimensions=(0.0, 0.0),
            ),
        ],
        link_mode=LinkMode.sequential(safe_z=2.0),
    )
    completed, _ = collect_completions([a, agg])
    out = aggregate_result(_by_key(completed)["agg"])
    assert _count_category(out.ops, CommandCategory.MOVING) == 6
    assert _count_category(out.ops, CommandCategory.STATE) == 1
    assert _count_category(out.ops, CommandCategory.MARKER) == 2
    _assert_vector_sections(out.ops, ["a"])
    assert _moves_outside_sections(out.ops) == [(0.0, 0.0, 2.0)]


# ── LinkMode.sequential — XY travel when positions differ ────────


def test_link_mode_sequential_xy_travel_when_positions_differ():
    """XY travel move is emitted when consecutive inputs end/start
    at different XY positions."""
    a = make_contour_compute("a", part=make_square_part(), workpiece_uid="a")
    b = make_contour_compute("b", part=make_square_part(), workpiece_uid="b")
    agg = _make_aggregate(
        "agg",
        [
            AggregateInput(
                source_key="a",
                placement_matrix=IDENTITY,
                uid="a",
                target_dimensions=(0.0, 0.0),
            ),
            AggregateInput(
                source_key="b",
                placement_matrix=TRANSLATE_20,
                uid="b",
                target_dimensions=(0.0, 0.0),
            ),
        ],
        link_mode=LinkMode.sequential(safe_z=2.0),
    )
    completed, _ = collect_completions([a, b, agg])
    out = aggregate_result(_by_key(completed)["agg"])
    assert _count_category(out.ops, CommandCategory.MOVING) == 14
    assert _count_category(out.ops, CommandCategory.STATE) == 2
    assert _count_category(out.ops, CommandCategory.MARKER) == 4
    _assert_vector_sections(out.ops, ["a", "b"])
    assert _moves_outside_sections(out.ops) == [
        (0.0, 0.0, 2.0),
        (20.0, 20.0, 2.0),
        (20.0, 20.0, 0.0),
        (20.0, 20.0, 2.0),
    ]


# ── multiple groups, each with linking ───────────────────────────


def test_link_mode_sequential_per_group():
    """Each AggregateGroup independently applies its own link_mode."""
    a = make_contour_compute("a", workpiece_uid="a")
    b = make_contour_compute("b", workpiece_uid="b")
    c = make_contour_compute("c", workpiece_uid="c")

    agg = NodeRequest(
        key="agg",
        generation_id=1,
        stage=StageSpec.Aggregate(
            spec=AggregateSpec(
                wrap_start=[Marker.JobStart(_tag=True)],
                groups=[
                    AggregateGroup(
                        start_markers=[],
                        inputs=[
                            AggregateInput(
                                source_key="a",
                                placement_matrix=IDENTITY,
                                uid="a",
                                target_dimensions=(0.0, 0.0),
                            ),
                        ],
                        end_markers=[],
                        link_mode=LinkMode.sequential(safe_z=2.0),
                    ),
                    AggregateGroup(
                        start_markers=[],
                        inputs=[
                            AggregateInput(
                                source_key="b",
                                placement_matrix=IDENTITY,
                                uid="b",
                                target_dimensions=(0.0, 0.0),
                            ),
                            AggregateInput(
                                source_key="c",
                                placement_matrix=IDENTITY,
                                uid="c",
                                target_dimensions=(0.0, 0.0),
                            ),
                        ],
                        end_markers=[],
                        link_mode=LinkMode.sequential(safe_z=2.0),
                    ),
                ],
                wrap_end=[Marker.JobEnd(_tag=True)],
                machine=MachineParams(),
            )
        ),
    )
    completed, _ = collect_completions([a, b, c, agg])
    out = aggregate_result(_by_key(completed)["agg"])
    assert _count_category(out.ops, CommandCategory.MOVING) == 19
    assert _count_category(out.ops, CommandCategory.STATE) == 3
    assert _count_category(out.ops, CommandCategory.MARKER) == 8
    _assert_vector_sections(out.ops, ["a", "b", "c"])
    assert _moves_outside_sections(out.ops) == [
        (0.0, 0.0, 2.0),
        (0.0, 0.0, 2.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
    ]


# ── Final lift skipped when already at safe_z ────────────────────


def test_link_mode_sequential_no_final_lift_when_at_safe_z():
    """No redundant final lift when the last tool pose is already at
    or above safe_z."""
    a = make_contour_compute("a", workpiece_uid="a")
    agg = _make_aggregate(
        "agg",
        [
            AggregateInput(
                source_key="a",
                placement_matrix=IDENTITY,
                uid="a",
                target_dimensions=(0.0, 0.0),
            ),
        ],
        link_mode=LinkMode.sequential(safe_z=0.0),
    )
    completed, _ = collect_completions([a, agg])
    out = aggregate_result(_by_key(completed)["agg"])
    assert _count_category(out.ops, CommandCategory.MOVING) == 5
    assert _count_category(out.ops, CommandCategory.STATE) == 1
    assert _count_category(out.ops, CommandCategory.MARKER) == 2
    _assert_vector_sections(out.ops, ["a"])
    assert _moves_outside_sections(out.ops) == []


# ── LinkMode existence / constructor checks ──────────────────────


def test_link_mode_constructors():
    none = LinkMode.none()
    assert none.tag == "none"
    assert none.safe_z == 0.0

    seq = LinkMode.sequential(safe_z=3.5)
    assert seq.tag == "sequential"
    assert seq.safe_z == 3.5
