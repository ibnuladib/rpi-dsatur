from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


@dataclass
class ColoringResult:
    success: bool
    coloring: dict[int, int]  # vertex_id -> color, partial if success=False
    failed_vertex: Optional[int] = None  # set when success=False


@dataclass
class ValidityReport:
    is_valid: bool
    num_conflicts: int
    num_list_violations: int
    conflict_edges: list[tuple[int, int]] = field(default_factory=list)
    violating_vertices: list[int] = field(default_factory=list)


class EditType(str, Enum):
    SHRINK_LIST = "shrink_list"
    REMOVE_VERTEX = "remove_vertex"


@dataclass
class Edit:
    index: int
    type: EditType
    vertex: int
    removed_colors: Optional[list[int]] = None  # populated only for SHRINK_LIST


@dataclass
class GraphInstance:
    graph_id: str
    n: int
    seed: int
    edges: list[tuple[int, int]]
    lists: dict[int, list[int]]  # vertex_id -> color list
    k: int
    avg_degree_realized: float


@dataclass
class EditOutcome:
    system: str  # "full_recompute" | "fixed_radius" | "rpi_dsatur"
    graph_id: str
    edit_index: int
    time_ms: float
    vertices_touched: int
    radius_used: int | str  # int, or "full_recompute"
    attempts: int
    fallback_triggered: bool
    valid: bool  # from ValidityReport.is_valid