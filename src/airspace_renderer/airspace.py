import logging
import re
import typing
from typing import Dict, List, Protocol, Tuple, runtime_checkable

import shapely

from airspace_renderer.country_borders import get_border_segment
from airspace_renderer.curved_geometries import arc_around_point, circle_around_point
from airspace_renderer.dms_to_decimal import (
    dms_match_to_point,
    dms_string_to_point,
    is_valid_dms_format,
)

__all__ = [
    "BorderProvider",
    "InputGeometryType",
    "render_polygon",
]


log = logging.getLogger(__name__)


@runtime_checkable
class BorderProvider(Protocol):
    def get_border(self, border_name: str) -> shapely.LineString: ...


class InputGeometry(Protocol):
    @classmethod
    def matches(cls, definition: str) -> re.Match | None: ...

    @classmethod
    def parse(
        cls,
        match: re.Match,
        previous: Tuple[float, float] | None,
        subsequent: Tuple[float, float] | None,
        border_provider: BorderProvider,
    ) -> List[Tuple[float, float]]: ...

    @classmethod
    def can_process(
        cls,
        previous: Tuple[float, float] | None,
        subsequent: Tuple[float, float] | None,
    ) -> bool: ...


class VertexInputGeometry:
    @classmethod
    def matches(cls, definition: str) -> re.Match | None:
        return is_valid_dms_format(definition)

    @classmethod
    def parse(
        cls,
        match: re.Match,
        previous: Tuple[float, float] | None,
        subsequent: Tuple[float, float] | None,
        border_provider: BorderProvider,
    ) -> List[Tuple[float, float]]:
        return [dms_match_to_point(match)]

    @classmethod
    def can_process(
        cls,
        previous: Tuple[float, float] | None,
        subsequent: Tuple[float, float] | None,
    ) -> bool:
        return True


class ArcInputGeometry:
    REGEX = re.compile(
        r"^ARC\((?P<center>[\d NSEW/.]+), (?P<radiusNm>[\d.]+), (?P<direction>cw|ccw)\)$"
    )

    @classmethod
    def matches(cls, definition: str) -> re.Match | None:
        return cls.REGEX.fullmatch(definition)

    @classmethod
    def parse(
        cls,
        match: re.Match,
        previous: Tuple[float, float] | None,
        subsequent: Tuple[float, float] | None,
        border_provider: BorderProvider,
    ) -> List[Tuple[float, float]]:
        assert previous is not None and subsequent is not None

        center = dms_string_to_point(match["center"])
        radius_nm = float(match["radiusNm"])
        direction = match["direction"]
        return arc_around_point(center, previous, subsequent, radius_nm, direction, 100)

    @classmethod
    def can_process(
        cls,
        previous: Tuple[float, float] | None,
        subsequent: Tuple[float, float] | None,
    ) -> bool:
        return previous is not None and subsequent is not None


class CircleInputGeometry:
    REGEX = re.compile(r"^CIRCLE\((?P<center>[\d NSEW/]+), (?P<radiusNm>[\d.]+)\)$")

    @classmethod
    def matches(cls, definition: str) -> re.Match | None:
        return cls.REGEX.fullmatch(definition)

    @classmethod
    def parse(
        cls,
        match: re.Match,
        previous: Tuple[float, float] | None,
        subsequent: Tuple[float, float] | None,
        border_provider: BorderProvider,
    ) -> List[Tuple[float, float]]:
        center = dms_string_to_point(match["center"])
        radius_nm = float(match["radiusNm"])
        return circle_around_point(center, radius_nm, 100)

    @classmethod
    def can_process(
        cls,
        previous: Tuple[float, float] | None,
        subsequent: Tuple[float, float] | None,
    ) -> bool:
        return True


class BorderInputGeometry:
    REGEX = re.compile(r"^BORDER\((?P<borderName>[A-Z+]+)(?P<invert>, I)?\)$")

    @classmethod
    def matches(cls, definition: str) -> re.Match | None:
        return cls.REGEX.fullmatch(definition)

    @classmethod
    def parse(
        cls,
        match: re.Match,
        previous: Tuple[float, float] | None,
        subsequent: Tuple[float, float] | None,
        border_provider: BorderProvider,
    ) -> List[Tuple[float, float]]:
        assert previous is not None and subsequent is not None

        border = border_provider.get_border(match["borderName"])
        invert = match["invert"] is not None

        border_segment = get_border_segment(
            shapely.Point(previous),
            shapely.Point(subsequent),
            border,
            invert,
        )
        if border_segment is None:
            return []

        return [(c[0], c[1]) for c in border_segment.coords]

    @classmethod
    def can_process(
        cls,
        previous: Tuple[float, float] | None,
        subsequent: Tuple[float, float] | None,
    ) -> bool:
        return previous is not None and subsequent is not None


type Vertex = Tuple[float, float]
type InputGeometryType = typing.Literal['vertex'] | typing.Literal['circle'] | typing.Literal['arc'] | typing.Literal['border']

DEFAULT_INPUT_GEOMETRY_TYPES: Dict[InputGeometryType, InputGeometry] = {
    "vertex": VertexInputGeometry,
    "circle": CircleInputGeometry,
    "arc": ArcInputGeometry,
    "border": BorderInputGeometry,
}


def render_polygon(
    geometry: str,
    border_provider: BorderProvider,
    input_geometry_overrides: Dict[InputGeometryType, InputGeometry] | None = None,
) -> Tuple[shapely.Polygon, List[InputGeometryType]]:
    log.debug(f'parsing geometry "{geometry}"')
    geometry_components: List[str] = geometry.split(" - ")
    input_geometry_types: List[InputGeometryType | None] = [
        None for _ in range(len(geometry_components))
    ]
    vertices: List[List[Vertex] | None] = [
        None for _ in range(len(geometry_components))
    ]
    component_indices_to_process = list(range(len(geometry_components)))
    while len(component_indices_to_process):
        log.debug(
            f"{len(component_indices_to_process)} components remaining to be processed"
        )
        len_before_iteration = len(component_indices_to_process)
        for i in component_indices_to_process:
            component = geometry_components[i]
            previous_vertices = vertices[(i - 1) % len(geometry_components)]
            previous_vertex = previous_vertices[-1] if previous_vertices else None
            subsequent_vertices = vertices[(i + 1) % len(geometry_components)]
            subsequent_vertex = subsequent_vertices[0] if subsequent_vertices else None
            input_geometry_type, component_vertices = _parse_polygon_component(
                component,
                previous_vertex,
                subsequent_vertex,
                border_provider,
                input_geometry_overrides,
            )
            if component_vertices is not None:
                assert input_geometry_type is not None
                component_indices_to_process.remove(i)
                input_geometry_types[i] = input_geometry_type
                vertices[i] = component_vertices
        len_after_iteration = len(component_indices_to_process)
        if len_before_iteration == len_after_iteration:
            raise RuntimeError(
                f"no progress made with {len_before_iteration} components remaining"
            )
    flattened_vertices, flattened_input_geometry_types = _flatten_vertices(vertices, input_geometry_types)
    return shapely.Polygon(shell=flattened_vertices), flattened_input_geometry_types


def _parse_polygon_component(
    component: str,
    previous: Tuple[float, float] | None,
    subsequent: Tuple[float, float] | None,
    border_provider: BorderProvider,
    input_geometry_overrides: Dict[InputGeometryType, InputGeometry] | None,
) -> Tuple[InputGeometryType, List[Tuple[float, float]]] | Tuple[None, None]:
    log.debug(
        f'parsing component "{component}" with previous "{previous}" and subsequent "{subsequent}"'
    )
    for input_type_name, input_type in _get_input_geometry_types(
        input_geometry_overrides
    ).items():
        if match := input_type.matches(component):
            if not input_type.can_process(previous, subsequent):
                log.debug(
                    f"skipping processing of component {component} because it cannot be processed"
                )
                return None, None
            return input_type_name, input_type.parse(
                match, previous, subsequent, border_provider
            )
    raise ValueError(
        f'no input geometry type matches component "{component}", skipping'
    )


def _flatten_vertices(segments: List[List[Vertex] | None], input_geometry_types: List[InputGeometryType | None]):
    vertices: List[Vertex] = []
    vertex_input_geometry_types: List[InputGeometryType] = []
    for i, s in enumerate(segments):
        if not s:
            continue
        input_geometry_type = input_geometry_types[i]
        assert input_geometry_type is not None
        vertices.extend(s)
        vertex_input_geometry_types.extend([input_geometry_type for _ in range(len(s))])
    assert len(vertices) == len(vertex_input_geometry_types), f"{len(vertices)} != {len(vertex_input_geometry_types)}"
    return vertices, vertex_input_geometry_types


def _get_input_geometry_types(
    input_geometry_overrides: Dict[InputGeometryType, InputGeometry] | None,
) -> Dict[InputGeometryType, InputGeometry]:
    if not input_geometry_overrides:
        return DEFAULT_INPUT_GEOMETRY_TYPES
    return {**DEFAULT_INPUT_GEOMETRY_TYPES, **input_geometry_overrides}
