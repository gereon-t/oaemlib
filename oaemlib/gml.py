import os
from dataclasses import dataclass, field
from functools import cached_property
from typing import TypeAlias

import numpy as np
import xmltodict
from scipy.spatial import KDTree

from oaemlib.edge import Edge

CoordinateList: TypeAlias = list[list[float]]


@dataclass
class GMLBuilding:
    floor_height: float
    center_of_mass: np.ndarray
    building_coordinates: CoordinateList

    @property
    def edges(self) -> list[Edge]:
        return [
            Edge(start=np.array(edge_coord[:3]), end=np.array(edge_coord[3:]))
            for edge_coord in self.building_coordinates
        ]


class GMLNeighborhood:
    def __init__(self, gml_buildings: list[GMLBuilding]) -> None:
        self.gml_buildings = gml_buildings

    def __len__(self) -> int:
        return len(self.gml_buildings)

    def __bool__(self) -> bool:
        return bool(self.gml_buildings)

    @cached_property
    def kdtree(self) -> KDTree:
        coordinates = np.array([building.center_of_mass for building in self.gml_buildings])
        return KDTree(coordinates[:, :2])

    @cached_property
    def edges(self) -> list[Edge]:
        return [edge for building in self.gml_buildings for edge in building.edges]

    def query(self, pos: np.ndarray, n_range: float = 100.0) -> "GMLNeighborhood":
        if not self.gml_buildings:
            return []

        query_indices = self.kdtree.query_ball_point(pos[:, :2].flatten(), r=n_range)
        return GMLNeighborhood([self.gml_buildings[index] for index in query_indices])

    @classmethod
    def from_gml(cls, gml: str, lod: int = 2) -> "GMLNeighborhood":
        return cls(lod2_buildings_parser(gml)) if lod == 2 else cls(lod1_buildings_parser(gml))


@dataclass
class GMLFileList:
    """
    Hashable list of gml files.

    This class is used to cache the gml files for a given position.
    """

    data_path: str
    files: list[str] = field(default_factory=list)

    def add(self, file: str) -> None:
        self.files.append(os.path.join(self.data_path, file))

    def __hash__(self) -> int:
        return "".join(sorted(self.files)).__hash__()


def _handle_surface_member(surface_member: dict) -> CoordinateList:
    polygon: dict | str = (
        surface_member.get("gml:Polygon", {}).get("gml:exterior", {}).get("gml:LinearRing", {}).get("gml:posList", {})
    )

    if isinstance(polygon, dict):
        polygon = str(polygon.get("#text", ""))

    if not polygon:
        return []

    coords = [float(c) for c in polygon.split(" ")]

    if len(coords) % 3 != 0:
        return []

    if len(coords) < 6:
        return []

    return [coords[i : i + 6] for i in range(0, len(coords) - 3, 3)]


def _extract_surface_members(surface: dict) -> CoordinateList:
    surface_members: list | dict = (
        surface.get("bldg:lod2MultiSurface", {}).get("gml:MultiSurface", {}).get("gml:surfaceMember", [])
    )

    if not surface_members:
        return []

    coords: CoordinateList = []

    if isinstance(surface_members, list):
        for surface_member in surface_members:
            coords.extend(_handle_surface_member(surface_member))
    else:
        coords.extend(_handle_surface_member(surface_members))

    return coords


def _parse_building_bounds(bounds: list[dict]) -> CoordinateList:
    building_coordinates: CoordinateList = []
    for surfaces in bounds:
        for surface in surfaces.values():
            building_coordinates.extend(_extract_surface_members(surface))

    return building_coordinates


def lod1_buildings_parser(gml: str) -> list[GMLBuilding]:
    data = xmltodict.parse(gml)
    cityobject_members = data.get("core:CityModel", {}).get("core:cityObjectMember", {})

    gml_buildings: list[GMLBuilding] = []
    if not cityobject_members:
        return gml_buildings

    if not isinstance(cityobject_members, list):
        cityobject_members = [cityobject_members]

    for cobj in cityobject_members:
        bldg = cobj.get("bldg:Building", {})

        # single lod1Solid
        if lod1solid := bldg.get("bldg:lod1Solid", {}):
            gml_buildings.append(_parse_lod1_building(lod1solid))

        # multiple lod1Solids
        if bldg_parts := bldg.get("bldg:consistsOfBuildingPart", {}):
            for bpart in bldg_parts:
                lod1solid = bpart.get("bldg:BuildingPart", {}).get("bldg:lod1Solid", {})
                gml_building = _parse_lod1_building(lod1solid)
                gml_buildings.append(gml_building)

    return gml_buildings


def lod2_buildings_parser(gml: str) -> list[GMLBuilding]:
    data = xmltodict.parse(gml)
    gml_buildings: list[GMLBuilding] = []

    if "core:CityModel" not in data:
        return gml_buildings

    if "core:cityObjectMember" not in data["core:CityModel"]:
        return gml_buildings

    buildings = data["core:CityModel"]["core:cityObjectMember"]

    if not isinstance(buildings, list):
        buildings = [buildings]

    for bdata in buildings:
        gml_buildings.append(_parse_lod2_building(bdata))

    return gml_buildings


def _parse_lod1_building(lod1solid: dict) -> GMLBuilding:
    """
    Parses the lod1Solid element of a building

    Args:
        lod1solid (dict): The lod1Solid element of a building.

    Returns:
        list[float]: A list representing the start and end points of the building edges
    """
    surface_members = (
        lod1solid.get("gml:Solid", {})
        .get("gml:exterior", {})
        .get("gml:CompositeSurface", {})
        .get("gml:surfaceMember", {})
    )

    building_coordinates: CoordinateList = []

    if not surface_members:
        return building_coordinates

    for surface in surface_members:
        building_coordinates.extend(_handle_surface_member(surface))

    floor_height = np.min([val[2] for val in building_coordinates])
    center_of_mass = np.mean(
        [[np.mean(e[::3]), np.mean(e[1::3]), np.mean(e[2::3])] for e in building_coordinates], axis=0
    )

    return GMLBuilding(
        floor_height=floor_height, center_of_mass=center_of_mass, building_coordinates=building_coordinates
    )


def _parse_lod2_building(building_data: dict) -> GMLBuilding:
    building_coordinates: CoordinateList = []

    if "bldg:Building" not in building_data:
        return building_coordinates

    building = building_data["bldg:Building"]

    if "bldg:boundedBy" in building:
        bounded_by = [building["bldg:boundedBy"]]
        for bounds in bounded_by:
            building_coordinates.extend(_parse_building_bounds(bounds))

    if "bldg:consistsOfBuildingPart" in building:
        building_parts = building["bldg:consistsOfBuildingPart"]
        for part in building_parts:
            building_part = part["bldg:BuildingPart"]
            bounds = building_part["bldg:boundedBy"]
            building_coordinates.extend(_parse_building_bounds(bounds))

    floor_height = np.min([val[2] for val in building_coordinates])
    center_of_mass = np.mean(
        [[np.mean(e[::3]), np.mean(e[1::3]), np.mean(e[2::3])] for e in building_coordinates], axis=0
    )

    return GMLBuilding(
        floor_height=floor_height, center_of_mass=center_of_mass, building_coordinates=building_coordinates
    )
