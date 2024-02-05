import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from intervaltree import Interval, IntervalTree
from pointset import PointSet

from oaemlib.edge import Edge
from oaemlib.gml_provider import GMLProvider
from oaemlib.gml import GMLNeighborhood

logger = logging.getLogger("root")


@dataclass
class Oaem:
    """
    Represents an Obstruction Adaptive Elevation Model (OAEM) that stores the elevation data for a given position in space.
    The OAEM is computed from a set of edges that define the building roof footprints.
    """

    position: PointSet
    res: float
    azimuth: np.ndarray = field(default_factory=lambda: np.array([]))
    elevation: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self) -> None:

        if not self.azimuth.size:
            self.azimuth = np.arange(-np.pi, np.pi, self.res)

        if not self.elevation.size:
            self.elevation = np.zeros_like(np.arange(0, 2 * np.pi, self.res))

        az_idx = np.argsort(self.azimuth)
        self.azimuth = self.azimuth[az_idx]
        self.elevation = self.elevation[az_idx]

    @property
    def az_el_str(self) -> str:
        """
        Returns the OAEM data as a string in azimuth:elevation format.
        """
        return "".join(f"{az:.3f}:{el:.3f}," for az, el in zip(self.azimuth, self.elevation))

    def query(self, azimuth: float) -> float:
        """
        Interpolates the elevation for a given azimuth angle.

        Args:
            azimuth (float): The azimuth angle in radians.

        Returns:
            float: The elevation in radians.
        """
        if azimuth > np.pi:
            azimuth -= 2 * np.pi

        return float(np.interp(azimuth, self.azimuth, self.elevation))


def compute_oaem(
    pos_x: float,
    pos_y: float,
    epsg: int,
    gml_provider: GMLProvider,
    res: float = np.deg2rad(0.5),
    pos_z: Optional[float] = None,
) -> Oaem:
    """
    Computes an Obstruction Adaptive Elevation Model (OAEM) for a given position.

    Args:
        position (PointSet): The query position.
        edge_provider (EdgeProvider): The edge provider to retrieve the building edges from.

    Returns:
        Oaem: An Obstruction Adaptive Elevation Model (OAEM) that stores the elevation data for the given position.
    """
    z_not_provided = pos_z is None

    if z_not_provided:
        pos_z = 0.0

    position = PointSet(xyz=np.array([pos_x, pos_y, pos_z]), epsg=epsg)
    gml_neighborhood = gml_provider.get_neighborhood(position)

    if z_not_provided:
        position.z = np.mean([building.floor_height for building in gml_neighborhood.gml_buildings])
        logger.info("Height not provided. Using mean floor height of %.2f m.", position.z)

    return oaem_from_edge_list(gml_neighborhood=gml_neighborhood, position=position, res=res)


def oaem_from_edge_list(gml_neighborhood: GMLNeighborhood, position: PointSet, res: float) -> Oaem:
    """
    Computes an Obstruction Adaptive Elevation Model (OAEM) for a given position from a list of building edges.

    Args:
        edge_list (list[Edge]): A list of edges that define the building boundaries.
        pos (PointSet): The query position.

    Returns:
        Oaem: An Obstruction Adaptive Elevation Model (OAEM) that stores the elevation data for the given position.
    """
    if not gml_neighborhood:
        return Oaem(position=position, res=res)

    edge_list = gml_neighborhood.edges
    interval_tree = build_interval_tree(edge_list=edge_list, pos=position.xyz.ravel())
    oaem_grid = np.arange(-np.pi, np.pi, res)
    oaem_temp = np.zeros((len(oaem_grid), 2), dtype=np.float64)

    for i, az in enumerate(oaem_grid):
        overlaps: set[Interval] = interval_tree[az]
        oaem_temp[i, 0] = az
        oaem_temp[i, 1] = max(
            0,
            max(
                (edge_list[overlap.data].get_elevation(az) for overlap in overlaps),
                default=0,
            ),
        )
    return Oaem(position=position, azimuth=oaem_temp[:, 0], elevation=oaem_temp[:, 1], res=res)


def build_interval_tree(edge_list: list[Edge], pos: np.ndarray) -> IntervalTree:
    """
    Builds an interval tree from a list of edges that define the building boundaries and a position.

    Args:
        edge_list (list[Edge]): A list of edges that define the building boundaries.
        pos (np.ndarray): The position as a numpy array of shape (3,) in the format (x, y, z).

    Returns:
        IntervalTree: An interval tree that stores the intervals of azimuth angles that intersect with the edges.
    """

    def add_to_interval_tree(interval_tree: IntervalTree, start: float, end: float, data: float) -> None:
        if start == end:
            return
        interval_tree.addi(start, end, data)

    interval_tree = IntervalTree()

    for i, edge in enumerate(edge_list):
        edge.set_position(pos=pos)
        az_1 = np.arctan2(edge.start[0] - pos[0], edge.start[1] - pos[1])
        az_2 = np.arctan2(edge.end[0] - pos[0], edge.end[1] - pos[1])
        if np.sign(az_1) != np.sign(az_2) and np.abs(az_1 - az_2) > np.pi:
            add_to_interval_tree(interval_tree=interval_tree, start=-np.pi, end=min(az_1, az_2), data=i)
            add_to_interval_tree(interval_tree=interval_tree, start=max(az_1, az_2), end=np.pi, data=i)
        else:
            add_to_interval_tree(
                interval_tree=interval_tree,
                start=min(az_1, az_2),
                end=max(az_1, az_2),
                data=i,
            )

    return interval_tree
