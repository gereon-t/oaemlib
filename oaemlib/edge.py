from dataclasses import dataclass, field
from functools import cached_property

import numpy as np


@dataclass
class Edge:
    """
    Represents an edge between two points in 3D space.
    Provides methods for calculating azimuth, elevation, and distance.
    """

    start: np.ndarray
    end: np.ndarray
    pos: np.ndarray = field(
        init=False, repr=False, default_factory=lambda: np.zeros(3, dtype=np.float64)
    )

    @cached_property
    def azimuth_start(self) -> float:
        return np.arctan2(self.start[0] - self.pos[0], self.start[1] - self.pos[1])

    @cached_property
    def azimuth_end(self) -> float:
        return np.arctan2(self.end[0] - self.pos[0], self.end[1] - self.pos[1])

    @cached_property
    def edge_vec(self) -> np.ndarray:
        return self.end - self.start

    @cached_property
    def edge_vec_norm(self) -> np.float64:
        return np.linalg.norm(self.edge_vec)

    @cached_property
    def pos_vec(self) -> np.ndarray:
        return self.start - self.pos

    @cached_property
    def pos_vec_norm(self) -> np.float64:
        return np.linalg.norm(self.pos_vec)

    def set_position(self, pos: np.ndarray) -> None:
        self.pos = pos

    def _interpolate_to_angle(self, angle: float) -> np.ndarray:
        beta = np.arctan(
            np.dot(self.edge_vec, self.pos_vec)
            / (self.edge_vec_norm * self.pos_vec_norm)
        )
        gamma = np.pi - (abs(beta) + abs(angle))
        c = self.pos_vec_norm * np.sin(angle) / np.sin(gamma)

        return self.start + c * self.edge_vec / self.edge_vec_norm

    def _distance_at_azimuth(self, azimuth: float) -> np.ndarray:
        # line intersection
        line_of_sight = np.array([np.sin(azimuth), np.cos(azimuth), 0])
        a = np.array(
            [[line_of_sight[0], self.edge_vec[0]], [line_of_sight[1], self.edge_vec[1]]]
        )
        b = self.start[:2] - self.pos[:2]
        x = np.linalg.solve(a, b)
        return x[0]

    def get_elevation(self, azimuth: float) -> np.ndarray:
        """
        Calculates the elevation angle between the edge and the line of sight at a given azimuth angle.

        Args:
            azimuth: The azimuth angle in radians.

        Returns:
            The elevation angle in radians.
        """
        distance = self._distance_at_azimuth(azimuth=azimuth)
        return np.arctan2(self.start[2] - self.pos[2], distance)
