from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pointset import PointSet
from pvlib import solarposition

from oaemlib.oaem import Oaem


@dataclass
class SunTrack:
    """
    Represents a sun track for a given position.
    """

    pos_x: float
    pos_y: float
    pos_z: float
    epsg: int = 25832
    date: datetime = field(default_factory=lambda: datetime.now().astimezone())
    vis_changes: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.pos = PointSet(
            xyz=np.array([self.pos_x, self.pos_y, self.pos_z]),
            epsg=self.epsg,
        )
        self.pos.to_epsg(4326)

    def get_sun_track(
        self,
        date: datetime,
        freq: timedelta = timedelta(minutes=1),
        daylight_only: bool = False,
    ) -> np.ndarray:
        start_time = datetime.combine(date, datetime.min.time())
        end_time = datetime.combine(date, datetime.max.time())

        times = pd.date_range(
            start_time,
            end_time,
            freq=freq,
            tz=date.tzinfo,
        )

        solpos = solarposition.get_solarposition(
            time=times, latitude=self.pos.x, longitude=self.pos.y, altitude=self.pos.z
        )

        if daylight_only:
            solpos = solpos.loc[solpos["apparent_elevation"] > 0, :]

        return np.c_[
            np.array([time.timestamp() for time in solpos.index], dtype=float),
            np.deg2rad(solpos["azimuth"].to_numpy(dtype=float)),
            np.deg2rad(solpos["apparent_elevation"].to_numpy(dtype=float)),
        ]

    @property
    def current_sunpos(self) -> tuple[float, float]:
        date = self.date
        solpos = solarposition.get_solarposition(
            time=date, latitude=self.pos.x, longitude=self.pos.y, altitude=self.pos.z
        )
        return float(np.deg2rad(solpos["azimuth"].iloc[0])), float(
            np.deg2rad(solpos["apparent_elevation"].iloc[0])
        )

    def intersect_with_oaem(self, oaem: Oaem) -> None:
        date = self.date
        sun_track = self.get_sun_track(date=date)
        oaem_query_func = np.vectorize(oaem.query)
        oaem_elevations = oaem_query_func(sun_track[:, 1])

        vis_idx = np.c_[sun_track[:, 0], sun_track[:, 2] > oaem_elevations]
        changes = np.where(np.abs(np.diff(vis_idx[:, 1])) == 1)[0]
        self.vis_changes = np.c_[vis_idx[changes, 0], vis_idx[changes, 1]]

    @property
    def until(self) -> float | None:
        if len(self.vis_changes) == 0:
            return None

        current_tstamp = self.date.timestamp()
        current_idx = np.searchsorted(self.vis_changes[:, 0], current_tstamp)

        if current_idx == len(self.vis_changes):
            return None

        return self.vis_changes[current_idx, 0]

    @property
    def since(self) -> float | None:
        if len(self.vis_changes) == 0:
            return None

        current_tstamp = self.date.timestamp()
        current_idx = np.searchsorted(self.vis_changes[:, 0], current_tstamp)

        if current_idx == 0:
            return None

        if current_idx == len(self.vis_changes):
            return self.vis_changes[-1, 0]

        return self.vis_changes[current_idx - 1, 0]
