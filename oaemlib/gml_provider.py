import logging
from functools import lru_cache
from typing import Protocol

from pointset import PointSet

from oaemlib.filepicker import FilePicker
from oaemlib.gml import GMLBuilding, GMLNeighborhood, GMLFileList, lod1_buildings_parser, lod2_buildings_parser
from oaemlib.wfs import WFSSettings, request_gml_neighborhood

logger = logging.getLogger("root")


class GMLProvider(Protocol):
    """
    Protocol for gml providers.

    GML providers are used to retrieve GML neighborhoods from a given position.
    They need to implement the get_neighborhood method.
    """

    def get_neighborhood(self, pos: PointSet) -> GMLNeighborhood: ...


class LocalGMLProvider:
    """
    GML provider for local CityGML data.

    This gml provider uses local CityGML data to retrieve buildings from a given position.
    The level of detail (LOD) of the data can be specified as 1 or 2. The path to the corresponding
    LOD1 or LOD2 data needs to be provided.
    """

    def __init__(self, file_picker: FilePicker) -> None:
        self.file_picker = file_picker

    @lru_cache(maxsize=128)
    def _build_gml_neighborhood(self, filepaths: GMLFileList) -> GMLNeighborhood:
        """
        Builds a GMLData object from a list of filepaths.

        The GMLData object is cached to avoid unnecessary parsing of the same file(s).
        """
        buildings: list[GMLBuilding] = []

        for file in filepaths.files:
            with open(file, "r", encoding="utf-8") as f:
                data = f.read()

            if self.file_picker.settings.lod == 1:
                buildings.extend(lod1_buildings_parser(data))
            elif self.file_picker.settings.lod == 2:
                buildings.extend(lod2_buildings_parser(data))
            else:
                raise ValueError("LOD must be 1 or 2")

        return GMLNeighborhood(buildings)

    @lru_cache(maxsize=512)
    def get_neighborhood(self, position: PointSet) -> GMLNeighborhood:
        filepaths = self.file_picker.get_files(position)
        gml_neighborhood = self._build_gml_neighborhood(filepaths)
        return gml_neighborhood.query(position.xyz, n_range=self.file_picker.settings.n_range)


class WFSGMLProvider:
    """
    Edge provider for WFS data.

    This edge provider uses the WFS API to retrieve building edges from a given position.
    By default, the WFS API of North Rhine-Westphalia (NRW), Germany is used. This API
    only provides LOD1 data.
    """

    def __init__(self, wfs_settings: WFSSettings) -> None:
        self.wfs_settings = wfs_settings

    def get_neighborhood(self, position: PointSet) -> GMLNeighborhood:
        return request_gml_neighborhood(position=position, wfs_settings=self.wfs_settings)
