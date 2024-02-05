import logging
from dataclasses import dataclass
import os
from typing import Protocol

import numpy as np
from pointset import PointSet

from oaemlib.gml import GMLFileList

logger = logging.getLogger("root")


@dataclass
class FileSettings:
    data_path: str
    epsg: int
    lod: int
    n_range: float


class FilePicker(Protocol):
    """
    Protocol for file pickers.

    File pickers are used to retrieve gml files from a given position.
    They need to implement the get_files method.
    """

    settings: FileSettings

    def get_files(self, position: PointSet) -> GMLFileList:
        """
        Returns a list of gml files that are relevant for the given position.

        Args:
            position (PointSet): Position for which the gml files are needed.

        Returns:
            GMLFileList: List of relevant gml files.
        """


@dataclass
class AllFilePicker(FilePicker):
    settings: FileSettings

    def get_files(self, position: PointSet) -> GMLFileList:
        """
        Returns a list of all gml files that are in the data path.

        Args:
            position (PointSet): Position for which the gml files are needed.

        Returns:
            GMLFileList: List of gml files.
        """
        if len(position) != 1:
            raise ValueError("Only one position is allowed")

        files = os.listdir(self.settings.data_path)

        file_list = GMLFileList(data_path=self.settings.data_path)

        for file in files:
            if file.endswith(".gml"):
                file_list.add(file)

        return file_list


@dataclass
class NRWFilePicker:
    settings: FileSettings

    def get_files(self, position: PointSet) -> GMLFileList:
        """
        Returns a list of gml files that are relevant for the given position.

        Args:
            position (PointSet): Position for which the gml files are needed.

        Returns:
            GMLFileList: List of relevant gml files.
        """
        if len(position) != 1:
            raise ValueError("Only one position is allowed")

        position.to_epsg(self.settings.epsg)

        file_list = GMLFileList(data_path=self.settings.data_path)
        x_val = int(np.floor(position.x / 1000))
        y_val = int(np.floor(position.y / 1000))

        file_list.add(f"LoD{self.settings.lod}_32_{x_val}_{y_val}_1_NW.gml")

        if (position.x - x_val * 1000) < self.settings.n_range:
            file_list.add(f"LoD{self.settings.lod}_32_{x_val-1}_{y_val}_1_NW.gml")

        if (position.y - y_val * 1000) < self.settings.n_range:
            file_list.add(f"LoD{self.settings.lod}_32_{x_val}_{y_val-1}_1_NW.gml")

        if (position.x - x_val * 1000) > (1000 - self.settings.n_range):
            file_list.add(f"LoD{self.settings.lod}_32_{x_val+1}_{y_val}_1_NW.gml")

        if (position.y - y_val * 1000) > (1000 - self.settings.n_range):
            file_list.add(f"LoD{self.settings.lod}_32_{x_val}_{y_val+1}_1_NW.gml")

        logger.info("Using gml files: %s", file_list.files)

        return file_list
