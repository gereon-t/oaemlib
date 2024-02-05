import logging
from dataclasses import dataclass

import requests
from pointset import PointSet
from oaemlib.gml import GMLNeighborhood, lod1_buildings_parser, lod2_buildings_parser

logger = logging.getLogger("root")


@dataclass
class WFSSettings:
    url: str
    base_request: str
    epsg: int
    lod: int = 1
    n_range: float = 100.0


def request_gml_neighborhood(position: PointSet, wfs_settings: WFSSettings) -> GMLNeighborhood:
    """
    Sends a request to the WFS server to retrieve the Level of Detail 1 (LOD1) CityGML data
    for the specified position.

    Args:
        position (PointSet): The position to retrieve the data for.
        wfs_settings (WFSSettings): The settings for the WFS server.

    Returns:
        list[GMLBuilding]: A list of GMLBuilding objects representing the retrieved CityGML data.

    Raises:
        requests.RequestException: If the WFS request fails.
    """
    position.to_epsg(wfs_settings.epsg)
    request_url = create_request(pos=position, wfs_settings=wfs_settings)
    logger.debug("Sending request %s", request_url)
    response = requests.get(request_url, timeout=10)
    logger.debug("received answer. Status code: %s", response.status_code)

    if response.status_code != 200:
        raise requests.RequestException("WFS request failed!")

    return parse_response(response, wfs_settings.lod)


def create_request(pos: PointSet, wfs_settings: WFSSettings) -> str:
    """
    Creates a WFS request URL for the specified position and range.

    Args:
        pos (PointSet): The position to retrieve the data for.
        wfs_settings (WFSSettings): The settings for the WFS server.

    Returns:
        str: The WFS request URL.

    """
    n_range = wfs_settings.n_range
    bbox_bl = [pos.x - n_range, pos.y - n_range]
    bbox_tr = [pos.x + n_range, pos.y + n_range]
    bbox = f"BBOX={bbox_bl[0]},{bbox_bl[1]},{bbox_tr[0]},{bbox_tr[1]},urn:ogc:def:crs:EPSG::{wfs_settings.epsg}"
    return f"{wfs_settings.url}?{wfs_settings.base_request}&{bbox}"


def parse_response(response: requests.Response, lod: int = 1) -> GMLNeighborhood:
    """
    Parses the response from the WFS server and returns a list of edges representing
    the buildings.

    Args:
        response (Response): The response object from the WFS server.

    Returns:
        list[GMLBuilding]: A list of GMLBuilding representing the buildings
    """
    logger.debug("parsing response ...")

    if lod == 1:
        gml_buildings = lod1_buildings_parser(response.content)
    elif lod == 2:
        gml_buildings = lod2_buildings_parser(response.content)
    else:
        raise ValueError("LOD must be 1 or 2")

    return GMLNeighborhood(gml_buildings)
