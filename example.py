from oaemlib import (
    WFSSettings,
    SunTrack,
    WFSGMLProvider,
    LocalGMLProvider,
    Oaem,
    NRWFilePicker,
    FileSettings,
    compute_oaem,
)
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np

import logging

logging.basicConfig(level=logging.INFO)


def plot(oaem: Oaem, suntrack: SunTrack) -> None:
    track = suntrack.get_sun_track(suntrack.date, freq=timedelta(minutes=1))

    _, ax = plt.subplots(subplot_kw={"projection": "polar"})
    plt.title(f"OAEM for {oaem.position.x:.2f}, {oaem.position.y:.2f}, {oaem.position.z:.2f}")
    ax.plot(oaem.azimuth, np.pi / 2 - oaem.elevation, color="blue", label="OAEM")
    ax.plot(track[:, 1], np.pi / 2 - track[:, 2], color="yellow", label="Sun")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlim(0, np.pi / 2)
    plt.legend()


def oaem_using_wfs(pos_x: float, pos_y: float, pos_z: float, epsg: int, wfs_settings: WFSSettings) -> Oaem:

    edge_provider = WFSGMLProvider(wfs_settings=wfs_settings)

    return compute_oaem(pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, epsg=epsg, gml_provider=edge_provider)


def oaem_using_local_gml(
    pos_x: float,
    pos_y: float,
    pos_z: float,
    epsg: int,
    file_settings: FileSettings,
) -> Oaem:
    nrw_file_picker = NRWFilePicker(file_settings)
    edge_provider = LocalGMLProvider(file_picker=nrw_file_picker)

    return compute_oaem(pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, epsg=epsg, gml_provider=edge_provider)


def intersect_sun_track(sun_track: SunTrack, oaem: Oaem) -> SunTrack:

    sun_track.intersect_with_oaem(oaem)

    if sun_track.until is not None and sun_track.since is not None:
        sun_duration = timedelta(seconds=sun_track.until - sun_track.since)

        print(
            f"Sun is visible for {sun_duration} from {datetime.fromtimestamp(sun_track.since)} to {datetime.fromtimestamp(sun_track.until)}"
        )
    else:
        print("Sun is not visible")


def main():
    pos_x = 364862
    pos_y = 5621215
    pos_z = 61.1
    epsg = 25832

    date = datetime.strptime("2021-06-21 12:00:00", "%Y-%m-%d %H:%M:%S")
    gml_source = "local"

    if gml_source == "wfs":
        print("Using WFS")
        wfs_settings = WFSSettings(
            url="https://www.wfs.nrw.de/geobasis/wfs_nw_3d-gebaeudemodell_lod1",
            base_request="Service=WFS&REQUEST=GetFeature&VERSION=1.1.0&TYPENAME=bldg:Building",
            epsg=25832,
            n_range=100.0,
            lod=1,
        )
        oaem = oaem_using_wfs(pos_x, pos_y, pos_z, epsg, wfs_settings)
    elif gml_source == "local":
        print("Using local GML")
        file_settings = FileSettings(
            data_path="./gmldata",
            epsg=25832,
            lod=2,
            n_range=100.0,
        )
        oaem = oaem_using_local_gml(pos_x, pos_y, pos_z, epsg, file_settings)
    else:
        raise ValueError(f"Unknown gml source: {gml_source}")

    suntrack = SunTrack(pos_x=pos_x, pos_y=pos_y, pos_z=pos_z, epsg=epsg, date=date)
    intersect_sun_track(suntrack, oaem)
    plot(oaem, suntrack)
    plt.show()


if __name__ == "__main__":
    main()
