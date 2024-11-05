"""
dir=chains_test_$d
Final version of the commander h5 file script.

This code generates 10 h5 files, one for each dirbe band, each containing 285 pid (chunks), one
for each cio file which contains observations for 1 yday. 

Notes:
- Time gaps are filled with BAD_DATA sentinels to keep the tods continuous.
- No polarization data is included. Psi is currently as of 02-05-2023 not included in the cio files,
  but this is on the todo list.

Run as:
    OMP_NUM_THREADS=1 python write_tods_final.py 

MAKE SURE TO CHANGE VERSION NUMBER AND OUTPUT PATH BEFORE RUNNING TO NOT ACCIDENTALLY OVERWRITE EXISTING FILES (OR SET overwrite=False).
"""


from __future__ import annotations

from dataclasses import dataclass
import itertools
from pathlib import Path
from datetime import timedelta
import multiprocessing

import time
from astropy.io import fits
import astropy.units as u
from astropy.coordinates import SkyCoord
import healpy as hp
from matplotlib import pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from astropy.time import Time, TimeDelta
from cosmoglobe.tod_tools import TODLoader
import pickle
from tqdm import tqdm

#from cProfile import Profile
#from pstats import SortKey, Stats

# Path objects
IRAS_OUT_PATH = Path("/mn/stornext/d5/data/duncanwa/IRAS/hdf_files")
IRAS_DATA_PATH = Path("/mn/stornext/d5/data/duncanwa/IRAS/sopobs_data")
IRAS_DETS = []
for i in range(62):
   if (i+1 != 17) & (i+1 != 20) & (i+1 != 36):
       IRAS_DETS.append(i+1)

# system constants
N_PROC = multiprocessing.cpu_count()

ROTATOR = hp.Rotator(coord=["E", "G"])
YDAYS = np.concatenate([np.arange(89345, 89366), np.arange(90001, 90265)])
START_TIME = Time("1981-01-01", format="isot", scale="utc")

# CIO constants
N_CIO_FILES = 285
BAD_DATA_SENTINEL = -16375
TSCAL = 2e-15
SAMP_RATE = 1 / 26
SAMP_RATE_DAYS = SAMP_RATE / (24 * 3600)


@dataclass
class YdayData:
    tods: dict[str, np.ndarray]
    pixels: dict[str, np.ndarray]
    flags: dict[str, np.ndarray]
    time_start: float
    time_stop: float
    sat_pos_start: np.ndarray
    sat_pos_stop: np.ndarray
    earth_pos_start: np.ndarray
    earth_pos_stop: np.ndarray


@dataclass
class CIO:
    tods: list[np.ndarray]
    pixels: list[np.ndarray]
    flags: list[np.ndarray]
    time_start: list[float]
    time_stop: list[float]
    sat_pos_start: list[np.ndarray]
    sat_pos_stop: list[np.ndarray]
    earth_pos_start: list[np.ndarray]
    earth_pos_stop: list[np.ndarray]


def get_cios(yday_data: list[YdayData]) -> dict[str, CIO]:
    output_dict = {}
    tods = []
    pixels = []
    flags = []
    time_starts = []
    time_stops = []
    sat_pos_start = []
    sat_pos_stop = []
    earth_pos_start = []
    earth_pos_stop = []
    for det in IRAS_DETS:
        for yday in yday_data:
            if f'IRAS_{det:02}' in yday.tods.keys():
                tods.append(yday.tods[f"IRAS_{det:02}"])
                pixels.append(yday.pixels[f"IRAS_{det:02}"])
                flags.append(yday.flags[f"IRAS_{det:02}"])
                time_starts.append(yday.time_start)
                time_stops.append(yday.time_stop)
                sat_pos_start.append(yday.earth_pos_start)
                sat_pos_stop.append(yday.earth_pos_stop)
                earth_pos_start.append(yday.earth_pos_start)
                earth_pos_stop.append(yday.earth_pos_stop)
        output_dict[f'IRAS_{det:02}'] =  CIO(tods=tods,
             pixels=pixels,
             flags=flags,
             time_start=time_starts,
             time_stop=time_stops,
             sat_pos_start=sat_pos_start,
             sat_pos_stop=sat_pos_stop,
             earth_pos_start=earth_pos_start,
             earth_pos_stop=earth_pos_stop)
    return output_dict


def get_yday_data(
        sopobs: int, nside_out: int, color_corr: bool, 
) -> list[YdayData]:



    with multiprocessing.Pool(processes=N_PROC) as pool:
        proc_chunks = [
            pool.apply_async(
                get_yday_cio_data,
                args=(sopobs, det, nside_out, color_corr),
            )
            for det in IRAS_DETS
        ]
        return [result.get() for result in proc_chunks if result]


def get_yday_cio_data(
    sopobs: int,
    det: int,
    nside_out: int,
    #planet_interps: dict[str, dict[str, interp1d]],
    color_corr: bool,
) -> YdayData:
    """Function which extracts and reorders the CIO data from one day CIO file."""

    f = f'{IRAS_DATA_PATH}/det_{det:02}/sopobs_{sopobs+1:04}.npy'
    t, lon, lat, flux = np.load(f)
    flag_tot = np.zeros(len(t))
    flag_tot[~np.isfinite(lon)] += 2**0
    flag_tot[~np.isfinite(flux)] += 2**1

    #yday = YDAYS[file_number]

    t = 53826

    # Convert time to MJD
    t = (START_TIME + TimeDelta(t, format="sec", scale="tai")).mjd
    sat_pos_start, earth_pos_start = 0,0
    sat_pos_stop, earth_pos_stop = 0,0


    # Extract tods, and modify pointing vectors per detector according to beam data
    tods: dict[str, np.ndarray] = {}
    pixels: dict[str, np.ndarray] = {}
    flags: dict[str, np.ndarray] = {}
    band_label = f'IRAS_{det:02}'



    goodvals = np.isfinite(lon) & np.isfinite(flux)
    c = SkyCoord(ra=lon[goodvals]*u.deg, dec=lat[goodvals]*u.deg, frame='icrs')
    pixels[band_label] = np.zeros(len(lon), dtype=int)
    pixels[band_label][goodvals] = hp.ang2pix(nside_out,
            c.galactic.l.value, c.galactic.b.value, lonlat=True)

    tods[band_label] = flux




    flags[band_label] = flag_tot

    return YdayData(
        tods, 
        pixels, 
        flags, 
        time_start=t, 
        time_stop=t, 
        sat_pos_start=sat_pos_start, 
        sat_pos_stop=sat_pos_stop, 
        earth_pos_start=earth_pos_start,
        earth_pos_stop=earth_pos_stop,
    )


def padd_array_gaps(splits: list[np.ndarray], padding: list[np.ndarray]) -> np.ndarray:
    return np.concatenate([np.append(s, p) for s, p in zip(splits, padding)])





def write_band(
        comm_tod: TODLoader, cio: CIO, filename: str, ndet: int, nside_out: int, n_pids: int
        , pid_0: int, det_num: int) -> None:
    COMMON_GROUP = "/common"
    HUFFMAN_COMPRESSION = ["huffman", {"dictNum": 1}]

    det_str = ""
    for i in range(ndet):
        det_str = det_str + f'IRAS_{i+1:02}'
        if i < ndet - 1:
            det_str = det_str + ', '
    comm_tod.init_file(freq=filename, od="", mode="w")



    for v in cio.keys():
        det1 = v
        break

    comm_tod.add_field(COMMON_GROUP + "/fsamp", 1 / SAMP_RATE)
    comm_tod.add_field(COMMON_GROUP + "/nside", [nside_out])
    comm_tod.add_field(COMMON_GROUP + "/det", np.string_(det_str + ","))

    comm_tod.add_field(COMMON_GROUP + "/polang", [0]*ndet)
    comm_tod.add_attribute(COMMON_GROUP + "/polang", "index", det_str)

    comm_tod.add_field(COMMON_GROUP + "/mbang", [0]*ndet)
    comm_tod.add_attribute(COMMON_GROUP + "/mbang", "index", det_str)


    for pid in range(n_pids):
        #DEBUG
        #print(cio[det1].sat_pos_start[pid])
        #print(cio[det1].sat_pos_stop[pid])
        pid_label = f"{pid+pid_0:06}"
        pid_common_group = pid_label + "/common"

        comm_tod.add_field(pid_common_group + "/time", [cio[det1].time_start[pid], 0, 0])
        comm_tod.add_attribute(pid_common_group + "/time", "index", "MJD, OBT, SCET")

        comm_tod.add_field(pid_common_group + "/time_end", [cio[det1].time_stop[pid], 0, 0])
        comm_tod.add_attribute(pid_common_group + "/time_end", "index", "MJD, OBT, SCET")

        comm_tod.add_field(pid_common_group + "/ntod", [len(cio[det1].tods[pid])])

        comm_tod.add_field(pid_common_group + "/satpos", np.array([0,0,0]))
        comm_tod.add_field(pid_common_group + "/satpos_end", np.array([0,0,0]))

        comm_tod.add_field(pid_common_group + "/earthpos", np.array([0,0,0]))
        comm_tod.add_field(pid_common_group + "/earthpos_end",
            np.array([0,0,0]))

        comm_tod.add_attribute(pid_common_group + "/satpos", "index", "X, Y, Z")
        comm_tod.add_attribute(
            pid_common_group + "/satpos", "coords", "heliocentric-ecliptic"
        )

        for i in range(ndet):
            det_lab = f'IRAS_{det_num:02}'
            pid_det_group = f"{pid_label}/{det_lab}"
            comm_tod.add_field(pid_det_group + "/flag", cio[det_lab].flags[pid], HUFFMAN_COMPRESSION)

            comm_tod.add_field(pid_det_group + "/tod", cio[det_lab].tods[pid])
            comm_tod.add_field(pid_det_group + "/pix", cio[det_lab].pixels[pid], HUFFMAN_COMPRESSION)

            # TODO: Get correct polarization angle (detector angle)
            psi_digitize_compression = [
                "digitize",
                {"min": 0, "max": 2 * np.pi, "nbins": 64},
            ]
            comm_tod.add_field(
                pid_det_group + "/psi",
                np.zeros_like(cio[det_lab].tods[pid]),
                [psi_digitize_compression, HUFFMAN_COMPRESSION],
            )

            comm_tod.add_field(pid_det_group + "/outP", np.zeros((2, 1)))

            const_scalars = np.array([1,1,1,1])
            comm_tod.add_field(pid_det_group + "/scalars", const_scalars)
            comm_tod.add_attribute(
                pid_det_group + "/scalars", "index", "gain, sigma0, fknee, alpha"
            )

        comm_tod.finalize_chunk(pid + 1)

    comm_tod.finalize_file()


def write_to_commander_tods(
    cios: dict[str, CIO],
    nside_out: int,
    version: int,
    out_path: Path,
    pid_0: int,
    chunk: int,
    chunk_size: int,
    overwrite: bool = False,
) -> None:
    manager = multiprocessing.Manager()
    pool = multiprocessing.Pool(processes=N_PROC)

    multiprocessor_manager_dicts = {}
    filenames = {}
    for det in IRAS_DETS:
        name = f"IRAS_{det:02}_n{nside_out}_v{version:02}_{chunk:03}"
        multiprocessor_manager_dicts[name] = manager.dict()
        filenames[f'IRAS_{det:02}'] = name

    comm_tod = TODLoader(
        out_path, "", version, multiprocessor_manager_dicts, overwrite=overwrite
    )

    #n_pids = 0
    #for key, cio in zip(cios.keys(), cios.values()):
    #    n_pids = len(cio.time_start)
    #    break
    #if n_pids == 0:
    #    raise ValueError("No CIOs found")


    x = [[]]
    for det in IRAS_DETS:
        x[0].append(
        pool.apply_async(
            write_band,
            args=(
                comm_tod,
                cios,
                filenames[f'IRAS_{det:02}'],
                1,
                nside_out,
                chunk_size,
                pid_0,
                det,
                )))

    for res1 in x:
        for res in res1:
            res.get()

    pool.close()
    pool.join()

    comm_tod.make_filelists()


# SW fsamp 24 Hz, LW 16 Hz
# Normal sampling, they reset either 0.5 sec, 1 sec, or 2 sec, to avoid detector
# saturation.
# Differential sampling/CDS, they only take a sample every two points in high
# intensity regions.

# We should make sure that the data are split based on the reset timing.

# Once a minute, the calibration lamp is flashed for less than one second.

def main() -> None:
    time_delta = timedelta(hours=1)

    pid_0 = 1
    nside_out = 2048  # 2**11

    start_time = time.perf_counter()
    color_corr = False
    version = 0

    from glob import glob

    print(f"{'Writing IRAS h5 files':=^50}")
    print(f"{version=}, {nside_out=}")
    pid_now = 0
    t0 = Time('1981-01-01', scale='utc')
    #for sopobs in tqdm(range(5787)):
    yday_data = []
    chunk_size = 2
    for sopobs in tqdm(range(10)):

        yday_data = get_yday_data(
            sopobs, nside_out=nside_out, color_corr=color_corr,
            )
        cios = get_cios(yday_data)
        cio_time = time.perf_counter() - start_time
        write_to_commander_tods(
            cios,
            nside_out=nside_out,
            version=version,
            out_path=IRAS_OUT_PATH,
            pid_0=sopobs+1,
            chunk=sopobs//chunk_size,
            chunk_size=chunk_size,
            overwrite=True,
        )
    h5_time = time.perf_counter() - start_time
    print("done")
    print(f"time spent writing to h5: {(h5_time/60):2.2f} minutes\n")
    print(f"total time: {((h5_time + cio_time)/60):2.2f} minutes")
    print(f"{'':=^50}")


if __name__ == "__main__":
    main()
