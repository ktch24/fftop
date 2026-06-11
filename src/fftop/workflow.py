#!/usr/bin/env python
# coding: utf-8

import os
import sys
import multiprocessing as mp

import zarr
from tqdm import tqdm


def afterEquals(s, part=1):
    return s.split('=', part)[part].strip()


def read_region_file(region_file_path):
    with open(region_file_path, 'r', encoding='utf-8') as f:
        contents = f.readlines()
    return [afterEquals(contents[i]) for i in range(18)]


def readSpec(output_file):
    with open(output_file, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()[:7]]


def run_dave_process(args):
    data_dir, regionNum, nx, ny, nt_start, nt_end, velSmooth, process_id = args

    from .dave4vm import run_dave4vm_series

    run_dave4vm_series(
        field_loc=data_dir,
        field_tag=regionNum,
        nx=nx,
        ny=ny,
        start_index=nt_start,
        end_index=nt_end,
        window=int(velSmooth),
    )

    return (process_id, nt_start, nt_end, 0)


def run_potential_process(args):
    regionNum, start_time, end_time, nx, ny, data_dir, process_id = args

    from .potential import main as potential_main

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "fftop-potential",
            str(regionNum),
            str(start_time),
            str(end_time),
            str(nx),
            str(ny),
            str(data_dir),
        ]
        potential_main()
    finally:
        sys.argv = old_argv

    return (process_id, start_time, end_time, 0)


def validate_magdown_success(outputDir, regionNum):
    spec_path = os.path.join(outputDir, 'specifications.txt')
    zarr_path = os.path.join(outputDir, 'Data', f'field_data_{regionNum}.zarr')
    if not os.path.isfile(spec_path):
        raise RuntimeError(f"MagDown failed: missing {spec_path}")
    if not os.path.isdir(zarr_path):
        raise RuntimeError(f"MagDown failed: missing field store {zarr_path}")


def split_ranges(total, n_chunks):
    chunk_size = total // n_chunks
    remainder = total % n_chunks
    start = 0
    out = []
    for i in range(n_chunks):
        end = start + chunk_size - 1 + (1 if i < remainder else 0)
        if start <= end:
            out.append((i, start, min(end, total - 1)))
        start = end + 1
    return out


def main(region_file, topology_only=False, skip_download=False, ck=16):
    if region_file is None:
        raise ValueError("Must provide a region processing file")

    region_file_path = os.fspath(region_file)
    varAR = read_region_file(region_file_path)
    (
        regionNum,
        downloadData,
        startYear,
        startMonth,
        startDay,
        startHour,
        endYear,
        endMonth,
        endDay,
        endHour,
        velSmooth,
        inputDir,
        outputDir,
        topology,
        cutoff,
        sampling,
        removeImages,
        regEmail,
    ) = varAR

    os.makedirs(inputDir, exist_ok=True)
    os.makedirs(outputDir, exist_ok=True)
    os.makedirs(os.path.join(outputDir, 'Data'), exist_ok=True)
    dataDir = os.path.join(outputDir, 'Data')

    specLoc = os.path.join(outputDir, 'specifications.txt')
    store_path = os.path.join(dataDir, f'field_data_{regionNum}.zarr')

    should_download = (not skip_download) and downloadData.casefold() in ('true', 'manual')

    # Only do download/preparation if we are not in topology-only mode
    if should_download and not topology_only:
        from .download import main as download_main

        old_argv = sys.argv[:]
        try:
            sys.argv = [
                "fftop-download",
                regionNum,
                startYear,
                startMonth,
                startDay,
                startHour,
                endYear,
                endMonth,
                endDay,
                endHour,
                inputDir,
                outputDir,
                regEmail,
                downloadData,
                velSmooth,
                cutoff,
                sampling,
            ]
            download_main()
        finally:
            sys.argv = old_argv

        validate_magdown_success(outputDir, regionNum)

    if not os.path.isfile(specLoc):
        raise FileNotFoundError(
            'specifications.txt does not exist; download/preparation failed or no prepared data are available.'
        )

    if not os.path.isdir(store_path):
        raise FileNotFoundError(
            f'Prepared field store does not exist: {store_path}'
        )

    # Topology-only mode: just run topology on existing processed data
    if topology_only:
        print("\n=== Running topology-only stage ===\n")
        from .topology import process_field_data

        process_field_data(
            param_file=specLoc,
            ck=ck,
            field_loc=store_path,
            field_tag=str(regionNum),
            vel_tag=str(velSmooth),
            savdir=outputDir,
            zarr_file=f"topology_{regionNum}.zarr",
        )
        return

    specDetails = readSpec(specLoc)
    nx, ny, nt = int(specDetails[1]), int(specDetails[2]), int(specDetails[3])

    num_cores = max(1, min(4, mp.cpu_count() - 1))
    ranges = split_ranges(nt, num_cores)

    root = zarr.open(store_path, mode='a')
    n_out = root['bz'].shape[2]
    if 'Ux' not in root:
        root.create_array('Ux', shape=(nx, ny, n_out), chunks=(nx, ny, 1), dtype='f8')
    if 'Uy' not in root:
        root.create_array('Uy', shape=(nx, ny, n_out), chunks=(nx, ny, 1), dtype='f8')
    if 'Uz' not in root:
        root.create_array('Uz', shape=(nx, ny, n_out), chunks=(nx, ny, 1), dtype='f8')

    process_args = [(dataDir, regionNum, nx, ny, start, end, velSmooth, pid) for pid, start, end in ranges]
    with mp.Pool(processes=num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(run_dave_process, process_args),
                total=len(process_args),
                desc='Processing velocity chunks',
                unit='chunk',
            )
        )
    failed = [r for r in results if r[-1] != 0]
    if failed:
        raise RuntimeError(f'DAVE4VM failed for chunks: {failed}')

    root = zarr.open(store_path, mode='a')
    n_out = root['bz'].shape[2]
    if 'Bxp' not in root:
        root.create_array('Bxp', shape=(nx, ny, n_out), chunks=(nx, ny, 1), dtype='f8')
    if 'Byp' not in root:
        root.create_array('Byp', shape=(nx, ny, n_out), chunks=(nx, ny, 1), dtype='f8')

    process_args = [(regionNum, start, end + 1, ny, nx, dataDir, pid) for pid, start, end in ranges]
    with mp.Pool(processes=num_cores) as pool:
        results = list(
            tqdm(
                pool.imap(run_potential_process, process_args),
                total=len(process_args),
                desc='Potential field chunks',
                unit='chunk',
            )
        )
    failed = [r for r in results if r[-1] != 0]
    if failed:
        raise RuntimeError(f'Potential field solver failed for chunks: {failed}')

    if topology.casefold() == "true":
        print("\n=== Running topology stage ===\n")
        from .topology import process_field_data

        process_field_data(
            param_file=specLoc,
            ck=ck,
            field_loc=store_path,
            field_tag=str(regionNum),
            vel_tag=str(velSmooth),
            savdir=outputDir,
            zarr_file=f"topology_{regionNum}.zarr",
        )

if __name__ == '__main__':
    mp.freeze_support()
    if len(sys.argv) < 2:
        raise ValueError("Usage: python -m fftop.workflow <region_file_path>")
    main(sys.argv[1])
