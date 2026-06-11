#!/usr/bin/env python
# coding: utf-8

import os
import sys
import datetime
import tarfile
import glob
import time
from datetime import timedelta

import numpy as np
import sunpy.map
import drms
import zarr


def downloadMag(start, end, inputFolder, sharp, email):
    client = drms.Client(email=email)
    query_string = f'hmi.sharp_cea_720s[{sharp}][{start}-{end}]{{Bp,Br,Bt}}'
    export_request = client.export(query_string, method='url-tar', protocol='fits')
    export_request.wait(timeout=3600)
    export_request.download(inputFolder)


def removeNaNs(field):
    field = np.asarray(field)
    field[np.isnan(field)] = 0.0
    return field


def frontName(time, regionNum):
    yt = str(time.year)
    mt = str(time.month).zfill(2)
    dt = str(time.day).zfill(2)
    ht = str(time.hour).zfill(2)
    mint = str(time.minute).zfill(2)
    return 'hmi.sharp_cea_720s.' + regionNum + '.' + yt + mt + dt + '_' + ht + mint + '00_TAI'


def _load_component(path, kind):
    arr = sunpy.map.Map(path).data.astype(np.float64)
    arr = removeNaNs(arr)
    if kind == 'By':
        arr = -1.0 * arr
    # SunPy map data are indexed as [y, x]; the DAVE4VM pipeline expects [x, y].
    return np.ascontiguousarray(arr.T, dtype=np.float64)



def _latest_tar_file(input_dir):
    tar_files = glob.glob(os.path.join(input_dir, '*.tar'))
    if not tar_files:
        return None
    return max(tar_files, key=os.path.getmtime)


def _wait_for_stable_file(path, checks=3, delay=2.0):
    """Wait until a file exists and its size is stable across several checks."""
    if path is None:
        return

    last_size = None
    stable_count = 0
    for _ in range(60):
        if not os.path.isfile(path):
            time.sleep(delay)
            continue

        try:
            size = os.path.getsize(path)
        except OSError:
            time.sleep(delay)
            continue

        if size > 0 and size == last_size:
            stable_count += 1
            if stable_count >= checks:
                return
        else:
            stable_count = 0
            last_size = size

        time.sleep(delay)

    raise RuntimeError(
        f"Downloaded tar file did not reach a stable size in time: {path}"
    )

def _open_store(data_dir, region_num, nx, ny, nmag):
    os.makedirs(data_dir, exist_ok=True)
    store_path = os.path.join(data_dir, f'field_data_{region_num}.zarr')
    root = zarr.open(store_path, mode='w')
    for name in ('bx', 'by', 'bz'):
        root.create_array(name, shape=(nx, ny, nmag), chunks=(nx, ny, 1), dtype='f8')
    root.create_array('valid', shape=(nmag,), chunks=(nmag,), dtype='i1')
    root['valid'][:] = 0
    return root, store_path


def main():
    regionNum = sys.argv[1]
    startYear = sys.argv[2]
    startMonth = sys.argv[3]
    startDay = sys.argv[4]
    startHour = sys.argv[5]
    endYear = sys.argv[6]
    endMonth = sys.argv[7]
    endDay = sys.argv[8]
    endHour = sys.argv[9]
    inputDir = sys.argv[10]
    outputDir = sys.argv[11]
    regEmail = sys.argv[12]
    downloadData = sys.argv[13]
    velSmooth = sys.argv[14]
    cutoff = sys.argv[15]
    sampling = sys.argv[16]

    startTime = f"{startYear}.{startMonth}.{startDay}_{startHour}:00:00"
    endTime = f"{endYear}.{endMonth}.{endDay}_{endHour}:12:00"

    if downloadData.casefold() == 'true':
        downloadMag(startTime, endTime, inputDir, regionNum, regEmail)

    nameTar = _latest_tar_file(inputDir)
    if nameTar is None:
        raise SystemExit('ERROR: tar file not found, possible connection issue. Consider a manual download (see online documentation).')

    print(f'Using tar file: {nameTar}')
    _wait_for_stable_file(nameTar)

    try:
        with tarfile.open(nameTar, 'r') as tf:
            print('Opened tar file')
            tf.extractall(inputDir)
            print('All FITS files extracted')
    except tarfile.ReadError as e:
        raise RuntimeError(
            f"Failed to extract tar archive {nameTar}. "
            "The file may still be syncing / incomplete on disk. "
            "Wait a moment and retry, or move the input/output directories off OneDrive."
        ) from e

    st = datetime.datetime(int(startYear), int(startMonth), int(startDay), int(startHour), 0, 0)
    et = datetime.datetime(int(endYear), int(endMonth), int(endDay), int(endHour), 12, 0)
    jump = timedelta(minutes=12)
    nt = int((et - st) / jump)
    writeInitialData = True
    root = None
    valid = np.zeros(nt + 1, dtype=np.int8)

    for i in range(nt + 1):
        print('Iteration number: ' + str(i))
        time = st + i * jump
        front = frontName(time, regionNum)
        pathBp = os.path.join(inputDir, front + '.Bp.fits')
        pathBt = os.path.join(inputDir, front + '.Bt.fits')
        pathBr = os.path.join(inputDir, front + '.Br.fits')
        if os.path.exists(pathBp) and os.path.exists(pathBr) and os.path.exists(pathBt):
            bx = _load_component(pathBp, 'Bx')
            by = _load_component(pathBt, 'By')
            bz = _load_component(pathBr, 'Bz')
            if root is None:
                nx, ny = bx.shape
                data_dir = os.path.join(outputDir, 'Data')
                root, store_path = _open_store(data_dir, regionNum, nx, ny, nt + 1)
            root['bx'][:, :, i] = bx
            root['by'][:, :, i] = by
            root['bz'][:, :, i] = bz
            valid[i] = 1
            if writeInitialData:
                tempIn = sunpy.map.Map(pathBr)
                xdim = str(tempIn.dimensions[0]).split('.')[0].strip()
                ydim = str(tempIn.dimensions[1]).split('.')[0].strip()
                refPx = str(tempIn.reference_pixel[0]).split(' ')[0].strip()
                refPy = str(tempIn.reference_pixel[1]).split(' ')[0].strip()
                observatory = tempIn.observatory
                instrument = tempIn.instrument
                detector = tempIn.detector
                obstime = str(tempIn.date).replace('-', '/')
                rc = str(tempIn.reference_coordinate).split('(lon, lat) in deg')[1].strip()
                rcx = rc.split(' ')[0].strip()[1:-1]
                rcy = rc.split(' ')[1].strip()[:-2]
                scaleX = str(tempIn.scale[0]).split(' ')[0].strip()
                scaleY = str(tempIn.scale[1]).split(' ')[0].strip()
                with open(os.path.join(outputDir, 'specifications.txt'), 'w') as datFile:
                    datFile.write(regionNum + '\n')
                    datFile.write(xdim + '\n')
                    datFile.write(ydim + '\n')
                    datFile.write(str(nt) + '\n')
                    datFile.write(velSmooth + '\n')
                    datFile.write(cutoff + '\n')
                    datFile.write(sampling)
                with open(os.path.join(outputDir, 'header.txt'), 'w') as datFile:
                    datFile.write(refPx + '\n')
                    datFile.write(refPy + '\n')
                    datFile.write(observatory + '\n')
                    datFile.write(instrument + '\n')
                    datFile.write(detector + '\n')
                    datFile.write(obstime + '\n')
                    datFile.write(rcx + '\n')
                    datFile.write(rcy + '\n')
                    datFile.write(scaleX + '\n')
                    datFile.write(scaleY)
                writeInitialData = False

    if root is not None:
        root['valid'][:] = valid

    if root is None:
        raise RuntimeError('No FITS component triplets were found; MagDown could not populate the field Zarr store.')

    print('Removing FITS files')
    for item in os.listdir(inputDir):
        if item.endswith('.fits'):
            os.remove(os.path.join(inputDir, item))


if __name__ == "__main__":
    main()
