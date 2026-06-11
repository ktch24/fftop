#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from scipy.fft import fftn, ifftn, fftfreq, fftshift, ifftshift
import numpy as np
import os
import sys
import zarr
from pathlib import Path


def _discover_field_store(field_loc: str | Path, region_num: str) -> Path | None:
    p = Path(field_loc)
    if p.suffix == '.zarr' and p.exists():
        return p
    if p.is_dir():
        candidate = p / f"field_data_{region_num}.zarr"
        if candidate.exists():
            return candidate
    return None


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute potential field from magnetic field data."
    )

    parser.add_argument("regionNum")
    parser.add_argument("startfl", type=int)
    parser.add_argument("endfl", type=int)
    parser.add_argument("ly", type=int)
    parser.add_argument("lx", type=int)
    parser.add_argument("outDir")

    args = parser.parse_args()

    regionNum = args.regionNum
    startfl = args.startfl
    endfl = args.endfl
    ly = args.ly
    lx = args.lx
    outDir = args.outDir

    store_path = _discover_field_store(outDir, regionNum)

    freqx = fftfreq(lx)
    freqy = fftfreq(ly)
    kx = fftshift(freqx)
    ky = fftshift(freqy)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    k_squared = KX**2 + KY**2
    sqrt_k_squared = np.zeros_like(k_squared, dtype=np.float64)
    nonzero = k_squared != 0
    sqrt_k_squared[nonzero] = np.sqrt(k_squared[nonzero])

    Kx = np.zeros_like(KX, dtype=np.complex128)
    Ky = np.zeros_like(KY, dtype=np.complex128)
    Kx[nonzero] = -1j * KX[nonzero] / sqrt_k_squared[nonzero]
    Ky[nonzero] = -1j * KY[nonzero] / sqrt_k_squared[nonzero]

    if store_path is not None:
        root = zarr.open(store_path, mode='a')
        if 'bz' not in root:
            raise KeyError(f"bz not found in {store_path}")
        if 'Bxp' not in root or 'Byp' not in root:
            raise KeyError(
                f"Bxp/Byp datasets must be created before worker processes start in {store_path}"
            )

    s1 = os.path.join(outDir, f'bz_{regionNum}_')
    s3 = '.txt'

    for i in range(startfl, endfl):
        print(f"Potential field iteration: {i}")
        try:
            if store_path is not None:
                BR = np.asarray(root['bz'][:, :, i], dtype=np.float64).T
            else:
                path = f"{s1}{i}{s3}"
                if not os.path.isfile(path):
                    continue
                pol = np.loadtxt(path, delimiter=" ", dtype=np.float64)
                BR = pol.reshape(lx, ly).T

            FFTB = fftshift(fftn(BR))
            Mx = FFTB * Kx
            My = FFTB * Ky
            Bxp = ifftn(ifftshift(Mx)).real
            Byp = ifftn(ifftshift(My)).real

            Bxp = Bxp.T
            Byp = Byp.T

            if store_path is not None:
                root['Bxp'][:, :, i] = Bxp.astype(np.float64)
                root['Byp'][:, :, i] = Byp.astype(np.float64)
            else:
                np.savetxt(os.path.join(outDir, f"Bxp_{regionNum}_{i}.txt"), np.reshape(Bxp, (lx*ly, 1)))
                np.savetxt(os.path.join(outDir, f"Byp_{regionNum}_{i}.txt"), np.reshape(Byp, (lx*ly, 1)))
        except Exception as e:
            print(f"Error processing index {i}: {e}")
            raise

if __name__ == "__main__":
    main()
