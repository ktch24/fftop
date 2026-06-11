#!/usr/bin/env python
# coding: utf-8

"""
FFTop – Free-Space Winding / Helicity Topology Pipeline
=======================================================

This module implements the full data-processing workflow for
computing free-space magnetic winding flux, helicity flux, and 
related flux-balance diagnostics from photospheric vector 
magnetograms and derived velocity fields.
It is designed for SHARP / HMI data products, but applies to any 
data arranged as (nx × ny × nz) slices with vector fields stored in 
text files per snapshot.

Core functionality
------------------
1. **Data Loading**
Each z-slice of magnetic (B, B_pot) and velocity (v) fields is stored
in flat text format with x-fastest ordering. The readers in this
module reshape these into arrays with consistent `(x, y)` indexing
and assemble 3-D volumes in chunks.

2. **Field Decomposition**
The velocity and magnetic fields are separated into:
       - Current-carrying part (U_cur)
       - Potential part (U_pot)
       - Total field (U)
along with parallel-to-B velocity removal and (B - B_pot) 
decomposition.

3. **Free-Space Winding Gauge**
Using zero-padded FFT convolution (Hockney scheme), we compute the
winding-gauge vector potential A^W in free space for each slice,
splitting vertical-mean and fluctuating contributions.

4. **Flux Density Computation**
From the resulting vector potentials, we compute:
       - Winding flux density:      windingFlux
       - Helicity flux density:     helicityFlux
       - Difference measures:       dLFlux  and  dHFlux
All are returned slice-wise as 3-D arrays.

5. **Parallel Chunk Processing**
The main entry point `process_field_data(...)` distributes the 
computation across CPU cores in chunks of slices (ck). Each worker 
reads raw files, builds fields, computes flux densities, and returns 
slabs. The main process writes results into a Zarr store and also 
integrates per-slice totals.

6. **Output**
   - A compressed Zarr dataset containing full 3-D flux-density volumes.
   - Per-slice integrated flux time series saved as `*_totals.npy`.

Typical Usage
-------------
process_field_data(
param_file=".../specifications.txt",
            ck=64,
            field_loc="path/to/input",
            field_tag="377",
            vel_tag="20",
            zarr_path=".../topology_377.zarr"
           )


Performance Notes
-----------------
- Parallelization uses Python `multiprocessing` with per-chunk 
  scheduling.
- Internal BLAS/FFT threading should be limited (OMP/MKL=1) for 
  optimal scaling.
- Zarr allows out-of-core storage and chunk-wise writing without 
  holding full arrays in memory.

This module is intended for research use in studies of magnetic 
helicity, winding, flare productivity, and field topology evolution 
in solar active regions.
"""


from __future__ import annotations
import numpy as np
from pathlib import Path
from typing import Dict, Iterable
import os
import shutil
import zarr
from pathlib import Path
from numcodecs import Blosc
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from tqdm import tqdm



def _load_xy_field_txt(path: Path, nx: int, ny: int, dtype=np.float64) -> np.ndarray:
    """
    Load a scalar field saved as one value per line and return array shaped (nx, ny),
    with indexing arr[x, y], matching the C++ reader.

    C++ logic filled field[x][y] with x incrementing fastest, so:
      arr[x, y] = flat[y*nx + x]
    which corresponds to reshape(flat, (ny, nx)).T
    """
    flat = np.loadtxt(path, dtype=dtype)
    if flat.size != nx * ny:
        raise ValueError(f"{path} has {flat.size} values, expected {nx*ny} for nx={nx}, ny={ny}")
    return flat.reshape(ny, nx).T  # shape (nx, ny)



def plot_2d_field(
    F2d,
    label="Field",
    cmap="RdBu_r",
    symmetric=True,
    percentile=None,
    index=None
):
    """
    Plot a single 2-D field with optional symmetric/percentile colour limits.

    Parameters
    ----------
    F2d : np.ndarray
        2-D array to plot.
    label : str, optional
        Title or label for the plot.
    cmap : str, default "RdBu_r"
        Colormap for imshow.
    symmetric : bool, default True
        If True, color limits are symmetric about zero.
    percentile : float or None, default None
        If set (e.g. 0.75), use that percentile of |F2d| for limits.
    index : int, optional
        Snapshot index for the title.
    """
    if percentile is not None:
        lim = np.percentile(np.abs(F2d), percentile * 100)
        vmin, vmax = (-lim, lim) if symmetric else (0, lim)
    elif symmetric:
        vmax = np.nanmax(np.abs(F2d))
        vmin = -vmax
    else:
        vmin, vmax = np.nanmin(F2d), np.nanmax(F2d)

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    im = ax.imshow(F2d.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    title = label
    if index is not None:
        title += f" (snapshot {index})"
    ax.set_title(title)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.show()



def plot_vector_field_components(
    F, labels=None, index=None, cmap="RdBu_r",
    symmetric=True, percentile=None
):
    """
    Plot 3-component field (F[0], F[1], F[2]) side by side.

    Parameters
    ----------
    F : np.ndarray
        Array shaped (3, nx, ny) containing the three components.
    labels : list of str, optional
        Labels for the components (default: ["F1","F2","F3"]).
    index : int, optional
        Snapshot index for the title.
    cmap : str, default "RdBu_r"
        Colormap for imshow.
    symmetric : bool, default True
        If True, set color limits symmetric around zero.
    percentile : float or None, default None
        If given (e.g. 0.75), use the given percentile of abs(F)
        to set the color limits, instead of max/min.
    """
    if labels is None:
        labels = ["F1", "F2", "F3"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # choose colour limits
    if percentile is not None:
        lim = np.percentile(np.abs(F), percentile * 100)
        vmin, vmax = (-lim, lim) if symmetric else (0, lim)
    elif symmetric:
        vmax = np.max(np.abs(F))
        vmin = -vmax
    else:
        vmin, vmax = np.min(F), np.max(F)

    for i, ax in enumerate(axes):
        im = ax.imshow(
            F[i].T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(labels[i])
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if index is not None:
        fig.suptitle(f"{labels} at snapshot {index}", fontsize=14)

    plt.show()



def plot_vector_field_slice(
    F, k, labels=None, title=None, cmap="RdBu_r",
    symmetric=True, percentile=None, transpose=True
):
    """
    Plot a z-slice (index k) of a 3D vector field.

    Parameters
    ----------
    F : np.ndarray
        Array shaped (nx, ny, nz, 3).
    k : int
        Z index (0 <= k < nz) to slice at.
    labels : list[str] | None
        Labels for components, default ["F_x","F_y","F_z"].
    title : str | None
        Figure title; if None, shows "k = {k}".
    cmap : str
        Matplotlib colormap for imshow.
    symmetric : bool
        If True, color limits symmetric around 0.
    percentile : float | None
        If set (e.g. 0.95), clip color limits to that percentile of |data|.
        Applied on the slice (all components together).
    transpose : bool
        If True (default), transpose to show as image with y vertical (imshow expects row-major).
    """
    if F.ndim != 4 or F.shape[-1] != 3:
        raise ValueError(f"F must have shape (nx, ny, nz, 3); got {F.shape}")

    nx, ny, nz, _ = F.shape
    if not (0 <= k < nz):
        raise IndexError(f"k must be in [0, {nz-1}], got {k}")

    if labels is None:
        labels = ["F_x", "F_y", "F_z"]

    # Extract slice (nx, ny, 3)
    slice3 = F[:, :, k, :]  # (nx, ny, 3)

    # Determine color limits
    if percentile is not None:
        lim = np.percentile(np.abs(slice3), percentile * 100.0)
        vmin, vmax = (-lim, lim) if symmetric else (0.0, lim)
    elif symmetric:
        vmax = np.max(np.abs(slice3))
        vmin = -vmax
    else:
        vmin, vmax = np.min(slice3), np.max(slice3)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

    # helper to prepare array for imshow
    def _img(a):
        return a.T if transpose else a

    for i, ax in enumerate(axes):
        im = ax.imshow(
            _img(slice3[..., i]),
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax
        )
        ax.set_title(labels[i])
        ax.set_xlabel("x")
        if i == 0:
            ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if title is None:
        title = f"Slice k = {k}"
    fig.suptitle(title, fontsize=14)
    plt.show()



def _load_xy_field_txt(path: Path, nx: int, ny: int, dtype=np.float64) -> np.ndarray:
    """
    Load one scalar field stored as 1 value per line.
    Returns shape (nx, ny) with arr[x, y].
    """
    flat = np.loadtxt(path, dtype=dtype)
    if flat.size != nx * ny:
        raise ValueError(f"{path} has {flat.size} values; expected {nx*ny} for nx={nx}, ny={ny}")
    return flat.reshape(ny, nx).T  # (nx, ny), x-fastest

def _indices(i_start: int, i_end: int) -> Iterable[int]:
    if i_end < i_start:
        raise ValueError("i_end must be >= i_start")
    return range(i_start, i_end + 1)



def _read_field_file(fname, nx, ny):
    """Read a single 2D field file. If missing, return zeros.
       NOTE: files are x-fastest → reshape(ny, nx).T"""
    if not os.path.exists(fname):
        return np.zeros((nx, ny))
    flat = np.loadtxt(fname)
    if flat.size != nx*ny:
        raise ValueError(f"{fname} has {flat.size} values; expected {nx*ny}")
    return flat.reshape((ny, nx)).T   # <-- important!

def read_volume_minimal(field_loc, field_tag, vel_tag, i_start, i_end, nx, ny):
    """
    Minimal loader: reads bx, by, bz and vx, vy, vz.
    Returns dict with (nx, ny, nz, 3) arrays.
    """
    nz = i_end - i_start + 1
    B = np.zeros((nx, ny, nz, 3))
    V = np.zeros((nx, ny, nz, 3))

    for k, idx in enumerate(range(i_start, i_end + 1)):
        # Build file paths
        vx = os.path.join(field_loc, f"Ux_{field_tag}_{vel_tag}_{idx}.txt")
        vy = os.path.join(field_loc, f"Uy_{field_tag}_{vel_tag}_{idx}.txt")
        vz = os.path.join(field_loc, f"Uz_{field_tag}_{vel_tag}_{idx}.txt")
        bx = os.path.join(field_loc, f"bx_{field_tag}_{idx}.txt")
        by = os.path.join(field_loc, f"by_{field_tag}_{idx}.txt")
        bz = os.path.join(field_loc, f"bz_{field_tag}_{idx}.txt")

        # Read with missing-file safety
        V[..., k, 0] = _read_field_file(vx, nx, ny)
        V[..., k, 1] = _read_field_file(vy, nx, ny)
        V[..., k, 2] = _read_field_file(vz, nx, ny)

        B[..., k, 0] = _read_field_file(bx, nx, ny)
        B[..., k, 1] = _read_field_file(by, nx, ny)
        B[..., k, 2] = _read_field_file(bz, nx, ny)

    return {"B": B, "V": V}



def read_volume_full(field_loc, field_tag, vel_tag, i_start, i_end, nx, ny):
    """
    Full loader: reads bx, by, bz, vx, vy, vz, and computes
    Ucur, Rcur, Upot, Rpot.  The z-components of Ucur/Upot/Rcur/Rpot
    are set to the sign of Bz (np.sign(bz_arr)).
    Returns dict of arrays (nx, ny, nz, 3).
    """
    nz = i_end - i_start + 1

    B    = np.zeros((nx, ny, nz, 3), dtype=np.float64)
    V    = np.zeros_like(B)
    Ucur = np.zeros_like(B)
    Rcur = np.zeros_like(B)
    Upot = np.zeros_like(B)
    Rpot = np.zeros_like(B)

    eps = np.float64(1e-12)
    for k, idx in enumerate(range(i_start, i_end + 1)):
        # file paths
        vx  = os.path.join(field_loc, f"Ux_{field_tag}_{vel_tag}_{idx}.txt")
        vy  = os.path.join(field_loc, f"Uy_{field_tag}_{vel_tag}_{idx}.txt")
        vz  = os.path.join(field_loc, f"Uz_{field_tag}_{vel_tag}_{idx}.txt")
        bx  = os.path.join(field_loc, f"bx_{field_tag}_{idx}.txt")
        by  = os.path.join(field_loc, f"by_{field_tag}_{idx}.txt")
        bz  = os.path.join(field_loc, f"bz_{field_tag}_{idx}.txt")
        bxp = os.path.join(field_loc, f"Bxp_{field_tag}_{idx}.txt")
        byp = os.path.join(field_loc, f"Byp_{field_tag}_{idx}.txt")

        # load as (nx, ny) arrays (use your corrected _read_field_file)
        vx_arr  = _read_field_file(vx,  nx, ny)
        vy_arr  = _read_field_file(vy,  nx, ny)
        vz_arr  = _read_field_file(vz,  nx, ny)
        bx_arr  = _read_field_file(bx,  nx, ny)
        by_arr  = _read_field_file(by,  nx, ny)
        bz_arr  = _read_field_file(bz,  nx, ny)
        bxp_arr = _read_field_file(bxp, nx, ny)
        byp_arr = _read_field_file(byp, nx, ny)

        # ---- line-of-sight (parallel-to-B) removal: v <- v - B * (v·B)/|B|^2 ----
        # compute |B|^2 and v·B
        B2   = bx_arr*bx_arr + by_arr*by_arr + bz_arr*bz_arr
        vdotB = vx_arr*bx_arr + vy_arr*by_arr + vz_arr*bz_arr

        # safe factor = (v·B)/( |B|^2 + eps )
        factor = vdotB / (B2 + eps)

        # subtract the parallel component
        vx_arr = vx_arr - factor * bx_arr
        vy_arr = vy_arr - factor * by_arr
        vz_arr = vz_arr - factor * bz_arr


        # stack into B and V
        B[..., k, 0] = bx_arr
        B[..., k, 1] = by_arr
        B[..., k, 2] = bz_arr
        V[..., k, 0] = vx_arr
        V[..., k, 1] = vy_arr
        V[..., k, 2] = vz_arr

        #line of sight correction
         

        # current-carrying and potential parts
        bx_cur = bx_arr - bxp_arr
        by_cur = by_arr - byp_arr

        mask = np.abs(bz_arr) > 1e-6

        rvx_cur = np.zeros((nx, ny), dtype=np.float64)
        rvy_cur = np.zeros((nx, ny), dtype=np.float64)
        rvx_pot = np.zeros((nx, ny), dtype=np.float64)
        rvy_pot = np.zeros((nx, ny), dtype=np.float64)

        ux_cur  = np.zeros((nx, ny), dtype=np.float64)
        uy_cur  = np.zeros((nx, ny), dtype=np.float64)
        ux_pot  = np.zeros((nx, ny), dtype=np.float64)
        uy_pot  = np.zeros((nx, ny), dtype=np.float64)

        rvx_cur[mask] = vz_arr[mask] * bx_cur[mask] / bz_arr[mask]
        rvy_cur[mask] = vz_arr[mask] * by_cur[mask] / bz_arr[mask]
        rvx_pot[mask] = vz_arr[mask] * bxp_arr[mask] / bz_arr[mask]
        rvy_pot[mask] = vz_arr[mask] * byp_arr[mask] / bz_arr[mask]

        ux_cur[mask]  = vx_arr[mask] - rvx_cur[mask]
        uy_cur[mask]  = vy_arr[mask] - rvy_cur[mask]
        ux_pot[mask]  = vx_arr[mask] - rvx_pot[mask]
        uy_pot[mask]  = vy_arr[mask] - rvy_pot[mask]

        # z component = sign of Bz
        sign_bz = np.sign(bz_arr).astype(np.float64)

        Rcur[..., k, 0] = rvx_cur
        Rcur[..., k, 1] = rvy_cur
        Rcur[..., k, 2] = 1.0

        Rpot[..., k, 0] = rvx_pot
        Rpot[..., k, 1] = rvy_pot
        Rpot[..., k, 2] = 1.0

        Ucur[..., k, 0] = ux_cur
        Ucur[..., k, 1] = uy_cur
        Ucur[..., k, 2] = 1.0

        Upot[..., k, 0] = ux_pot
        Upot[..., k, 1] = uy_pot
        Upot[..., k, 2] = 1.0

    return {
        "B": B,
        "V": V,
        "Ucur": Ucur,
        "Rcur": Rcur,
        "Upot": Upot,
        "Rpot": Rpot,
    }



# ---------- geometry precompute: W(x,y) for the mean vertical part ----------
def precompute_W_rect(nx, ny, dx, dy, pad_factor=2, use_discrete_symbol=True, include_const=True):
    """
    Precompute W = (Wx, Wy) for a rectangular grid S:
        W(x,y) = (1/2π) ∫_S r^⊥ / |r|^2 dA',  r = (x-x', y-y', 0), r^⊥ = (-r_y, r_x, 0)
    so that for a slice-mean vertical field u_c = (0,0,<u_z>), A^W[u_c] = <u_z> * W.

    Free-space convolution via zero-padding (Hockney): pad -> FFT -> multiply by ∇^⊥ Δ^{-1} symbol -> iFFT -> crop.
    """
    npx, npy = pad_factor*nx, pad_factor*ny
    chi = np.zeros((npx, npy), dtype=float)
    chi[:nx, :ny] = 1.0

    # Wavenumbers on the padded domain
    kx = 2*np.pi * np.fft.fftfreq(npx, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(npy, d=dy)
    kx2d, ky2d = np.meshgrid(kx, ky, indexing='ij')

    if use_discrete_symbol:
        # Discrete symbols (2nd-order): derivative i*sin(kh)/h ; Laplacian 4 sin^2(kh/2)/h^2
        kx_t = np.sin(kx2d*dx)/dx
        ky_t = np.sin(ky2d*dy)/dy
        lap_t = 4*np.sin(0.5*kx2d*dx)**2/dx**2 + 4*np.sin(0.5*ky2d*dy)**2/dy**2
        inv_lap = np.zeros_like(lap_t)
        m = lap_t > 0
        inv_lap[m] = 1.0/lap_t[m]
        Kx =  1j * ky_t * inv_lap
        Ky = -1j * kx_t * inv_lap
    else:
        k2 = kx2d**2 + ky2d**2
        inv_k2 = np.zeros_like(k2)
        m = k2 > 0
        inv_k2[m] = 1.0/k2[m]
        Kx =  1j * ky2d * inv_k2
        Ky = -1j * kx2d * inv_k2

    if include_const:
        const = 1.0/(2*np.pi)
        Kx *= const
        Ky *= const

    Chi = np.fft.fft2(chi)
    Wx_pad = np.fft.ifft2(Kx * Chi).real
    Wy_pad = np.fft.ifft2(Ky * Chi).real

    Wx = Wx_pad[:nx, :ny]
    Wy = Wy_pad[:nx, :ny]
    return Wx, Wy



def get_product_single(a,b):
    #Returns signed product of two single vector fields
    #Last index is the signed one
    mult = a*b
    return np.sum(mult, axis = 3)



def getFLHDenSingle(b,a, signed = True):
    multProd = a*b
    flh_density = get_product_single(b,a)
    if signed:
        return flh_density   #For use with the flh
    else:
        return np.abs(flh_density)   #For use with the winding? Apparently not.


# ---------- sampled-kernel free-space winding gauge (FD/rectangular rule + FFT) ----------
def _build_rect_kernels_fft(nx, ny, dx, dy, pad_factor=2, eps=0.0):
    """
    Build FFTs of the sampled real-space winding-gauge kernels for a rectangular
    grid. This is the FFT-accelerated version of the direct rectangular-rule
    Biot--Savart/winding-gauge sum, with the self term set to zero.

    The kernels include the quadrature factor dx*dy and the 1/(2*pi) winding
    prefactor. Therefore densities computed with these kernels should NOT be
    multiplied by another 1/(2*pi).
    """
    npx, npy = pad_factor * nx, pad_factor * ny

    xs = (np.arange(npx) - npx // 2) * dx
    ys = (np.arange(npy) - npy // 2) * dy
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    r2 = X * X + Y * Y

    if eps > 0.0:
        r2 = r2 + (r2 == 0.0) * eps**2

    c = (dx * dy) / (2.0 * np.pi)

    Kx = np.zeros_like(r2, dtype=np.float64)
    Ky = np.zeros_like(r2, dtype=np.float64)
    Kzx = np.zeros_like(r2, dtype=np.float64)
    Kzy = np.zeros_like(r2, dtype=np.float64)

    mask = r2 > 0.0
    Kx[mask] = c * (-Y[mask] / r2[mask])
    Ky[mask] = c * (+X[mask] / r2[mask])
    Kzx[mask] = c * (+Y[mask] / r2[mask])
    Kzy[mask] = c * (-X[mask] / r2[mask])

    # Put the zero displacement at index (0,0) before FFT convolution.
    Kx_hat = np.fft.fft2(np.fft.ifftshift(Kx))
    Ky_hat = np.fft.fft2(np.fft.ifftshift(Ky))
    Kzx_hat = np.fft.fft2(np.fft.ifftshift(Kzx))
    Kzy_hat = np.fft.fft2(np.fft.ifftshift(Kzy))

    return Kx_hat, Ky_hat, Kzx_hat, Kzy_hat, npx, npy


def winding_fd_fast(u, dx, dy, pad_factor=2, kernels=None):
    """
    FFT-accelerated sampled-kernel free-space winding-gauge calculation.

    Parameters
    ----------
    u : ndarray
        Shape (nx, ny, 3) or (nx, ny, ns, 3), with components [ux, uy, uz].
    dx, dy : float
        Grid spacings.
    pad_factor : int
        Linear-convolution padding factor, normally 2.
    kernels : tuple or None
        Optional kernels returned by _build_rect_kernels_fft. Pass this when
        calling repeatedly for the same nx, ny, dx, dy.

    Returns
    -------
    A : ndarray
        Same leading shape as u, with last component [Ax, Ay, Az].

    Notes
    -----
    This implements the same rectangular-rule calculation as
    winding_naive_rect from spectralHelicity.py:

        Ax = sum [-(y-y')/r^2] uz' dxdy/(2*pi)
        Ay = sum [ +(x-x')/r^2] uz' dxdy/(2*pi)
        Az = sum {[(y-y') ux' - (x-x') uy']/r^2} dxdy/(2*pi)

    with the singular self contribution set to zero.
    """
    u = np.asarray(u, dtype=np.float64)
    squeeze_slice = False
    if u.ndim == 3:
        u = u[..., None, :]
        squeeze_slice = True
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError(f"u must have shape (nx, ny, 3) or (nx, ny, ns, 3); got {u.shape}")

    nx, ny, ns, _ = u.shape
    if kernels is None:
        kernels = _build_rect_kernels_fft(nx, ny, dx, dy, pad_factor=pad_factor)

    Kx_hat, Ky_hat, Kzx_hat, Kzy_hat, npx, npy = kernels

    def pad2(a):
        out = np.zeros((npx, npy), dtype=np.float64)
        out[:nx, :ny] = a
        return out

    A = np.zeros_like(u, dtype=np.float64)
    for s in range(ns):
        Ux = np.fft.fft2(pad2(u[:, :, s, 0]))
        Uy = np.fft.fft2(pad2(u[:, :, s, 1]))
        Uz = np.fft.fft2(pad2(u[:, :, s, 2]))

        A[:, :, s, 0] = np.fft.ifft2(Uz * Kx_hat).real[:nx, :ny]
        A[:, :, s, 1] = np.fft.ifft2(Uz * Ky_hat).real[:nx, :ny]
        A[:, :, s, 2] = np.fft.ifft2(Ux * Kzx_hat + Uy * Kzy_hat).real[:nx, :ny]

    return A[:, :, 0, :] if squeeze_slice else A



# ---------- full free-space winding gauge (split mean + fluctuations, all free-space) ----------
def winding_gauge_free_space_split(u, dx, dy, Wx=None, Wy=None,
                                   pad_factor=2, use_discrete_symbol=True, include_const=False,
                                   active_mask=None):
    """
    Compute free-space A^W(x,y, slice) for u(x,y,slice) via zero-padded FFTs.
    *Each slice is independent; the mean <u_z> is computed per-slice (no assumption of equality across slices).*

    A^W(x,y) = (1/2π) ∫_S [ u(x',y') × r / |r|^2 ] dA',  r=(x-x',y-y',0)

    We split: u = u' + u_c with u_c = (0,0,<u_z>) and u'_z = u_z - <u_z>.
    Then A^W[u] = A^W[u'] + <u_z>*W, where W is precomputed once for the geometry.
    The fluctuation part A^W[u'] is also computed as *free-space* via zero-padding (not torus).

    Parameters
    ----------
    u : array, shape (Nx, Ny, Ns, 3)  # Ns = number of slices (e.g., times)
        Components order [ux, uy, uz] along the last axis.
    dx, dy : float
        Grid spacings.
    Wx, Wy : optional precomputed geometry fields from precompute_W_rect (shape Nx×Ny).
    pad_factor : int
        Zero-padding factor (>=2 recommended). Larger reduces wrap-around from periodic images.
    use_discrete_symbol : bool
        Use discrete symbols (recommended to match trapezoid rule). Else continuous symbols.
    include_const : bool
        Include the 1/(2π) prefactor in the spectral symbols.

    Returns
    -------
    A : array, shape (Nx, Ny, Ns, 3)
        Free-space winding-gauge vector potential per slice.
    """
    u = np.asarray(u)
    Nx, Ny, Ns, _ = u.shape
    if Wx is None or Wy is None:
        Wx, Wy = precompute_W_rect(Nx, Ny, dx, dy,
                                   pad_factor=pad_factor,
                                   use_discrete_symbol=use_discrete_symbol,
                                   include_const=include_const)

    # Padded sizes and symbols
    Npx, Npy = pad_factor*Nx, pad_factor*Ny
    kx = 2*np.pi * np.fft.fftfreq(Npx, d=dx)
    ky = 2*np.pi * np.fft.fftfreq(Npy, d=dy)
    kx2d, ky2d = np.meshgrid(kx, ky, indexing='ij')

    if use_discrete_symbol:
        kx_t = np.sin(kx2d*dx)/dx
        ky_t = np.sin(ky2d*dy)/dy
        lap_t = 4*np.sin(0.5*kx2d*dx)**2/dx**2 + 4*np.sin(0.5*ky2d*dy)**2/dy**2
        inv_lap = np.zeros_like(lap_t)
        m = lap_t > 0
        inv_lap[m] = 1.0/lap_t[m]
        Kx_uZ =  1j * ky_t * inv_lap                 # maps Uz' -> Ax_hat
        Ky_uZ = -1j * kx_t * inv_lap                 # maps Uz' -> Ay_hat
        # For Az: i*(kx_t*Uy - ky_t*Ux)*inv_lap
        Az_symbol = 1j * inv_lap
    else:
        k2 = kx2d**2 + ky2d**2
        inv_k2 = np.zeros_like(k2)
        m = k2 > 0
        inv_k2[m] = 1.0/k2[m]
        Kx_uZ =  1j * ky2d * inv_k2
        Ky_uZ = -1j * kx2d * inv_k2
        Az_symbol = 1j * inv_k2

    if include_const:
        const = 1.0/(2*np.pi)
        Kx_uZ *= const
        Ky_uZ *= const
        Az_symbol = const * Az_symbol

    # Helper: pad to padded domain (upper-left placement)
    def pad2(a):
        out = np.zeros((Npx, Npy), dtype=float)
        out[:Nx, :Ny] = a
        return out

    A = np.zeros_like(u, dtype=float)

    for s in range(Ns):
        ux = u[:, :, s, 0]
        uy = u[:, :, s, 1]
        uz = u[:, :, s, 2]

        # Per-slice mean (no cross-slice assumption). When an active mask is
        # supplied, compute the mean only over active pixels so weak-field / cut-off
        # regions do not bias the mean vertical contribution.
        if active_mask is not None:
            ms = np.asarray(active_mask[:, :, s], dtype=bool)
            if np.any(ms):
                mz = float(uz[ms].mean())
            else:
                mz = 0.0
        else:
            mz = float(uz.mean())
        uzp = uz - mz  # fluctuation with zero slice-mean

        # Pad fields (free-space)
        Ux = np.fft.fft2(pad2(ux))
        Uy = np.fft.fft2(pad2(uy))
        Uzp = np.fft.fft2(pad2(uzp))

        # Spectral multipliers (free-space, padded)
        Ax_hat = Kx_uZ * Uzp
        Ay_hat = Ky_uZ * Uzp
        Az_hat = Az_symbol * (kx2d * Uy - ky2d * Ux) if not use_discrete_symbol \
                 else Az_symbol * ( (np.sin(kx2d*dx)/dx) * Uy - (np.sin(ky2d*dy)/dy) * Ux )

        # Back to real, crop to physical domain
        Ax_fs = np.fft.ifft2(Ax_hat).real[:Nx, :Ny]
        Ay_fs = np.fft.ifft2(Ay_hat).real[:Nx, :Ny]
        Az_fs = np.fft.ifft2(Az_hat).real[:Nx, :Ny]

        # Add the geometry contribution of the slice-mean vertical field
        Ax = Ax_fs + mz * Wx
        Ay = Ay_fs + mz * Wy
        Az = Az_fs

        A[:, :, s, 0] = Ax
        A[:, :, s, 1] = Ay
        A[:, :, s, 2] = Az

    return A



def integrate_xy_per_slice(F, dx=360, dy=360, weight_kind="trapz", mask=None, return_mean=False):
    """
    I[z] = ∬ F(x,y,z) dx dy  (one value per slice)
    F shape: (Nx, Ny, Nz)  [first index = x, second = y]
    """
    F = np.asarray(F)
    Nx, Ny, Nz = F.shape

    # planar weights
    if weight_kind == "uniform":
        W = np.ones((Nx, Ny), dtype=F.dtype)
    elif weight_kind == "trapz":
        W = np.ones((Nx, Ny), dtype=F.dtype)
        W[0,:]*=0.5; W[-1,:]*=0.5; W[:,0]*=0.5; W[:,-1]*=0.5
        W[0,0]*=0.5; W[0,-1]*=0.5; W[-1,0]*=0.5; W[-1,-1]*=0.5  # corners -> 1/4
    else:
        raise ValueError("weight_kind must be 'uniform' or 'trapz'.")

    # mask handling
    if mask is None:
        W3 = W[..., None]                         # (Nx,Ny,1)
        areas = np.full(Nz, W.sum() * dx * dy)
        F_eff = F
    else:
        M = np.asarray(mask)
        if M.shape == (Nx, Ny):
            W3 = (W * M)[..., None]
            areas = np.full(Nz, (W * M).sum() * dx * dy)
            F_eff = F
        elif M.shape == (Nx, Ny, Nz):
            W3 = W[..., None] * M
            areas = (W3.reshape(Nx*Ny, Nz).sum(axis=0)) * dx * dy
            F_eff = F
        else:
            raise ValueError("mask must be None, (Nx,Ny), or (Nx,Ny,Nz).")

    I = (F_eff * W3).reshape(Nx*Ny, Nz).sum(axis=0) * dx * dy

    if return_mean:
        mean = np.divide(I, areas, out=np.zeros_like(I), where=areas > 0)
        return I, mean
    return I




def _discover_field_store(field_loc, field_tag):
    p = Path(field_loc)

    if p.suffix == '.zarr' and p.exists():
        return str(p)

    candidates = []
    if p.is_dir():
        candidates.extend([
            p / f"field_data_{field_tag}.zarr",
            p / "Data" / f"field_data_{field_tag}.zarr",
        ])
        candidates.extend(sorted(p.glob("field_data_*.zarr")))
        data_dir = p / "Data"
        if data_dir.is_dir():
            candidates.extend(sorted(data_dir.glob("field_data_*.zarr")))

    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return None


def _read_zarr_field(root, key, idx, nx, ny, dtype=np.float64, allow_missing=False):
    if key not in root:
        if allow_missing:
            return np.zeros((nx, ny), dtype=dtype)
        raise KeyError(f"Dataset {key} not found in Zarr store")
    arr = np.asarray(root[key][:, :, idx], dtype=dtype)
    if arr.shape == (ny, nx):
        arr = arr.T
    if arr.shape != (nx, ny):
        raise ValueError(f"Dataset {key}[..., {idx}] has shape {arr.shape}; expected {(nx, ny)}")
    return arr

def _read_field_file(fname, nx, ny, dtype=np.float64):
    """Read single 2D field file (x-fastest). If missing, return zeros."""
    if not os.path.exists(fname):
        return np.zeros((nx, ny), dtype=dtype)
    flat = np.loadtxt(fname)
    if flat.size != nx * ny:
        raise ValueError(f"{fname} has {flat.size} values; expected {nx*ny}")
    arr = flat.reshape((ny, nx)).T
    if arr.dtype != dtype:
        arr = arr.astype(dtype, copy=False)
    return arr

def read_raw_chunk(k0, k1, nx, ny, field_loc, field_tag, vel_tag):
    """Read slices [k0,k1) from a Zarr field store matching the pipeline layout."""
    kk = k1 - k0
    raw = {name: np.empty((nx, ny, kk), dtype=np.float64) for name in ["vx", "vy", "vz", "bx", "by", "bz", "bxp", "byp"]}
    store_path = _discover_field_store(field_loc, field_tag)
    if store_path is None:
        raise RuntimeError(
            f"Could not find Zarr store for field_tag={field_tag} under {field_loc}. "
            "This loader now expects the Zarr pipeline layout."
        )

    root = zarr.open(store_path, mode='r')
    required = {'vx': 'Ux', 'vy': 'Uy', 'vz': 'Uz', 'bx': 'bx', 'by': 'by', 'bz': 'bz', 'bxp': 'Bxp', 'byp': 'Byp'}
    missing = [dset for dset in required.values() if dset not in root]
    if missing:
        raise KeyError(f"Missing required datasets in {store_path}: {missing}")

    for k, idx in enumerate(range(k0, k1)):
        for key, dset in required.items():
            raw[key][..., k] = _read_zarr_field(root, dset, idx, nx, ny)
    return raw



def build_base_fields(raw, eps_bz=1e-6, eps_b2=1e-5):
    """
    Construct B, corrected V, Ucur, Upot, and total U from raw arrays.

    This follows the C++ cutoff logic:
      - remove the component of v parallel to B only where |B|^2 > 1e-5;
      - form the apparent/emergence velocity terms only where |Bz| > 1e-6;
      - where |Bz| <= 1e-6, set the horizontal components of Ucur/Upot to zero.
    """
    nx, ny, kk = raw["bx"].shape
    B = np.empty((nx, ny, kk, 3), dtype=np.float64)
    V = np.empty_like(B)
    Ucur = np.zeros_like(B)
    Upot = np.zeros_like(B)
    U = np.zeros_like(B)

    bx = raw["bx"].astype(np.float64, copy=False)
    by = raw["by"].astype(np.float64, copy=False)
    bz = raw["bz"].astype(np.float64, copy=False)
    bxp = raw["bxp"].astype(np.float64, copy=False)
    byp = raw["byp"].astype(np.float64, copy=False)

    vx = raw["vx"].astype(np.float64, copy=True)
    vy = raw["vy"].astype(np.float64, copy=True)
    vz = raw["vz"].astype(np.float64, copy=True)

    B[..., 0] = bx
    B[..., 1] = by
    B[..., 2] = bz

    # C++ line-of-sight / parallel-to-B correction:
    # if (bsq > 0.00001) v <- v - (v.B) B / |B|^2, otherwise leave v unchanged.
    B2 = bx*bx + by*by + bz*bz
    vdotB = vx*bx + vy*by + vz*bz
    mask_b2 = B2 > eps_b2

    vx[mask_b2] -= vdotB[mask_b2] * bx[mask_b2] / B2[mask_b2]
    vy[mask_b2] -= vdotB[mask_b2] * by[mask_b2] / B2[mask_b2]
    vz[mask_b2] -= vdotB[mask_b2] * bz[mask_b2] / B2[mask_b2]

    V[..., 0] = vx
    V[..., 1] = vy
    V[..., 2] = vz

    bx_cur = bx - bxp
    by_cur = by - byp

    mask_bz = np.abs(bz) > eps_bz

    rvx_cur = np.zeros_like(bz, dtype=np.float64)
    rvy_cur = np.zeros_like(bz, dtype=np.float64)
    rvx_pot = np.zeros_like(bz, dtype=np.float64)
    rvy_pot = np.zeros_like(bz, dtype=np.float64)

    rvx_cur[mask_bz] = vz[mask_bz] * bx_cur[mask_bz] / bz[mask_bz]
    rvy_cur[mask_bz] = vz[mask_bz] * by_cur[mask_bz] / bz[mask_bz]
    rvx_pot[mask_bz] = vz[mask_bz] * bxp[mask_bz] / bz[mask_bz]
    rvy_pot[mask_bz] = vz[mask_bz] * byp[mask_bz] / bz[mask_bz]

    # C++ behaviour: if |Bz| <= 1e-6, these horizontal U components stay zero.
    Ucur[..., 0][mask_bz] = vx[mask_bz] - rvx_cur[mask_bz]
    Ucur[..., 1][mask_bz] = vy[mask_bz] - rvy_cur[mask_bz]
    Ucur[..., 2] = 1.0

    Upot[..., 0][mask_bz] = vx[mask_bz] - rvx_pot[mask_bz]
    Upot[..., 1][mask_bz] = vy[mask_bz] - rvy_pot[mask_bz]
    Upot[..., 2] = 1.0

    # Total field-line velocity as in the C++ output combination:
    # total = current + potential - velocity-only.
    U[..., 0] = Ucur[..., 0] + Upot[..., 0] - V[..., 0]
    U[..., 1] = Ucur[..., 1] + Upot[..., 1] - V[..., 1]
    U[..., 2] = 1.0

    return {
        "B": B,
        "V": V,
        "Ucur": Ucur,
        "Upot": Upot,
        "U": U,
    }

def compute_all(base, steps=(360.0, 360.0, 1.0), cutoff=50.0):
    """
    Compute helicityFlux, windingFlux, dLFlux, dHFlux for a base dict:
    base['Ucur'], base['Upot'], base['B'], base['U'] shaped (nx, ny, kk, 3).

    NOTE: This function now uses `winding_fd_fast`, i.e. the sampled real-space
    rectangular-rule winding-gauge kernel accelerated by FFT. This replaces the
    previous mean/fluctuation `winding_gauge_free_space_split` route.
    """
    dx, dy, dz = steps
    nx_, ny_, kk, _ = base['Ucur'].shape

    # ensure float64
    for k in ('Ucur', 'Upot', 'B', 'U'):
        if base[k].dtype != np.float64:
            base[k] = base[k].astype(np.float64, copy=False)

    kernels = _build_rect_kernels_fft(nx_, ny_, dx, dy, pad_factor=2)
    work = np.empty((nx_, ny_, kk, 3), dtype=np.float64)
    bx = base["B"][..., 0]
    by = base["B"][..., 1]
    bz = base["B"][..., 2]

    bmag = np.sqrt(bx * bx + by * by + bz * bz, dtype=np.float64)
    active = bmag > float(cutoff)
    active_f = active.astype(np.float64)
    sign = np.where(bz > 0.0, 1.0, np.where(bz < 0.0, -1.0, 0.0)).astype(np.float64)
    sigma = sign * active_f
    bz_masked = bz * active_f

    # Winding (current)
    work[..., 0] = base['Ucur'][..., 0] * sigma
    work[..., 1] = base['Ucur'][..., 1] * sigma
    work[..., 2] = sigma
    cleanedA = winding_fd_fast(work, dx, dy, pad_factor=2, kernels=kernels)
    windCurDen = getFLHDenSingle(cleanedA, work, signed=True).astype(np.float64)
    del cleanedA

    # Winding (potential)
    work[..., 0] = base['Upot'][..., 0] * sigma
    work[..., 1] = base['Upot'][..., 1] * sigma
    work[..., 2] = sigma
    cleanedA = winding_fd_fast(work, dx, dy, pad_factor=2, kernels=kernels)
    windPotDen = getFLHDenSingle(cleanedA, work, signed=True).astype(np.float64)
    del cleanedA

    # Winding (total)
    work[..., 0] = base['U'][..., 0] * sigma
    work[..., 1] = base['U'][..., 1] * sigma
    work[..., 2] = sigma
    cleanedA = winding_fd_fast(work, dx, dy, pad_factor=2, kernels=kernels)
    windDen = getFLHDenSingle(cleanedA, work, signed=True).astype(np.float64)
    del cleanedA

    # Helicity (current)
    work[..., 0] = base['Ucur'][..., 0] * bz_masked
    work[..., 1] = base['Ucur'][..., 1] * bz_masked
    work[..., 2] = bz_masked
    cleanedA = winding_fd_fast(work, dx, dy, pad_factor=2, kernels=kernels)
    helCurDen = getFLHDenSingle(cleanedA, work, signed=True).astype(np.float64)
    del cleanedA

    # Helicity (potential)
    work[..., 0] = base['Upot'][..., 0] * bz_masked
    work[..., 1] = base['Upot'][..., 1] * bz_masked
    work[..., 2] = bz_masked
    cleanedA = winding_fd_fast(work, dx, dy, pad_factor=2, kernels=kernels)
    helPotDen = getFLHDenSingle(cleanedA, work, signed=True).astype(np.float64)
    del cleanedA

    # Helicity (total)
    work[..., 0] = base['U'][..., 0] * bz_masked
    work[..., 1] = base['U'][..., 1] * bz_masked
    work[..., 2] = bz_masked
    cleanedA = winding_fd_fast(work, dx, dy, pad_factor=2, kernels=kernels)
    helDen = getFLHDenSingle(cleanedA, work, signed=True).astype(np.float64)
    del cleanedA

    dHFlux = np.abs(helCurDen) - np.abs(helPotDen)
    dLFlux = np.abs(windCurDen) - np.abs(windPotDen)

    # free memory of intermediates
    del work, helCurDen, helPotDen, windCurDen, windPotDen

    return {
        "helicityFlux": -helDen,
        "windingFlux":  -windDen,
        "dLFlux":       dLFlux,
        "dHFlux":       dHFlux,
    }




def _save_vector_field_txt(output_dir, basename, region, step, F):
    """
    Save a vector field F shaped (nx, ny, 3) in j-then-i order, matching the
    usual C++ flattening convention used elsewhere in the pipeline.
    """
    os.makedirs(output_dir, exist_ok=True)
    fname = os.path.join(output_dir, f"{basename}_{region}_{step}.txt")
    nx, ny, _ = F.shape
    with open(fname, "w") as f:
        for j in range(ny):
            for i in range(nx):
                fx, fy, fz = F[i, j]
                f.write(f"{fx:.16e} {fy:.16e} {fz:.16e}\n")


def _process_chunk(args):
    """
    Worker-callable function for imap_unordered: args is a tuple
    (k0, k1, nx, ny, field_loc, field_tag, vel_tag, steps, cutoff, debug_save_fields, debug_field_dir)
    Returns (k0, k1, slabs_dict).
    """
    k0, k1, nx, ny, field_loc, field_tag, vel_tag, steps, cutoff, debug_save_fields, debug_field_dir = args

    # read raw data for this chunk
    raw = read_raw_chunk(k0, k1, nx, ny, field_loc, field_tag, vel_tag)
    base = build_base_fields(raw)

    if debug_save_fields:
        for local_k, step in enumerate(range(k0, k1)):
            _save_vector_field_txt(debug_field_dir, "photosphereVField_py", field_tag, step, base["V"][:, :, local_k, :])
            _save_vector_field_txt(debug_field_dir, "photosphereUCurField_py", field_tag, step, base["Ucur"][:, :, local_k, :])
            _save_vector_field_txt(debug_field_dir, "photosphereUPotField_py", field_tag, step, base["Upot"][:, :, local_k, :])

    slabs = compute_all(base, steps=steps, cutoff=cutoff)

    return k0, k1, slabs


def process_field_data(
    param_file,
    ck=64,
    field_loc="AR_377_test_Input",
    field_tag="377",
    vel_tag="20",
    savdir="AR_377_test_Output",
    zarr_file="topology.zarr",
    steps=(360.0, 360.0, 1.0),
    n_workers=None,
    debug_save_fields=False,
    debug_field_dir=None,
):
    """
    Full topology-processing version.

    Reads raw field/velocity chunks, builds the corrected velocity fields, computes
    the full topology density slabs using compute_all(...), writes the resulting
    3-D density arrays to a Zarr store, and saves per-slice integrated totals.
    """

    with open(param_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    try:
        nx = int(lines[1])
        ny = int(lines[2])
        nz = int(lines[3])
    except (IndexError, ValueError):
        raise ValueError("Invalid parameter file; expected line 2=nx, line 3=ny, line 4=nz")

    dx, dy, dz = steps

    # The cutoff normally lives on line 5 of specifications.txt. Keep the function
    # argument as the default/fallback, but use the file value when present.
    try:
        cutoff = float(lines[4])
    except (IndexError, ValueError):
        cutoff = float(cutoff)

    if n_workers is None:
        n_workers = max(1, min(cpu_count(), int(np.ceil(nz / max(1, ck)))))
    else:
        n_workers = max(1, int(n_workers))

    if debug_save_fields and debug_field_dir is None:
        debug_field_dir = os.path.join(savdir, "debug_vector_fields")

    print(
        f"Topology mode: nx={nx}, ny={ny}, nz={nz}, ck={ck}, "
        f"cutoff={cutoff}, workers={n_workers}"
    )

    os.makedirs(savdir, exist_ok=True)
    zarr_path = os.path.join(savdir, zarr_file)
    if os.path.exists(zarr_path):
        shutil.rmtree(zarr_path)

    root = zarr.open(zarr_path, mode="w", zarr_format=2)
    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)

    output_names = ["helicityFlux", "windingFlux", "dLFlux", "dHFlux"]
    arrays = {
        name: root.create_array(
            name,
            shape=(nx, ny, nz),
            chunks=(nx, ny, min(ck, nz)),
            dtype="f8",
            compressor=compressor,
        )
        for name in output_names
    }

    root.attrs["field_tag"] = str(field_tag)
    root.attrs["vel_tag"] = str(vel_tag)
    root.attrs["nx"] = int(nx)
    root.attrs["ny"] = int(ny)
    root.attrs["nz"] = int(nz)
    root.attrs["ck"] = int(ck)
    root.attrs["dx"] = float(dx)
    root.attrs["dy"] = float(dy)
    root.attrs["dz"] = float(dz)
    root.attrs["cutoff"] = float(cutoff)
    root.attrs["outputs"] = output_names

    totals = {name: np.zeros(nz, dtype=np.float64) for name in output_names}

    chunk_args = [
        (k0, min(k0 + ck, nz), nx, ny, field_loc, field_tag, vel_tag,
         steps, cutoff, debug_save_fields, debug_field_dir)
        for k0 in range(0, nz, ck)
    ]

    if n_workers == 1:
        iterator = map(_process_chunk, chunk_args)
        results_iter = tqdm(iterator, total=len(chunk_args), desc="Topology chunks", unit="chunk")
    else:
        pool = Pool(processes=n_workers)
        results_iter = tqdm(
            pool.imap_unordered(_process_chunk, chunk_args),
            total=len(chunk_args),
            desc="Topology chunks",
            unit="chunk",
        )

    try:
        for k0, k1, slabs in results_iter:
            print(f"Writing topology chunk {k0}:{k1}")
            for name in output_names:
                slab = slabs[name].astype(np.float64, copy=False)
                arrays[name][:, :, k0:k1] = slab

                # C++ totals use a rectangular outer integration, so use uniform
                # weights here rather than trapezoidal edge weights.
                totals[name][k0:k1] = integrate_xy_per_slice(
                    slab, dx=dx, dy=dy, weight_kind="uniform"
                )

            root.attrs["last_completed_chunk"] = [int(k0), int(k1)]
    finally:
        if n_workers != 1:
            pool.close()
            pool.join()

    # Save totals both inside the Zarr store and as standalone .npy files.
    for name in output_names:
        root.create_array(
            f"{name}_totals",
            data=totals[name],
            chunks=(min(ck, nz),),
            dtype="f8",
            compressor=compressor,
        )
        np.save(os.path.join(savdir, f"{name}_totals.npy"), totals[name])

    print(f"Topology densities saved to: {zarr_path}")
    print(f"Per-slice totals saved in: {savdir}")
    return zarr_path

def write_config_file(
    filepath,
    region_number=377,
    start_year=2011, start_month=2, start_day=11, start_hour=23,
    end_year=2011, end_month=2, end_day=21, end_hour=23,
    velocity_smoothing=20,
    input_directory="/extra/tmp/ktch24_prs/ARTop-main/AR_377_New_Input",
    output_directory="/extra/tmp/ktch24_prs/ARTop-main/AR_377_New_Output",
    cutoff=20,
    sampling=1,
    registered_email="christopher.prior@durham.ac.uk",
    download_data=True,
    topology=True,
    remove_downloaded_images=False
):
    """Write a configuration file with the given parameters."""
    lines = [
        f"Region number={region_number}",
        f"Download data={'true' if download_data else 'false'}",
        f"Start year={start_year}",
        f"Start month={start_month:02d}",
        f"Start day={start_day:02d}",
        f"Start hour={start_hour:02d}",
        f"End year={end_year}",
        f"End month={end_month:02d}",
        f"End day={end_day:02d}",
        f"End hour={end_hour:02d}",
        f"Velocity smoothing={velocity_smoothing}",
        f"Input directory={input_directory}",
        f"Output directory={output_directory}",
        f"Topology={'true' if topology else 'false'}",
        f"Cutoff={cutoff}",
        f"Sampling={sampling}",
        f"Remove downloaded images={'true' if remove_downloaded_images else 'false'}",
        f"Registered email={registered_email}"
    ]

    with open(filepath, "w") as f:
        f.write("\n".join(lines))

    print(f"Configuration file written to: {filepath}")

