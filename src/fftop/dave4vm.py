import argparse
from pathlib import Path

import numpy as np
import zarr
from numba import njit, prange


@njit(cache=True)
def _apply_filter_globally_nb(xfilter, yfilter, quantity, midpoint):
    nx, ny = quantity.shape
    nwx = xfilter.shape[0]
    nwy = yfilter.shape[0]
    xfiltered = np.zeros((nx, ny), dtype=np.float64)
    out = np.zeros((nx, ny), dtype=np.float64)

    shiftx = midpoint - 1
    shifty = midpoint - 1

    for i in range(nx):
        for j in range(ny):
            s = 0.0
            for k in range(nwx):
                xi = i + k - shiftx
                if 0 <= xi < nx:
                    s += xfilter[k] * quantity[xi, j]
            xfiltered[i, j] = s

    for i in range(nx):
        for j in range(ny):
            s = 0.0
            for k in range(nwy):
                yj = j + k - shifty
                if 0 <= yj < ny:
                    s += yfilter[k] * xfiltered[i, yj]
            out[i, j] = s

    return out


@njit(cache=True)
def _gaussian_solve_9x9_nb(A, b):
    M = np.empty((9, 10), dtype=np.float64)
    for i in range(9):
        for j in range(9):
            M[i, j] = A[i, j]
        M[i, 9] = b[i]

    for col in range(9):
        pivot = col
        maxv = abs(M[col, col])
        for row in range(col + 1, 9):
            v = abs(M[row, col])
            if v > maxv:
                maxv = v
                pivot = row

        if maxv < 1e-12:
            return np.zeros(9, dtype=np.float64), False

        if pivot != col:
            for j in range(col, 10):
                tmp = M[col, j]
                M[col, j] = M[pivot, j]
                M[pivot, j] = tmp

        piv = M[col, col]
        for j in range(col, 10):
            M[col, j] /= piv

        for row in range(9):
            if row != col:
                fac = M[row, col]
                if fac != 0.0:
                    for j in range(col, 10):
                        M[row, j] -= fac * M[col, j]

    x = np.empty(9, dtype=np.float64)
    for i in range(9):
        x[i] = M[i, 9]
    return x, True


@njit(parallel=True, cache=True)
def _solve_velocity_systems_nb(mats, rhs, mask):
    nx, ny = mask.shape
    vx = np.zeros((nx, ny), dtype=np.float64)
    vy = np.zeros((nx, ny), dtype=np.float64)
    vz = np.zeros((nx, ny), dtype=np.float64)

    for i in prange(nx):
        for j in range(ny):
            if mask[i, j]:
                sol, ok = _gaussian_solve_9x9_nb(mats[i, j], rhs[i, j])
                if ok:
                    vx[i, j] = sol[0]
                    vy[i, j] = sol[1]
                    vz[i, j] = sol[6]
    return vx, vy, vz


@njit(cache=True)
def _derivatives_periodic_nb(Bx, By, Bz, dx, dy):
    nx, ny = Bx.shape
    Bxdx = np.empty((nx, ny), dtype=np.float64)
    Bydx = np.empty((nx, ny), dtype=np.float64)
    Bzdx = np.empty((nx, ny), dtype=np.float64)
    Bxdy = np.empty((nx, ny), dtype=np.float64)
    Bydy = np.empty((nx, ny), dtype=np.float64)
    Bzdy = np.empty((nx, ny), dtype=np.float64)

    C2 = 0.12019
    C1 = 0.74038

    for i in range(nx):
        ipp = i + 2
        if ipp > nx - 1:
            ipp -= nx
        ip = i + 1
        if ip > nx - 1:
            ip -= nx
        imm = i - 2
        if imm < 0:
            imm += nx
        im = i - 1
        if im < 0:
            im += nx
        for j in range(ny):
            jpp = j + 2
            if jpp > ny - 1:
                jpp -= ny
            jp = j + 1
            if jp > ny - 1:
                jp -= ny
            jmm = j - 2
            if jmm < 0:
                jmm += ny
            jm = j - 1
            if jm < 0:
                jm += ny

            Bxdx[i, j] = (C1 * (Bx[ip, j] - Bx[im, j]) - C2 * (Bx[ipp, j] - Bx[imm, j])) / dx
            Bydx[i, j] = (C1 * (By[ip, j] - By[im, j]) - C2 * (By[ipp, j] - By[imm, j])) / dx
            Bzdx[i, j] = (C1 * (Bz[ip, j] - Bz[im, j]) - C2 * (Bz[ipp, j] - Bz[imm, j])) / dx
            Bxdy[i, j] = (C1 * (Bx[i, jp] - Bx[i, jm]) - C2 * (Bx[i, jpp] - Bx[i, jmm])) / dy
            Bydy[i, j] = (C1 * (By[i, jp] - By[i, jm]) - C2 * (By[i, jpp] - By[i, jmm])) / dy
            Bzdy[i, j] = (C1 * (Bz[i, jp] - Bz[i, jm]) - C2 * (Bz[i, jpp] - Bz[i, jmm])) / dy

    return Bxdx, Bydx, Bzdx, Bxdy, Bydy, Bzdy


class Dave4VMNumba:
    def __init__(self, dx: float, dy: float, nx: int, ny: int):
        self.dx = float(dx)
        self.dy = float(dy)
        self.nx = int(nx)
        self.ny = int(ny)

    def get_derivatives(self, bx: np.ndarray, by: np.ndarray, bz: np.ndarray) -> None:
        self.Bx = np.ascontiguousarray(bx, dtype=np.float64)
        self.By = np.ascontiguousarray(by, dtype=np.float64)
        self.Bz = np.ascontiguousarray(bz, dtype=np.float64)
        (
            self.Bxdx,
            self.Bydx,
            self.Bzdx,
            self.Bxdy,
            self.Bydy,
            self.Bzdy,
        ) = _derivatives_periodic_nb(self.Bx, self.By, self.Bz, self.dx, self.dy)

    def weight_functions(self, window_size: int) -> None:
        self.nw = 2 * int(window_size // 2) + 1
        self.midpoint = ((self.nw - 1) // 2) + 1
        i = np.arange(self.nw, dtype=np.float64)
        rel = i - (self.midpoint - 1)
        self.filterMat1D = np.full(self.nw, 1.0 / float(self.nw), dtype=np.float64)
        self.filterMatX1D = self.dx * rel / float(self.nw)
        self.filterMatXSq1D = (self.dx ** 2) * rel * rel / float(self.nw)
        self.filterMatY1D = self.dy * rel / float(self.nw)
        self.filterMatYSq1D = (self.dy ** 2) * rel * rel / float(self.nw)

    def apply_filter_globally(self, xfilter: np.ndarray, yfilter: np.ndarray, quantity: np.ndarray) -> np.ndarray:
        return _apply_filter_globally_nb(
            np.ascontiguousarray(xfilter, dtype=np.float64),
            np.ascontiguousarray(yfilter, dtype=np.float64),
            np.ascontiguousarray(quantity, dtype=np.float64),
            self.midpoint,
        )

    def set_matrix_terms_pre_filter(self, bxdt: np.ndarray, bydt: np.ndarray, bzdt: np.ndarray) -> None:
        self.Bzdt = np.ascontiguousarray(bzdt, dtype=np.float64)

        self.Bzsq = self.Bz * self.Bz
        self.BzBzdx = self.Bz * self.Bzdx
        self.BzBzdy = self.Bz * self.Bzdy
        self.BzdxBzdy = self.Bzdx * self.Bzdy
        self.Bzdxsq = self.Bzdx * self.Bzdx
        self.Bzdysq = self.Bzdy * self.Bzdy
        self.BxBzdx = self.Bx * self.Bzdx
        self.BxBzdy = self.Bx * self.Bzdy
        self.ByBzdx = self.By * self.Bzdx
        self.ByBzdy = self.By * self.Bzdy

        self.BxBz = self.Bx * self.Bz
        self.ByBz = self.By * self.Bz
        self.BzBy = self.ByBz.copy()  # preserves the original C++ naming quirk
        self.Bysq = self.By * self.By
        self.Bxsq = self.Bx * self.Bx
        self.BxdxBzdx = self.Bxdx * self.Bzdx
        self.BxdxBzdy = self.Bxdx * self.Bzdy
        self.BzBxdy = self.Bz * self.Bxdy
        self.BydxBzdx = self.Bydx * self.Bzdx
        self.BydxBzdy = self.Bydx * self.Bzdy
        self.BzBydy = self.Bz * self.Bydy
        self.BxBxdx = self.Bx * self.Bxdx
        self.BxBxdy = self.Bx * self.Bxdy
        self.ByBydx = self.By * self.Bydx
        self.ByBydy = self.By * self.Bydy
        self.Bxdxsq = self.Bxdx * self.Bxdx

        self.BzdtBzdx = self.Bzdt * self.Bzdx
        self.BzdtBzdy = self.Bzdt * self.Bzdy
        self.BzBzdt = self.Bz * self.Bzdt
        self.BzdtBxdx = self.Bzdt * self.Bxdx
        self.Bzdtsq = self.Bzdt * self.Bzdt
        self.BzBxdx = self.Bz * self.Bxdx

        self.BxBy = self.Bx * self.By
        self.ByBxdx = self.By * self.Bxdx
        self.BxBzdt = self.Bx * self.Bzdt
        self.ByBzdt = self.By * self.Bzdt

        self.BzdxBydy = self.Bzdx * self.Bydy
        self.BzdyBydy = self.Bzdy * self.Bydy
        self.BydyBzdx = self.BzdxBydy.copy()
        self.BydyBzdy = self.BzdyBydy.copy()
        self.BxdxBydy = self.Bxdx * self.Bydy
        self.Bydysq = self.Bydy * self.Bydy
        self.BzBydz = self.Bz * self.BzBy
        self.BxBydy = self.Bx * self.Bydy
        self.BzdtBydy = self.Bzdt * self.Bydy

    def compose_matrix_and_invert(self):
        AF = self.apply_filter_globally
        f1 = self.filterMat1D
        fx = self.filterMatX1D
        fy = self.filterMatY1D
        fxx = self.filterMatXSq1D
        fyy = self.filterMatYSq1D

        BzdxsqSm = AF(f1, f1, self.Bzdxsq)
        BzdysqSm = AF(f1, f1, self.Bzdysq)
        BzdxBzdySm = AF(f1, f1, self.BzdxBzdy)
        Bzdxsqxd = AF(fx, f1, self.Bzdxsq)
        Bzdxsqyd = AF(f1, fy, self.Bzdxsq)
        Bzdysqxd = AF(fx, f1, self.Bzdysq)
        Bzdysqyd = AF(f1, fy, self.Bzdysq)
        BzBzdxSm = AF(f1, f1, self.BzBzdx)
        BzBzdySm = AF(f1, f1, self.BzBzdy)
        BzBzdxxd = AF(fx, f1, self.BzBzdx)
        BzBzdyxd = AF(fx, f1, self.BzBzdy)
        BzBzdxyd = AF(f1, fy, self.BzBzdx)
        BzBzdyyd = AF(f1, fy, self.BzBzdy)
        BzdxBzdyxd = AF(fx, f1, self.BzdxBzdy)
        BzdxBzdyyd = AF(f1, fy, self.BzdxBzdy)
        BzsqSm = AF(f1, f1, self.Bzsq)
        Bzdxsqxdsq = AF(fxx, f1, self.Bzdxsq)
        BzdxBzdyxdyd = AF(fx, fy, self.BzdxBzdy)
        BzdxBzdyydSq = AF(f1, fyy, self.BzdxBzdy)
        BzdxBzdyxdSq = AF(fxx, f1, self.BzdxBzdy)
        Bzdxsqydsq = AF(f1, fyy, self.Bzdxsq)
        Bzdysqydsq = AF(f1, fyy, self.Bzdysq)
        Bzdxsqxdyd = AF(fx, fy, self.Bzdxsq)
        Bzdysqxdyd = AF(fx, fy, self.Bzdysq)
        BzBzdxxdsq = AF(fxx, f1, self.BzBzdx)
        BzBzdyxdsq = AF(fxx, f1, self.BzBzdy)
        BzBzdxydsq = AF(f1, fyy, self.BzBzdx)
        BzBzdyydsq = AF(f1, fyy, self.BzBzdy)
        BzdxBzdyydsq = AF(f1, fyy, self.BzdxBzdy)
        BzdxBzdyxdsq = AF(fxx, f1, self.BzdxBzdy)
        Bzdysqxdsq = AF(fxx, f1, self.Bzdysq)
        ByBzdxSm = AF(f1, f1, self.ByBzdx)
        ByBzdySm = AF(f1, f1, self.ByBzdy)

        G00 = BzdxsqSm.copy()
        G10 = BzdxBzdySm.copy()
        G11 = BzdysqSm.copy()
        G20 = BzBzdxSm + Bzdxsqxd
        G21 = BzBzdySm + BzdxBzdyxd
        G22 = BzsqSm + 2.0 * BzBzdxxd + Bzdxsqxdsq
        G30 = BzBzdxSm + BzdxBzdyyd
        G31 = BzBzdySm + Bzdysqyd
        G32 = BzsqSm + BzBzdxxd + BzBzdyyd + BzdxBzdyxdyd
        G33 = BzsqSm + 2.0 * BzBzdyyd + Bzdysqydsq
        G40 = Bzdxsqyd.copy()
        G41 = BzdxBzdyyd.copy()
        G42 = BzBzdxyd + Bzdxsqxdyd
        G43 = BzBzdxyd + BzdxBzdyydSq
        G44 = Bzdxsqydsq.copy()
        G50 = BzdxBzdyxd.copy()
        G51 = Bzdysqxd.copy()
        G52 = BzBzdyxd + BzdxBzdyxdSq
        G53 = BzBzdyxd + Bzdysqxdyd
        G54 = BzdxBzdyxdyd.copy()
        G55 = Bzdysqxdsq.copy()

        BxdxBzdxSm = AF(f1, f1, self.BxdxBzdx)
        BxdxBzdySm = AF(f1, f1, self.BxdxBzdy)
        BzBxdxSm = AF(f1, f1, self.BzBxdx)
        BzBxdySm = AF(f1, f1, self.BzBxdy)
        BxdxBzdxxd = AF(fx, f1, self.BxdxBzdx)
        BxdxBzdyxd = AF(fx, f1, self.BxdxBzdy)
        BxdxBzdxyd = AF(f1, fy, self.BxdxBzdx)
        BxdxBzdyyd = AF(f1, fy, self.BxdxBzdy)
        BxdxsqSm = AF(f1, f1, self.Bxdxsq)
        BxBzdxSm = AF(f1, f1, self.BxBzdx)
        BxBzdySm = AF(f1, f1, self.BxBzdy)
        BxBzSm = AF(f1, f1, self.BxBz)
        ByBzSm = AF(f1, f1, self.ByBz)
        BxBySm = AF(f1, f1, self.BxBy)
        BxBzdyyd = AF(f1, fy, self.BxBzdy)
        BxBzdyxd = AF(fx, f1, self.BxBzdy)
        BxBzdxyd = AF(f1, fy, self.BxBzdx)
        BxBzdxxd = AF(fx, f1, self.BxBzdx)
        BxdxBzdxxdyd = AF(fx, fy, self.BxdxBzdx)
        BxdxBzdyxdyd = AF(fx, fy, self.BxdxBzdy)
        BzBxdxxd = AF(fx, f1, self.BzBxdx)
        BzBxdxyd = AF(f1, fy, self.BzBxdx)
        BxdxBzdxxdsq = AF(fxx, f1, self.BxdxBzdx)
        BxdxBzdyxdsq = AF(fxx, f1, self.BxdxBzdy)
        BxdxBzdxydsq = AF(f1, fyy, self.BxdxBzdx)
        BxdxBzdyydsq = AF(f1, fyy, self.BxdxBzdy)
        BxBxdxSm = AF(f1, f1, self.BxBxdx)
        BxBxdxxd = AF(fx, f1, self.BxBxdx)
        BxBxdxyd = AF(f1, fy, self.BxBxdx)
        BxBxdyyd = AF(f1, fy, self.BxBxdy)
        Bxdxsqxd = AF(fx, f1, self.Bxdxsq)
        Bxdxsqxdyd = AF(fx, fy, self.Bxdxsq)
        Bxdxsqxdsq = AF(fxx, f1, self.Bxdxsq)
        Bxdxsqydsq = AF(f1, fyy, self.Bxdxsq)
        BxsqSm = AF(f1, f1, self.Bxsq)
        BysqSm = AF(f1, f1, self.Bysq)
        ByBxdxSm = AF(f1, f1, self.ByBxdx)
        ByBxdxxd = AF(fx, f1, self.ByBxdx)
        ByBxdxyd = AF(f1, fy, self.ByBxdx)
        ByBzdxxd = AF(fx, f1, self.ByBzdx)
        ByBzdyyd = AF(f1, fy, self.ByBzdy)
        ByBzdxyd = AF(f1, fy, self.ByBzdx)
        ByBzdyxd = AF(fx, f1, self.ByBzdy)
        Bxdxsqyd = AF(f1, fy, self.Bxdxsq)
        BzdxBydySm = AF(f1, f1, self.BzdxBydy)
        BydyBzdySm = AF(f1, f1, self.BzdyBydy)
        BzBydySm = AF(f1, f1, self.BzBydy)
        BzdxBydyxd = AF(fx, f1, self.BzdxBydy)
        BzdyBydyyd = AF(f1, fy, self.BzdyBydy)
        BydyBzdxyd = AF(f1, fy, self.BydyBzdx)
        BydyBzdyxd = AF(fx, f1, self.BydyBzdy)
        BxdxBydySm = AF(f1, f1, self.BxdxBydy)
        BydysqSm = AF(f1, f1, self.Bydysq)
        Bydysqyd = AF(f1, fy, self.Bydysq)
        Bydysqydsq = AF(f1, fyy, self.Bydysq)
        BzdyBydyxd = AF(fx, f1, self.BzdyBydy)
        BzBydyxd = AF(fx, f1, self.BzBydy)
        BzdxBydyxdsq = AF(fxx, f1, self.BzdxBydy)
        BzBydzyd = AF(f1, fy, self.BzBydz)
        BzdyBydyxdyd = AF(fx, fy, self.BzdyBydy)
        BzdxBydyxdyd = AF(fx, fy, self.BzdxBydy)
        BzdyBydyxdsq = AF(fxx, f1, self.BzdyBydy)
        BxBydySm = AF(f1, f1, self.BxBydy)
        BxdxBydyxd = AF(fx, f1, self.BxdxBydy)
        Bydysqxd = AF(fx, f1, self.Bydysq)
        BxBydyxd = AF(fx, f1, self.BxBydy)
        BxdxBydyxdsq = AF(fxx, f1, self.BxdxBydy)
        Bydysqxdsq = AF(fxx, f1, self.Bydysq)
        BzdxBydyyd = AF(f1, fy, self.BzdxBydy)
        BydyBzdyyd = AF(f1, fy, self.BydyBzdy)
        BzBydyyd = AF(f1, fy, self.BzBydy)
        BydyBzdyydsq = AF(f1, fyy, self.BydyBzdy)
        BzdxBydyydsq = AF(f1, fyy, self.BzdxBydy)
        ByBydySm = AF(f1, f1, self.ByBydy)
        BxdxBydyyd = AF(f1, fy, self.BxdxBydy)
        ByBydyxd = AF(fx, f1, self.ByBydy)
        BxBydyyd = AF(f1, fy, self.BxBydy)
        BxdxBydyxdyd = AF(fx, fy, self.BxdxBydy)
        Bydysqxdyd = AF(fx, fy, self.Bydysq)
        ByBydyyd = AF(f1, fy, self.ByBydy)
        BxdxBydyydsq = AF(f1, fyy, self.BxdxBydy)
        BzdtBydySm = AF(f1, f1, self.BzdtBydy)
        BzdtBydyxd = AF(fx, f1, self.BzdtBydy)
        BzdtBydyyd = AF(f1, fy, self.BzdtBydy)

        S60 = -(BxdxBzdxSm + BzdxBydySm)
        S61 = -(BxdxBzdySm + BydyBzdySm)
        S62 = -(BzBxdxSm + BzBydySm + BxdxBzdxxd + BzdxBydyxd)
        S63 = -(BzBxdxSm + BzBydySm + BxdxBzdyyd + BzdyBydyyd)
        S64 = -(BxdxBzdxyd + BydyBzdxyd)
        S65 = -(BxdxBzdyxd + BydyBzdyxd)
        S66 = BxdxsqSm + 2.0 * BxdxBydySm + BydysqSm
        S70 = -(BxBzdxSm + BxdxBzdxxd + BzdxBydyxd)
        S71 = -(BxBzdySm + BxdxBzdyxd + BzdyBydyxd)
        S72 = -(BxBzSm + BzBxdxxd + BzBydyxd + BxBzdxxd + BxdxBzdxxdsq + BzdxBydyxdsq)
        S73 = -(BxBzSm + BzBxdxxd + BzBydyxd + BxBzdyyd + BxdxBzdyxdyd + BzdyBydyxdyd)
        S74 = -(BxBzdxyd + BxdxBzdxxdyd + BzdxBydyxdyd)
        S75 = -(BxBzdyxd + BxdxBzdyxdsq + BzdyBydyxdsq)
        S76 = BxBxdxSm + BxBydySm + Bxdxsqxd + 2.0 * BxdxBydyxd + Bydysqxd
        S77 = BxsqSm + 2.0 * BxBxdxxd + 2.0 * BxBydyxd + Bxdxsqxdsq + 2.0 * BxdxBydyxdsq + Bydysqxdsq
        S80 = -(ByBzdxSm + BxdxBzdxyd + BzdxBydyyd)
        S81 = -(ByBzdySm + BxdxBzdyyd + BydyBzdyyd)
        S82 = -(ByBzSm + ByBzdxxd + BzBxdxyd + BzBydyyd + BxdxBzdxxdyd + BzdxBydyxdyd)
        S83 = -(ByBzSm + BzBxdxyd + BzBydyyd + ByBzdyyd + BxdxBzdyydsq + BydyBzdyydsq)
        S84 = -(ByBzdxyd + BxdxBzdxydsq + BzdxBydyydsq)
        S85 = -(ByBzdyxd + BxdxBzdyxdyd + BzdyBydyxdyd)
        S86 = ByBxdxSm + ByBydySm + Bxdxsqyd + 2.0 * BxdxBydyyd + Bydysqyd
        S87 = BxBySm + ByBxdxxd + ByBydyxd + BxBxdxyd + BxBydyyd + Bxdxsqxdyd + 2.0 * BxdxBydyxdyd + Bydysqxdyd
        S88 = BysqSm + 2.0 * ByBxdxyd + 2.0 * ByBydyyd + Bxdxsqydsq + 2.0 * BxdxBydyydsq + Bydysqydsq

        BzdtBxdxSm = AF(f1, f1, self.BzdtBxdx)
        BzdtBzdxSm = AF(f1, f1, self.BzdtBzdx)
        BzdtBzdySm = AF(f1, f1, self.BzdtBzdy)
        BzBzdtSm = AF(f1, f1, self.BzBzdt)
        BzdtBzdxxd = AF(fx, f1, self.BzdtBzdx)
        BzdtBzdyyd = AF(f1, fy, self.BzdtBzdy)
        BzdtBzdyxd = AF(fx, f1, self.BzdtBzdy)
        BzdtBzdxyd = AF(f1, fy, self.BzdtBzdx)
        BxBzdtSm = AF(f1, f1, self.BxBzdt)
        BzdtBxdxxd = AF(fx, f1, self.BzdtBxdx)
        BzdtBxdxyd = AF(f1, fy, self.BzdtBxdx)
        ByBzdtSm = AF(f1, f1, self.ByBzdt)
        BzdtsqSm = AF(f1, f1, self.Bzdtsq)

        G90 = BzdtBzdxSm.copy()
        G91 = BzdtBzdySm.copy()
        G92 = BzBzdtSm + BzdtBzdxxd
        G93 = BzBzdtSm + BzdtBzdyyd
        G94 = BzdtBzdxyd.copy()
        G95 = BzdtBzdyxd.copy()
        S96 = -(BzdtBxdxSm + BzdtBydySm)
        S97 = -(BxBzdtSm + BzdtBxdxxd + BzdtBydyxd)
        S98 = -(ByBzdtSm + BzdtBxdxyd + BzdtBydyyd)

        mats = np.zeros((self.nx, self.ny, 9, 9), dtype=np.float64)
        rhs = np.zeros((self.nx, self.ny, 9), dtype=np.float64)

        mats[:, :, 0, :] = np.stack([G00, G10, G20, G30, G40, G50, S60, S70, S80], axis=-1)
        mats[:, :, 1, :] = np.stack([G10, G11, G21, G31, G41, G51, S61, S71, S81], axis=-1)
        mats[:, :, 2, :] = np.stack([G20, G21, G22, G32, G42, G52, S62, S72, S82], axis=-1)
        mats[:, :, 3, :] = np.stack([G30, G31, G32, G33, G43, G53, S63, S73, S83], axis=-1)
        mats[:, :, 4, :] = np.stack([G40, G41, G42, G43, G44, G54, S64, S74, S84], axis=-1)
        mats[:, :, 5, :] = np.stack([G50, G51, G52, G53, G54, G55, S65, S75, S85], axis=-1)
        mats[:, :, 6, :] = np.stack([S60, S61, S62, S63, S64, S65, S66, S76, S86], axis=-1)
        mats[:, :, 7, :] = np.stack([S70, S71, S72, S73, S74, S75, S76, S77, S87], axis=-1)
        mats[:, :, 8, :] = np.stack([S80, S81, S82, S83, S84, S85, S86, S87, S88], axis=-1)
        rhs[:, :, :] = np.stack([-G90, -G91, -G92, -G93, -G94, -G95, -S96, -S97, -S98], axis=-1)

        det_proxy = G00 + G11 + G22 + G33 + G44 + G55 + S66 + S77 + S88
        mask = det_proxy > 1.0
        return _solve_velocity_systems_nb(mats, rhs, mask)



def _discover_field_store(field_loc: str | Path, field_tag: str) -> Path | None:
    p = Path(field_loc)
    if p.suffix == '.zarr' and p.exists():
        return p
    if p.is_dir():
        candidate = p / f"field_data_{field_tag}.zarr"
        if candidate.exists():
            return candidate
    return None


def _read_from_store(store_path: Path, name: str, idx: int, nx: int, ny: int) -> np.ndarray:
    root = zarr.open(store_path, mode='r')
    if name not in root:
        raise KeyError(f"Dataset '{name}' not found in {store_path}")
    arr = np.asarray(root[name][:, :, idx], dtype=np.float64)
    if arr.shape == (ny, nx):
        # Backward compatibility for stores written with axes reversed.
        arr = arr.T
    if arr.shape != (nx, ny):
        raise ValueError(f"Dataset {name}[..., {idx}] has shape {arr.shape}, expected {(nx, ny)}")
    return np.ascontiguousarray(arr)


def _read_valid_from_store(store_path: Path, idx: int) -> bool:
    root = zarr.open(store_path, mode='r')
    if 'valid' not in root:
        return True
    valid = root['valid']
    if idx < 0 or idx >= valid.shape[0]:
        return False
    return bool(valid[idx])


def _get_store_nsteps(store_path: Path) -> int:
    root = zarr.open(store_path, mode='r')
    for name in ('bx', 'by', 'bz'):
        if name in root:
            return int(root[name].shape[2])
    raise KeyError(f"No magnetic field datasets found in {store_path}")


def _ensure_velocity_store(store_path: Path, nx: int, ny: int, total_steps: int) -> None:
    root = zarr.open(store_path, mode='a')
    for name in ('Ux', 'Uy', 'Uz'):
        if name not in root:
            raise KeyError(
                f"{name} dataset must be created before worker processes start in {store_path}"
            )
        ds = root[name]
        if ds.shape[0] != nx or ds.shape[1] != ny:
            raise ValueError(
                f"Dataset {name} has spatial shape {ds.shape[:2]}, expected {(nx, ny)}"
            )
        if ds.shape[2] < total_steps:
            ds.resize((nx, ny, total_steps))


def _write_to_store(store_path: Path, name: str, idx: int, arr: np.ndarray, total_steps: int) -> None:
    root = zarr.open(store_path, mode='a')
    arr = np.asarray(arr, dtype=np.float32)

    required_steps = max(int(total_steps), int(idx) + 1)

    if name not in root:
        raise KeyError(
            f"{name} dataset must be created before worker processes start in {store_path}"
        )

    ds = root[name]
    if ds.shape[0] != arr.shape[0] or ds.shape[1] != arr.shape[1]:
        raise ValueError(
            f"Dataset {name} has spatial shape {ds.shape[:2]}, expected {(arr.shape[0], arr.shape[1])}"
        )
    if ds.shape[2] < required_steps:
        ds.resize((ds.shape[0], ds.shape[1], required_steps))

    root[name][:, :, idx] = arr

def read_field_file(path: str | Path, nx: int, ny: int) -> np.ndarray:
    vals = np.loadtxt(path, dtype=np.float64)
    vals = np.atleast_1d(vals)
    if vals.size != nx * ny:
        raise ValueError(f"Expected {nx*ny} values in {path}, found {vals.size}.")
    return np.ascontiguousarray(vals.reshape(ny, nx).T, dtype=np.float64)


def write_field_file(path: str | Path, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float64)
    with open(path, "w", encoding="utf-8") as f:
        for j in range(arr.shape[1]):
            for i in range(arr.shape[0]):
                f.write(f"{arr[i, j]}\n")


def run_dave4vm_series(field_loc: str | Path,
                       field_tag: str,
                       nx: int,
                       ny: int,
                       start_index: int,
                       end_index: int,
                       window: int,
                       dx: float = 360.0,
                       dy: float = 360.0,
                       dt_seconds: float = 720.0) -> None:
    field_loc = Path(field_loc)
    store_path = _discover_field_store(field_loc, field_tag)
    dtinv = 1.0 / float(dt_seconds)

    if store_path is not None:
        total_steps = _get_store_nsteps(store_path)
        _ensure_velocity_store(store_path, nx, ny, total_steps)
    else:
        total_steps = end_index + 1

    for findex in range(start_index, end_index + 1):
        print(f"Making velocity step {findex} of {end_index}")

        findex2 = findex + 1
        if store_path is not None:
            valid_minus = _read_valid_from_store(store_path, findex)
            valid_plus = _read_valid_from_store(store_path, findex2)
            if not (valid_minus and valid_plus):
                print(f"Skipping velocity step {findex}: missing snapshot in required pair ({findex}, {findex2})")
                continue

            bxminus = _read_from_store(store_path, 'bx', findex, nx, ny)
            byminus = _read_from_store(store_path, 'by', findex, nx, ny)
            bzminus = _read_from_store(store_path, 'bz', findex, nx, ny)
            bxplus = _read_from_store(store_path, 'bx', findex2, nx, ny)
            byplus = _read_from_store(store_path, 'by', findex2, nx, ny)
            bzplus = _read_from_store(store_path, 'bz', findex2, nx, ny)
        else:
            bxminus = read_field_file(field_loc / f"bx_{field_tag}_{findex}.txt", nx, ny)
            byminus = read_field_file(field_loc / f"by_{field_tag}_{findex}.txt", nx, ny)
            bzminus = read_field_file(field_loc / f"bz_{field_tag}_{findex}.txt", nx, ny)
            bxplus = read_field_file(field_loc / f"bx_{field_tag}_{findex2}.txt", nx, ny)
            byplus = read_field_file(field_loc / f"by_{field_tag}_{findex2}.txt", nx, ny)
            bzplus = read_field_file(field_loc / f"bz_{field_tag}_{findex2}.txt", nx, ny)

        bx = 0.5 * (bxplus + bxminus)
        by = 0.5 * (byplus + byminus)
        bz = 0.5 * (bzplus + bzminus)
        bxt = dtinv * (bxplus - bxminus)
        byt = dtinv * (byplus - byminus)
        bzt = dtinv * (bzplus - bzminus)

        dtest = Dave4VMNumba(dx=dx, dy=dy, nx=nx, ny=ny)
        dtest.get_derivatives(bx, by, bz)
        dtest.weight_functions(window)
        dtest.set_matrix_terms_pre_filter(bxt, byt, bzt)
        vx, vy, vz = dtest.compose_matrix_and_invert()

        if store_path is not None:
            _write_to_store(store_path, 'Ux', findex, vx, total_steps)
            _write_to_store(store_path, 'Uy', findex, vy, total_steps)
            _write_to_store(store_path, 'Uz', findex, vz, total_steps)
        else:
            write_field_file(field_loc / f"Ux_{field_tag}_{window}_{findex}.txt", vx)
            write_field_file(field_loc / f"Uy_{field_tag}_{window}_{findex}.txt", vy)
            write_field_file(field_loc / f"Uz_{field_tag}_{window}_{findex}.txt", vz)


def main() -> None:
    parser = argparse.ArgumentParser(description="Numba-accelerated Python replacement for the DAVE4VM C++ executable.")
    parser.add_argument("field_loc")
    parser.add_argument("field_tag")
    parser.add_argument("nx", type=int)
    parser.add_argument("ny", type=int)
    parser.add_argument("start_index", type=int)
    parser.add_argument("end_index", type=int)
    parser.add_argument("window", type=int)
    parser.add_argument("--dx", type=float, default=360.0)
    parser.add_argument("--dy", type=float, default=360.0)
    parser.add_argument("--dt-seconds", type=float, default=720.0)
    args = parser.parse_args()

    run_dave4vm_series(
        field_loc=args.field_loc,
        field_tag=args.field_tag,
        nx=args.nx,
        ny=args.ny,
        start_index=args.start_index,
        end_index=args.end_index,
        window=args.window,
        dx=args.dx,
        dy=args.dy,
        dt_seconds=args.dt_seconds,
    )


if __name__ == "__main__":
    main()
