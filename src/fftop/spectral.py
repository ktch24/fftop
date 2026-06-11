import numpy as np
import math
import pyvista as pv
import matplotlib.pyplot as plt


# ------------------------
# Naive rectangular-rule (slow O(N^2)) for one slice
# ------------------------
def winding_naive_rect(u, dx, dy):
    """
    Direct rectangular-rule summation (slow): O((nx*ny)^2).
    u: (nx, ny, 3)
    """
    nx, ny, _ = u.shape
    xv = np.arange(nx)*dx
    yv = np.arange(ny)*dy

    ux = u[..., 0]
    uy = u[..., 1]
    uz = u[..., 2]

    Ax = np.zeros((nx, ny), dtype=float)
    Ay = np.zeros((nx, ny), dtype=float)
    Az = np.zeros((nx, ny), dtype=float)

    c = (dx*dy)/(2*np.pi)

    for i in range(nx):
        rx = (xv[i] - xv)[:, None]          # (nx,1)
        for j in range(ny):
            ry = (yv[j] - yv)[None, :]      # (1,ny)
            r2 = rx*rx + ry*ry              # (nx,ny)

            # Build 1/r^2 safely, zeroing the self-term
            inv_r2 = np.zeros_like(r2, dtype=float)
            np.divide(1.0, r2, out=inv_r2, where=(r2 > 0.0))

            Ax[i, j] = c * np.sum((-ry * inv_r2) * uz)
            Ay[i, j] = c * np.sum(( +rx * inv_r2) * uz)
            Az[i, j] = c * ( np.sum(( +ry * inv_r2) * ux)
                            +np.sum(( -rx * inv_r2) * uy) )

    return np.stack([Ax, Ay, Az], axis=-1)

# ------------------------
# Fast FD via sampled-kernel + FFT (linear convolution, free-space)
# ------------------------
def _build_rect_kernels_fft(nx, ny, dx, dy, pad_factor=2, eps=0.0):
    npx, npy = pad_factor*nx, pad_factor*ny
    xs = (np.arange(npx) - npx//2)*dx
    ys = (np.arange(npy) - npy//2)*dy
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    r2 = X*X + Y*Y
    if eps>0:  # optional regularization at r=0 (usually leave at 0 and mask instead)
        r2 = r2 + (r2==0.0)*eps**2
    c = (dx*dy)/(2*np.pi)
    Kx  = np.zeros_like(r2); Ky=np.zeros_like(r2)
    Kzx = np.zeros_like(r2); Kzy=np.zeros_like(r2)
    m = r2>0
    Kx[m]  = c * (-Y[m]/r2[m])
    Ky[m]  = c * ( +X[m]/r2[m])
    Kzx[m] = c * ( +Y[m]/r2[m])
    Kzy[m] = c * ( -X[m]/r2[m])
    # center at (0,0) index for linear conv
    Kx_hat  = np.fft.fft2(np.fft.ifftshift(Kx))
    Ky_hat  = np.fft.fft2(np.fft.ifftshift(Ky))
    Kzx_hat = np.fft.fft2(np.fft.ifftshift(Kzx))
    Kzy_hat = np.fft.fft2(np.fft.ifftshift(Kzy))
    return (Kx_hat, Ky_hat, Kzx_hat, Kzy_hat, npx, npy)

def winding_fd_fast(u, dx, dy, pad_factor=2, kernels=None):
    """
    u: (nx,ny,3) or (nx,ny,ns,3). Returns A with same leading shape (...,3).
    """
    u = np.asarray(u)
    if u.ndim==3: u = u[...,None,:]
    nx, ny, ns, _ = u.shape
    if kernels is None:
        kernels = _build_rect_kernels_fft(nx, ny, dx, dy, pad_factor=pad_factor)
    Kx_hat, Ky_hat, Kzx_hat, Kzy_hat, npx, npy = kernels
    def pad2(a):
        out = np.zeros((npx,npy))
        out[:nx,:ny]=a
        return out
    A = np.zeros_like(u, float)
    for s in range(ns):
        Ux = np.fft.fft2(pad2(u[:,:,s,0]))
        Uy = np.fft.fft2(pad2(u[:,:,s,1]))
        Uz = np.fft.fft2(pad2(u[:,:,s,2]))
        Ax = np.fft.ifft2(Uz*Kx_hat).real[:nx,:ny]
        Ay = np.fft.ifft2(Uz*Ky_hat).real[:nx,:ny]
        Az = np.fft.ifft2(Ux*Kzx_hat + Uy*Kzy_hat).real[:nx,:ny]
        A[:,:,s,0],A[:,:,s,1],A[:,:,s,2]=Ax,Ay,Az
    return A if A.shape[2]>1 else A[:,:,0,:]

# ------------------------
# Spectral (operator-symbol) route, free-space via padding
# ------------------------
def _build_spectral_multipliers(npx, npy, dx, dy, discrete=True):
    """
    Precompute operator multipliers on the padded grid.
    Returns (Mx_x, My_x, Mz) where:
      Ax_hat = Mx_x * Uz_hat
      Ay_hat = My_x * Uz_hat
      Az_hat = Mz * (Dx*Uy_hat - Dy*Ux_hat)   (we'll pass Dx,U y_hat etc.)
    """
    # frequency grids
    kx = 2*np.pi*np.fft.fftfreq(npx, d=dx)[:,None]
    ky = 2*np.pi*np.fft.fftfreq(npy, d=dy)[None,:]
    KX, KY = np.meshgrid(kx[:,0], ky[0,:], indexing='ij')

    if discrete:
        Dx = np.sin(KX*dx)/dx
        Dy = np.sin(KY*dy)/dy
        Lap = 4*np.sin(0.5*KX*dx)**2/dx**2 + 4*np.sin(0.5*KY*dy)**2/dy**2
        Inv = np.zeros_like(Lap); Inv[Lap>0] = 1.0/Lap[Lap>0]
        Mx = (1j/(2*np.pi)) * Dy * Inv        # Ax_hat = Mx * Uz_hat
        My = (1j/(2*np.pi)) * (-Dx) * Inv     # Ay_hat = My * Uz_hat
        Mz_pref = (1j/(2*np.pi)) * Inv        # Az_hat = Mz_pref*(Dx*Uy - Dy*Ux)
        # zero DC (already through Inv==0)
    else:
        K2 = KX**2 + KY**2
        Inv = np.zeros_like(K2); Inv[K2>0] = 1.0/K2[K2>0]
        Dx, Dy = KX, KY
        Mx = (1j/(2*np.pi)) * Dy * Inv
        My = (1j/(2*np.pi)) * (-Dx) * Inv
        Mz_pref = (1j/(2*np.pi)) * Inv
    return Mx, My, Mz_pref, Dx, Dy

def winding_spectral(u, dx, dy, pad_factor=2, discrete=True, multipliers=None):
    """
    u: (nx,ny,3) or (nx,ny,ns,3). Returns A with same leading shape (...,3).
    Uses i*tilde(k)/Lambda_d (discrete) or i*k/k^2 (continuous). Includes 1/(2π).
    """
    u = np.asarray(u)
    if u.ndim==3: u = u[...,None,:]
    nx, ny, ns, _ = u.shape
    npx, npy = pad_factor*nx, pad_factor*ny

    if multipliers is None:
        Mx, My, Mz_pref, Dx, Dy = _build_spectral_multipliers(npx, npy, dx, dy, discrete=discrete)
    else:
        Mx, My, Mz_pref, Dx, Dy = multipliers

    def pad2(a):
        out = np.zeros((npx,npy))
        out[:nx,:ny]=a
        return out

    A = np.zeros_like(u, float)
    for s in range(ns):
        Ux = np.fft.fft2(pad2(u[:,:,s,0]))
        Uy = np.fft.fft2(pad2(u[:,:,s,1]))
        Uz = np.fft.fft2(pad2(u[:,:,s,2]))

        Ax = np.fft.ifft2( Mx * Uz ).real[:nx,:ny]
        Ay = np.fft.ifft2( My * Uz ).real[:nx,:ny]
        Az = np.fft.ifft2( Mz_pref * (Dx*Uy - Dy*Ux) ).real[:nx,:ny]

        A[:,:,s,0],A[:,:,s,1],A[:,:,s,2]=Ax,Ay,Az

    return A if A.shape[2]>1 else A[:,:,0,:]

def ddx(F, d):
    G = np.empty_like(F, dtype=float)
        # interior: central
    G[1:-1, :, :] = (F[2:, :, :] - F[:-2, :, :]) / (2*d)
        # boundaries: 2nd-order one-sided
    G[0, :, :]    = (-3*F[0, :, :] + 4*F[1, :, :] - F[2, :, :]) / (2*d)
    G[-1, :, :]   = ( 3*F[-1, :, :] - 4*F[-2, :, :] + F[-3, :, :]) / (2*d)
    return G

def ddy(F, d):
    G = np.empty_like(F, dtype=float)
    G[:, 1:-1, :] = (F[:, 2:, :] - F[:, :-2, :]) / (2*d)
    G[:, 0, :]    = (-3*F[:, 0, :] + 4*F[:, 1, :] - F[:, 2, :]) / (2*d)
    G[:, -1, :]   = ( 3*F[:, -1, :] - 4*F[:, -2, :] + F[:, -3, :]) / (2*d)
    return G

def ddz(F, d):
    G = np.empty_like(F, dtype=float)
    G[:, :, 1:-1] = (F[:, :, 2:] - F[:, :, :-2]) / (2*d)
    G[:, :, 0]    = (-3*F[:, :, 0] + 4*F[:, :, 1] - F[:, :, 2]) / (2*d)
    G[:, :, -1]   = ( 3*F[:, :, -1] - 4*F[:, :, -2] + F[:, :, -3]) / (2*d)
    return G

def curl_A(A, x, y, z):
    """
    Compute curl of a vector field A on a Cartesian grid:
        curl(A) = ∇×A

    A: (nx, ny, nz, 3)
    x,y,z: 1D arrays of coordinates (assumed uniformly spaced)

    Returns:
        C: (nx, ny, nz, 3) where
           C[...,0] = (∂Az/∂y - ∂Ay/∂z)
           C[...,1] = (∂Ax/∂z - ∂Az/∂x)
           C[...,2] = (∂Ay/∂x - ∂Ax/∂y)
    """
    A = np.asarray(A)
    assert A.ndim == 4 and A.shape[-1] == 3, "A must have shape (nx,ny,nz,3)"

    dx = float(x[1] - x[0])
    dy = float(y[1] - y[0])
    dz = float(z[1] - z[0])

    Ax = A[..., 0]
    Ay = A[..., 1]
    Az = A[..., 2]


    dAz_dy = ddy(Az, dy)
    dAy_dz = ddz(Ay, dz)

    dAx_dz = ddz(Ax, dz)
    dAz_dx = ddx(Az, dx)

    dAy_dx = ddx(Ay, dx)
    dAx_dy = ddy(Ax, dy)

    Cx = dAz_dy - dAy_dz
    Cy = dAx_dz - dAz_dx
    Cz = dAy_dx - dAx_dy

    return np.stack([Cx, Cy, Cz], axis=-1)

def z_slice(B, z, z0=None, iz=None):
    """
    Extract a z-slice from B(x,y,z).

    Provide either:
      z0 : physical z value
      iz : integer index

    Returns:
      Bslice : (nx, ny, 3)
      iz     : slice index used
      zval   : physical z value
    """
    if (z0 is None) == (iz is None):
        raise ValueError("Provide exactly one of z0 or iz")

    if z0 is not None:
        iz = np.argmin(np.abs(z - z0))

    return B[:, :, iz, :], iz, z[iz]

def winding_density(A, B, eps=0.0):
    """
    w = A · (B / |B|)
    A, B: (nx, ny, nz, 3)
    Returns: (nx, ny, nz, 1)
    """
    Bmag = np.linalg.norm(B, axis=-1)
    mask = (Bmag > eps) if eps > 0 else (Bmag != 0)

    w = np.zeros(Bmag.shape, dtype=A.dtype)
    AB = np.sum(A * B, axis=-1)   # includes z-term
    w[mask] = AB[mask] / Bmag[mask]

    return w[..., np.newaxis]



def normalize_by_Bz(B, eps=0.0):
    """
    Compute B / Bz safely.

    B   : array of shape (nx, ny, nz, 3)
    eps : cutoff for |Bz| (default 0 => exact zeroing)

    Returns:
        C : array of shape (nx, ny, nz, 3)
            C = 0 where Bz == 0 (or |Bz| <= eps)
    """
    B = np.asarray(B)
    assert B.ndim == 4 and B.shape[-1] == 3

    Bz = B[..., 2]

    if eps == 0.0:
        mask = (Bz != 0.0)
    else:
        mask = (np.abs(Bz) > eps)

    C = np.zeros_like(B, dtype=B.dtype)

    C[mask] = B[mask] / Bz[mask][..., np.newaxis]

    return C
 


def dot_density(A, B):
    """
    Compute dot product A · B

    A, B : arrays of shape (nx, ny, nz, 3)

    Returns:
        d : array of shape (nx, ny, nz, 1)
    """
    d = np.sum(A * B, axis=-1)
    return d[..., np.newaxis]


def scalar_z_slice(S, z, z0=None, iz=None):
    """
    S: (nx, ny, nz) or (nx, ny, nz, 1)
    z: (nz,)
    returns: (nx, ny) slice, iz, zval
    """
    if S.ndim == 4:
        if S.shape[-1] != 1:
            raise ValueError("If S is 4D it must have last dim = 1.")
        S3 = S[..., 0]
    elif S.ndim == 3:
        S3 = S
    else:
        raise ValueError("S must be (nx,ny,nz) or (nx,ny,nz,1).")

    if (z0 is None) == (iz is None):
        raise ValueError("Provide exactly one of z0 or iz")

    if z0 is not None:
        iz = int(np.argmin(np.abs(z - z0)))

    return S3[:, :, iz], iz, z[iz]

def plot_component_slice(
    B, x, y, z,
    component="z",
    z0=None,
    iz=None,
    cmap="RdBu_r",
    vlim=None
):
    """
    Plot a scalar component Bx, By, or Bz at fixed z.
    """
    comp_index = {"x":0, "y":1, "z":2}[component.lower()]
    Bslice, iz, zval = z_slice(B, z, z0=z0, iz=iz)

    data = Bslice[:, :, comp_index]

    if vlim is None:
        vmax = np.max(np.abs(data))
        vlim = (-vmax, vmax)

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.pcolormesh(
        x, y, data.T,
        cmap=cmap,
        vmin=vlim[0],
        vmax=vlim[1],
        shading="auto"
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"$B_{component}$ at z = {zval:.2f}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(f"$B_{component}$")

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()


def plot_scalar_slice(
    S, x, y, z,
    z0=None, iz=None,
    title="",
    label="",
    cmap="RdBu_r",
    vlim=None,
    center_zero=False
):
    """
    Pretty pcolormesh plot for a scalar density slice at fixed z.
    """
    data, iz, zval = scalar_z_slice(S, z, z0=z0, iz=iz)

    if vlim is None:
        if center_zero:
            vmax = np.max(np.abs(data))
            vlim = (-vmax, vmax)
        else:
            vlim = (np.min(data), np.max(data))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.pcolormesh(
        x, y, data.T,
        shading="auto",
        cmap=cmap,
        vmin=vlim[0],
        vmax=vlim[1],
    )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if title:
        ax.set_title(f"{title}  (z = {zval:.2f})")
    else:
        ax.set_title(f"z = {zval:.2f}")

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(label if label else "value")

    ax.set_aspect("equal")
    plt.tight_layout()
    plt.show()

def rho_A_dot_B_over_Bz(A, B, eps=0.0):
    """
    rho = (A·B)/Bz with safe zeroing where |Bz|<=eps (or Bz==0 if eps=0).
    A,B: (nx,ny,nz,3)
    returns: (nx,ny,nz)
    """
    Bz = B[..., 2]
    if eps == 0.0:
        mask = (Bz != 0.0)
    else:
        mask = (np.abs(Bz) > eps)

    rho = np.zeros(Bz.shape, dtype=A.dtype)
    AB = np.sum(A * B, axis=-1)
    rho[mask] = AB[mask] / Bz[mask]
    return rho

def plot_field_lines_pyvista(B, x, y, z, seeds,
                             max_steps=8000,
                             initial_step_length=None,
                             terminal_speed=1e-12,
                             integrator_type=45,
                             color_by="|B|",
                             tube_radius=None,
                             line_width=2):
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    grid = pv.StructuredGrid(X, Y, Z)

    grid["B"] = np.ascontiguousarray(B.reshape(-1, 3), dtype=float)

    if color_by == "|B|":
        grid["|B|"] = np.linalg.norm(B, axis=-1).reshape(-1).astype(float)
        scalars = "|B|"
    elif color_by == "Bz":
        grid["Bz"] = B[..., 2].reshape(-1).astype(float)
        scalars = "Bz"
    else:
        scalars = None

    seed_points = pv.PolyData(np.asarray(seeds, dtype=float))

    kwargs = dict(
        source=seed_points,
        vectors="B",
        integrator_type=integrator_type,
        max_steps=max_steps,
        terminal_speed=terminal_speed,
    )
    if initial_step_length is not None:
        kwargs["initial_step_length"] = float(initial_step_length)

    sl = grid.streamlines_from_source(**kwargs)

    p = pv.Plotter(notebook=True)
    if tube_radius is not None:
        p.add_mesh(sl.tube(radius=float(tube_radius)), scalars=scalars)
    else:
        p.add_mesh(sl, scalars=scalars, line_width=line_width)

    p.add_axes()
    p.show_grid()
    return p.show()

def make_rect_grid(B, rho, x, y, z, B_name="B", rho_name="rho"):
    """
    B:   (nx, ny, nz, 3)
    rho: (nx, ny, nz) or (nx, ny, nz, 1)
    x,y,z: 1D arrays
    """
    grid = pv.RectilinearGrid(x, y, z)

    # VTK expects point data flattened in Fortran order for rect/structured grids
    Bf = np.ascontiguousarray(B).reshape(-1, 3, order="F")
    grid.point_data[B_name] = Bf

    rho3 = rho[..., 0] if (rho.ndim == 4 and rho.shape[-1] == 1) else rho
    grid.point_data[rho_name] = np.ascontiguousarray(rho3).ravel(order="F")

    return grid


def trace_streamlines(grid, seeds, vectors="B",
                      max_steps=8000,
                      initial_step_length=None,
                      terminal_speed=1e-12,
                      integrator_type=45):
    seed_pd = pv.PolyData(np.asarray(seeds, dtype=float))

    kwargs = dict(
        source=seed_pd,
        vectors=vectors,
        integrator_type=integrator_type,
        max_steps=max_steps,
        terminal_speed=terminal_speed,
    )
    if initial_step_length is not None:
        kwargs["initial_step_length"] = float(initial_step_length)

    sl = grid.streamlines_from_source(**kwargs)
    return sl

def check_point_mapping(grid, B, x, y, z, i, j, k):
    p = np.array([[x[i], y[j], z[k]]], float)
    samp = pv.PolyData(p).sample(grid)
    B_vtk = np.array(samp.point_data["B"][0])
    B_np  = np.array(B[i, j, k, :])
    print("B numpy:", B_np)
    print("B vtk  :", B_vtk)
    print("diff   :", B_vtk - B_np)
    
def streamlines_from_seeds(grid, seeds, vectors="B",
                           max_steps=20000,
                           initial_step_length=None,
                           terminal_speed=1e-14,
                           integrator_type=45):
    seed_pd = pv.PolyData(np.asarray(seeds, float))

    kwargs = dict(
        source=seed_pd,
        vectors=vectors,
        integrator_type=integrator_type,
        max_steps=max_steps,
        terminal_speed=terminal_speed,
    )
    if initial_step_length is not None:
        kwargs["initial_step_length"] = float(initial_step_length)

    return grid.streamlines_from_source(**kwargs)

    
def make_rect_grid_with_fields(B, rho, x, y, z, B_name="B", rho_name="rho"):
    """
    B:   (nx, ny, nz, 3)
    rho: (nx, ny, nz) or (nx, ny, nz, 1)
    x,y,z: 1D arrays

    Uses Fortran-order flattening (required for correct mapping on RectilinearGrid).
    """
    grid = pv.RectilinearGrid(x, y, z)

    grid.point_data[B_name] = np.ascontiguousarray(B).reshape(-1, 3, order="F")

    rho3 = rho[..., 0] if (rho.ndim == 4 and rho.shape[-1] == 1) else rho
    grid.point_data[rho_name] = np.ascontiguousarray(rho3).ravel(order="F")

    return grid

def integrate_rho_along_one_fieldline(grid, seed, vectors="B", rho_name="rho",
                                      max_steps=20000,
                                      initial_step_length=None,
                                      terminal_speed=1e-14,
                                      integrator_type=45,
                                      integration_direction="both"):
    """
    Returns ∫ rho ds along the (longest) streamline seeded at 'seed'.
    Returns np.nan if no valid streamline is returned.
    """
    seed_pd = pv.PolyData(np.asarray([seed], dtype=float))

    kwargs = dict(
        source=seed_pd,
        vectors=vectors,
        integrator_type=integrator_type,       # 45 = adaptive RK4/5
        max_steps=max_steps,
        terminal_speed=terminal_speed,
        integration_direction=integration_direction,  # "forward" / "backward" / "both"
    )
    if initial_step_length is not None:
        kwargs["initial_step_length"] = float(initial_step_length)

    sl = grid.streamlines_from_source(**kwargs)

    if sl.n_cells < 1 or sl.n_points < 2:
        return np.nan

    sls = sl.sample(grid)
    rho_vals = np.asarray(sls.point_data[rho_name], dtype=float)
    pts = np.asarray(sls.points, dtype=float)

    # integrate longest polyline cell (sometimes multiple cells come back)
    best_I, best_L = np.nan, -np.inf
    for cid in range(sls.n_cells):
        cell = sls.get_cell(cid)
        ids = np.asarray(cell.point_ids, dtype=int)
        if ids.size < 2:
            continue
        p = pts[ids]
        r = rho_vals[ids]

        ds = np.linalg.norm(p[1:] - p[:-1], axis=1)
        I = np.sum(0.5*(r[1:] + r[:-1]) * ds)   # trapezoid
        L = ds.sum()

        if L > best_L:
            best_L, best_I = L, I

    return float(best_I)

def fieldline_integrated_map(grid, seed_x, seed_y, seed_z,
                             max_steps=20000,
                             initial_step_length=None,
                             terminal_speed=1e-14,
                             integration_direction="forward"):
    """
    Returns I2 with shape (len(seed_x), len(seed_y))
    """
    I2 = np.full((len(seed_x), len(seed_y)), np.nan, dtype=float)

    for i, xx in enumerate(seed_x):
        for j, yy in enumerate(seed_y):
            I2[i, j] = integrate_rho_along_one_fieldline(
                grid,
                seed=(float(xx), float(yy), float(seed_z)),
                max_steps=max_steps,
                initial_step_length=initial_step_length,
                terminal_speed=terminal_speed,
                integration_direction=integration_direction
            )
    return I2
    
def plot_seed_map(I2, seed_x, seed_y, title=r"$\int \rho\,ds$"):
    plt.figure(figsize=(6,5))
    plt.pcolormesh(seed_x, seed_y, I2.T, shading="auto")
    plt.gca().set_aspect("equal")
    plt.colorbar(label=title)
    plt.xlabel("seed x")
    plt.ylabel("seed y")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_seed_map_pretty_imshow(I2, seed_x, seed_y, title=r"$\int \rho\,ds$",
                                interpolation="bicubic"):
    # imshow expects [y,x] indexing, so transpose
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(
        I2.T,
        origin="lower",
        extent=[seed_x.min(), seed_x.max(), seed_y.min(), seed_y.max()],
        aspect="equal",
        interpolation=interpolation
    )
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(title)
    ax.set_xlabel("seed x")
    ax.set_ylabel("seed y")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()


# ------------------------
# Helpers: disc mask, cylindrical components, radial bins
# ------------------------
def disc_mask(nx, ny, dx, dy, R, center=(0.0,0.0), taper=0.0):
    x = (np.arange(nx)-nx//2)*dx
    y = (np.arange(ny)-ny//2)*dy
    X, Y = np.meshgrid(x, y, indexing='ij')
    r = np.hypot(X-center[0], Y-center[1])
    M = (r <= R).astype(float)
    if taper > 0.0:
        ann = (r > R - taper) & (r <= R)
        t = (R - r[ann]) / taper
        M[ann] = 0.5*(1 - np.cos(np.pi*t))
    return M

def cart_to_cyl(Ax, Ay, X, Y):
    """Return (Ar, Atheta); A_r = Ax cosθ + Ay sinθ; A_θ = -Ax sinθ + Ay cosθ."""
    r = np.hypot(X, Y) + 1e-30
    cosT = X / r
    sinT = Y / r
    Ar  = Ax*cosT + Ay*sinT
    Ath = -Ax*sinT + Ay*cosT
    return Ar, Ath

def radial_bin_average(F, r, mask, nbins=64, rmax=None):
    """Average field F over annuli; returns (r_centers, F_avg)."""
    if rmax is None:
        rmax = r[mask>0].max() if np.any(mask>0) else r.max()
    edges = np.linspace(0.0, rmax, nbins+1)
    centers = 0.5*(edges[:-1]+edges[1:])
    vals = np.zeros(nbins, dtype=float)
    counts = np.zeros(nbins, dtype=float)
    idx = np.digitize(r.ravel(), edges) - 1
    good = (idx>=0) & (idx<nbins) & (mask.ravel()>0)
    np.add.at(vals, idx[good], F.ravel()[good])
    np.add.at(counts, idx[good], 1.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        avg = np.where(counts>0, vals/counts, np.nan)
    return centers, avg



import numpy as np

def _simpson_weights_uniform(n: int, h: float) -> np.ndarray:
    """
    1D quadrature weights for a uniform grid with n points and spacing h.

    Uses:
      - composite Simpson 1/3 when n-1 is even
      - composite Simpson 1/3 + Simpson 3/8 on the last 3 intervals when n-1 is odd
      - trapezoid only when n == 2

    Returns weights w such that
        integral ≈ sum_i w[i] * f[i]
    """
    if n < 2:
        raise ValueError("Need at least 2 grid points per axis.")

    m = n - 1  # number of intervals
    w = np.zeros(n, dtype=float)

    if m == 1:
        # Trapezoid fallback
        w[:] = h * np.array([0.5, 0.5])
        return w

    if m % 2 == 0:
        # Standard composite Simpson 1/3
        w[0] = 1.0
        w[-1] = 1.0
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w *= h / 3.0
        return w

    # Odd number of intervals: use Simpson 1/3 on the first m-3 intervals
    # and Simpson 3/8 on the last 3 intervals.
    k = m - 3  # even

    if k > 0:
        # Simpson 1/3 on points 0..k
        w[:k+1][0] += h / 3.0
        w[:k+1][-1] += h / 3.0
        w[1:k:2] += 4.0 * h / 3.0
        w[2:k-1:2] += 2.0 * h / 3.0

    # Simpson 3/8 on points k..k+3
    w[k:k+4] += (3.0 * h / 8.0) * np.array([1.0, 3.0, 3.0, 1.0])

    return w


def volume_integral_simpson(arr: np.ndarray, dx: float, dy: float, dz: float) -> float:
    """
    Volume integral of data on a uniform Cartesian grid.

    Parameters
    ----------
    arr : ndarray
        Shape (nx, ny, nz) or (nx, ny, nz, 1).
    dx, dy, dz : float
        Grid spacings.

    Returns
    -------
    float
        Approximation to ∭ arr(x,y,z) dV
    """
    arr = np.asarray(arr)

    if arr.ndim == 4:
        if arr.shape[-1] != 1:
            raise ValueError("If 4D, expected shape (nx, ny, nz, 1).")
        arr = arr[..., 0]
    elif arr.ndim != 3:
        raise ValueError("Expected shape (nx, ny, nz) or (nx, ny, nz, 1).")

    nx, ny, nz = arr.shape

    wx = _simpson_weights_uniform(nx, dx)
    wy = _simpson_weights_uniform(ny, dy)
    wz = _simpson_weights_uniform(nz, dz)

    # Tensor-product quadrature:
    # integral = sum_{i,j,k} wx[i] wy[j] wz[k] arr[i,j,k]
    return np.einsum("i,j,k,ijk->", wx, wy, wz, arr, optimize=True)

def trilinear_scalar_interp(F, x, y, z):
    """
    F: (nx, ny, nz) scalar field on regular grid
    x,y,z: 1D coordinate arrays

    Returns callable f(pt) -> scalar
    """
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    nx, ny, nz = F.shape

    def interp(pt):
        xx, yy, zz = pt

        # fractional indices
        ix = (xx - x[0]) / dx
        iy = (yy - y[0]) / dy
        iz = (zz - z[0]) / dz

        i = int(np.floor(ix))
        j = int(np.floor(iy))
        k = int(np.floor(iz))

        if i < 0 or i >= nx-1 or j < 0 or j >= ny-1 or k < 0 or k >= nz-1:
            return np.nan

        fx = ix - i
        fy = iy - j
        fz = iz - k

        # corners
        c000 = F[i  , j  , k  ]
        c100 = F[i+1, j  , k  ]
        c010 = F[i  , j+1, k  ]
        c110 = F[i+1, j+1, k  ]
        c001 = F[i  , j  , k+1]
        c101 = F[i+1, j  , k+1]
        c011 = F[i  , j+1, k+1]
        c111 = F[i+1, j+1, k+1]

        c00 = c000*(1-fx) + c100*fx
        c10 = c010*(1-fx) + c110*fx
        c01 = c001*(1-fx) + c101*fx
        c11 = c011*(1-fx) + c111*fx

        c0 = c00*(1-fy) + c10*fy
        c1 = c01*(1-fy) + c11*fy

        return c0*(1-fz) + c1*fz

    return interp

    
def trilinear_vector_interp(B, x, y, z):
    fx = trilinear_scalar_interp(B[...,0], x, y, z)
    fy = trilinear_scalar_interp(B[...,1], x, y, z)
    fz = trilinear_scalar_interp(B[...,2], x, y, z)

    def interp(pt):
        return np.array([fx(pt), fy(pt), fz(pt)], dtype=float)

    return interp

def trace_and_integrate_dz(B_interp, rho_interp,
                                 r0, dz_step, z_target,
                                 eps_Bz=0.0, max_steps=200000):
    """
    dr/dz = B/Bz
    I = ∫ rho dz
    """
    r = np.array(r0, dtype=float)
    pts = [r.copy()]

    rho_prev = rho_interp(r)
    if not np.isfinite(rho_prev):
        return np.array(pts), np.nan

    def f(rr):
        B = B_interp(rr)
        bz = B[2]
        if not np.isfinite(B).all():
            return None
        if eps_Bz == 0.0:
            if bz == 0.0:
                return None
        else:
            if bz <= eps_Bz:
                return None
        return B / bz

    I = 0.0

    for _ in range(max_steps):
        if (dz_step > 0 and r[2] >= z_target) or (dz_step < 0 and r[2] <= z_target):
            break

        k1 = f(r)
        if k1 is None: break

        k2 = f(r + 0.5*dz_step*k1)
        if k2 is None: break

        k3 = f(r + 0.5*dz_step*k2)
        if k3 is None: break

        k4 = f(r + dz_step*k3)
        if k4 is None: break

        r_new = r + (dz_step/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        r_new[2] = r[2] + dz_step  # exact z step

        rho_new = rho_interp(r_new)
        if not np.isfinite(rho_new):
            break

        I += 0.5*(rho_prev + rho_new)*dz_step

        r = r_new
        rho_prev = rho_new
        pts.append(r.copy())

    return np.array(pts), float(I)

def trace_and_integrate_dz_scalar(B_interp, rho_interp,
                                 r0, dz_step, z_target,
                                 eps_Bz=0.0, max_steps=200000):
    """
    dr/dz = B/Bz
    I = ∫ rho dz
    """
    r = np.array(r0, dtype=float)
    pts = [r.copy()]

    rho_prev = rho_interp(r)
    if not np.isfinite(rho_prev):
        return np.array(pts), np.nan

    def f(rr):
        B = B_interp(rr)
        bz = B[2]
        if not np.isfinite(B).all():
            return None
        if eps_Bz == 0.0:
            if bz == 0.0:
                return None
        else:
            if bz <= eps_Bz:
                return None
        return B / bz

    I = 0.0

    for _ in range(max_steps):
        if (dz_step > 0 and r[2] >= z_target) or (dz_step < 0 and r[2] <= z_target):
            break

        k1 = f(r)
        if k1 is None: break

        k2 = f(r + 0.5*dz_step*k1)
        if k2 is None: break

        k3 = f(r + 0.5*dz_step*k2)
        if k3 is None: break

        k4 = f(r + dz_step*k3)
        if k4 is None: break

        r_new = r + (dz_step/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        r_new[2] = r[2] + dz_step  # exact z step

        rho_new = rho_interp(r_new)
        if not np.isfinite(rho_new):
            break

        I += 0.5*(rho_prev + rho_new)*dz_step

        r = r_new
        rho_prev = rho_new
        pts.append(r.copy())

    return float(I)


# ---------- vectorised trilinear interpolation on a uniform grid ----------

def _interp_scalar_trilin(F, x, y, z, pts):
    """
    F: (nx,ny,nz)
    pts: (N,3)
    returns: vals (N,), valid (N,)
    assumes x,y,z are uniform (linspace-like)
    """
    x0, y0, z0 = float(x[0]), float(y[0]), float(z[0])
    dx, dy, dz = float(x[1]-x[0]), float(y[1]-y[0]), float(z[1]-z[0])

    nx, ny, nz_ = F.shape
    px, py, pz = pts[:, 0], pts[:, 1], pts[:, 2]

    ix = (px - x0) / dx
    iy = (py - y0) / dy
    iz = (pz - z0) / dz

    i = np.floor(ix).astype(np.int64)
    j = np.floor(iy).astype(np.int64)
    k = np.floor(iz).astype(np.int64)

    valid = (i >= 0) & (i < nx-1) & (j >= 0) & (j < ny-1) & (k >= 0) & (k < nz_-1)

    out = np.full(px.shape, np.nan, dtype=float)
    if not np.any(valid):
        return out, valid

    i0 = i[valid]; j0 = j[valid]; k0 = k[valid]
    fx = (ix[valid] - i0).astype(float)
    fy = (iy[valid] - j0).astype(float)
    fz = (iz[valid] - k0).astype(float)

    c000 = F[i0,   j0,   k0  ]
    c100 = F[i0+1, j0,   k0  ]
    c010 = F[i0,   j0+1, k0  ]
    c110 = F[i0+1, j0+1, k0  ]
    c001 = F[i0,   j0,   k0+1]
    c101 = F[i0+1, j0,   k0+1]
    c011 = F[i0,   j0+1, k0+1]
    c111 = F[i0+1, j0+1, k0+1]

    c00 = c000*(1-fx) + c100*fx
    c10 = c010*(1-fx) + c110*fx
    c01 = c001*(1-fx) + c101*fx
    c11 = c011*(1-fx) + c111*fx

    c0 = c00*(1-fy) + c10*fy
    c1 = c01*(1-fy) + c11*fy

    out_valid = c0*(1-fz) + c1*fz
    out[valid] = out_valid
    return out, valid


def _interp_vector_trilin(V, x, y, z, pts):
    """
    V: (nx,ny,nz,3)
    returns vecs (N,3), valid (N,)
    """
    vx, v1 = _interp_scalar_trilin(V[..., 0], x, y, z, pts)
    vy, v2 = _interp_scalar_trilin(V[..., 1], x, y, z, pts)
    vz, v3 = _interp_scalar_trilin(V[..., 2], x, y, z, pts)
    valid = v1 & v2 & v3
    return np.stack([vx, vy, vz], axis=-1), valid


# ---------- main: integrate rho dz along field lines for a seed grid ----------

def integrate_rho_dz_seed_grid(B, rho, x, y, z,
                               seed_x, seed_y, seed_z,
                               dz_step=None, z_target=None,
                               eps_Bz=1e-14,
                               max_steps=None,
                               batch_size=8192):
    """
    Vectorised/batched equal-Δz integration:

      dr/dz = B/Bz   (assume Bz>0 everywhere; stop if Bz<=eps_Bz or leaves domain)
      I = ∫ rho(r(z)) dz  (trapezoid in z)

    Inputs:
      B:   (nx,ny,nz,3)
      rho: (nx,ny,nz) scalar integrand density already computed (e.g. A·B/Bz)
      seed_x, seed_y: 1D arrays defining seed plane
      seed_z: float
      dz_step: default uses grid z spacing
      z_target: default z.max()
      max_steps: optional cap on steps (otherwise derived from target)
      batch_size: tune memory/speed

    Returns:
      I2: (len(seed_x), len(seed_y)) float array (nan if seed fails immediately)
    """
    if dz_step is None:
        dz_step = float(z[1] - z[0])
    dz_step = float(dz_step)

    if z_target is None:
        z_target = float(z.max())
    z_target = float(z_target)

    if max_steps is None:
        max_steps = int(np.ceil((z_target - seed_z) / dz_step))
    max_steps = int(max_steps)

    # seeds -> (N,3)
    Xs, Ys = np.meshgrid(seed_x, seed_y, indexing="ij")
    Ntot = Xs.size
    seeds = np.column_stack([Xs.ravel(), Ys.ravel(), np.full(Ntot, float(seed_z))])

    I_all = np.full(Ntot, np.nan, dtype=float)

    # RHS for RK4: f(r)=dr/dz = B/Bz
    def f(rpts):
        Bp, vmask = _interp_vector_trilin(B, x, y, z, rpts)
        bz = Bp[:, 2]
        ok = vmask & np.isfinite(Bp).all(axis=1) & (bz > eps_Bz)
        out = np.full_like(Bp, np.nan, dtype=float)
        out[ok] = Bp[ok] / bz[ok, None]
        return out, ok

    def rho_at(rpts):
        rv, rmask = _interp_scalar_trilin(rho, x, y, z, rpts)
        ok = rmask & np.isfinite(rv)
        out = np.full(rv.shape, np.nan, dtype=float)
        out[ok] = rv[ok]
        return out, ok

    for s0 in range(0, Ntot, batch_size):
        s1 = min(Ntot, s0 + batch_size)
        r = seeds[s0:s1].copy()
        m = r.shape[0]

        # initial rho
        rho_prev, ok_prev = rho_at(r)
        active = ok_prev.copy()

        I = np.zeros(m, dtype=float)

        for _ in range(max_steps):
            if not np.any(active):
                break

            k1, ok1 = f(r)
            ok = active & ok1
            if not np.any(ok):
                active[:] = False
                break

            r2 = r.copy()
            r2[ok] = r[ok] + 0.5 * dz_step * k1[ok]
            k2, ok2 = f(r2)
            ok &= ok2

            r3 = r.copy()
            r3[ok] = r[ok] + 0.5 * dz_step * k2[ok]
            k3, ok3 = f(r3)
            ok &= ok3

            r4 = r.copy()
            r4[ok] = r[ok] + dz_step * k3[ok]
            k4, ok4 = f(r4)
            ok &= ok4

            r_new = r.copy()
            r_new[ok] = r[ok] + (dz_step/6.0) * (k1[ok] + 2*k2[ok] + 2*k3[ok] + k4[ok])
            r_new[ok, 2] = r[ok, 2] + dz_step  # exact z stepping

            rho_new = np.full(m, np.nan, dtype=float)
            ok_rho = np.zeros(m, dtype=bool)
            rho_new_ok, ok_rho_ok = rho_at(r_new[ok])
            rho_new[ok] = rho_new_ok
            ok_rho[ok] = ok_rho_ok

            ok_all = ok & ok_rho & np.isfinite(rho_prev)
            I[ok_all] += 0.5 * (rho_prev[ok_all] + rho_new[ok_all]) * dz_step

            r = r_new
            rho_prev = rho_new
            active = ok_all

            if dz_step > 0:
                done = r[:, 2] >= z_target
            else:
                done = r[:, 2] <= z_target
            active &= ~done

            if not np.any(active):
                break

        # seeds that never became active keep NaN; others store I
        I_all[s0:s1] = np.where(ok_prev, I, np.nan)

    return I_all.reshape(len(seed_x), len(seed_y))



def _simpson_mixed_weights(n, h=1.0):
    """
    1D weights for equally spaced quadrature:
      - Simpson 1/3 if n-1 is even
      - Simpson 1/3 + Simpson 3/8 on the last 3 intervals if n-1 is odd
    """
    if n < 2:
        raise ValueError("Need at least 2 points.")
    if n == 2:
        return h * np.array([0.5, 0.5], dtype=float)
    if n == 3:
        return (h / 3.0) * np.array([1.0, 4.0, 1.0], dtype=float)
    if n == 4:
        return (3.0 * h / 8.0) * np.array([1.0, 3.0, 3.0, 1.0], dtype=float)

    w = np.zeros(n, dtype=float)
    nint = n - 1

    if nint % 2 == 0:
        w[0] = 1.0
        w[-1] = 1.0
        w[1:-1:2] = 4.0
        w[2:-1:2] = 2.0
        w *= h / 3.0
    else:
        m = n - 3  # first block uses points 0,...,m-1

        w13 = np.zeros(n, dtype=float)
        w13[0] = 1.0
        w13[m - 1] = 1.0
        w13[1:m - 1:2] = 4.0
        w13[2:m - 1:2] = 2.0
        w += (h / 3.0) * w13

        w[-4:] += (3.0 * h / 8.0) * np.array([1.0, 3.0, 3.0, 1.0])

    return w


def _to_2d(arr):
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0]
    if arr.ndim == 2:
        return arr
    raise ValueError("Array must have shape (nx, ny, 1) or (nx, ny)")


def _interp_to_shape(b, target_shape):
    """
    Fast bilinear interpolation from b -> target_shape
    assuming both grids span the same domain.
    NaNs treated as zero.
    """
    b = np.nan_to_num(b, nan=0.0, posinf=0.0, neginf=0.0)

    nx2, ny2 = b.shape
    nx1, ny1 = target_shape

    # fractional coordinates in source grid
    x = np.linspace(0, nx2 - 1, nx1)
    y = np.linspace(0, ny2 - 1, ny1)

    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)

    x1i = np.clip(x0 + 1, 0, nx2 - 1)
    y1i = np.clip(y0 + 1, 0, ny2 - 1)

    wx = x - x0
    wy = y - y0

    # expand for broadcasting
    wx = wx[:, None]
    wy = wy[None, :]

    f00 = b[x0[:, None], y0[None, :]]
    f01 = b[x0[:, None], y1i[None, :]]
    f10 = b[x1i[:, None], y0[None, :]]
    f11 = b[x1i[:, None], y1i[None, :]]

    return (
        (1 - wx) * (1 - wy) * f00 +
        (1 - wx) * wy       * f01 +
        wx       * (1 - wy) * f10 +
        wx       * wy       * f11
    )

def integrate_product_2d(a, b, dx=1.0, dy=1.0):
    """
    Multiply two arrays and integrate over x,y.

    - `a` defines the target grid
    - if `b` has a different shape, it is interpolated onto `a`'s grid
    - NaNs are treated as zero
    - integration uses Simpson 1/3 with 3/8 fallback

    Parameters
    ----------
    a, b : ndarray
        Shape (nx, ny, 1) or (nx, ny)
    dx, dy : float
        Grid spacings for the first array's grid

    Returns
    -------
    float
    """
    a2 = _to_2d(a)
    b2 = _to_2d(b)

    if b2.shape != a2.shape:
        b2 = _interp_to_shape(b2, a2.shape)

    prod = np.nan_to_num(a2, nan=0.0, posinf=0.0, neginf=0.0) * \
           np.nan_to_num(b2, nan=0.0, posinf=0.0, neginf=0.0)

    wx = _simpson_mixed_weights(a2.shape[0], dx)
    wy = _simpson_mixed_weights(a2.shape[1], dy)

    return np.einsum("i,ij,j->", wx, prod, wy)
