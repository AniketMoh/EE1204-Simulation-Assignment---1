"""
q1_parallel_plate_capacitor.py
==============================
EE1204 Simulation Assignment 1 — Question 1
2D Parallel-Plate Capacitor with Finite Dimensions

Solves Laplace's equation  ∇²V = 0  on a 120×120 grid using Gauss–Seidel
relaxation.  Two cases are computed:
    (a) Uniform dielectric κ = 1 (vacuum)
    (b) Dielectric interface at y = 0  with  κ₁ = 4 (top)  /  κ₂ = 1 (bottom)

Figures produced (saved to ./figures/q1/):
    fig1 — Equipotential contours + field streamlines (uniform κ)
    fig2 — |E| along vertical midline (uniform κ)
    fig3 — Equipotential contours + field lines (dielectric interface)
    fig4 — |E| comparison: uniform vs dielectric
    fig5 — Convergence history of Gauss–Seidel solver
    fig6 — |E| heat map (log scale, uniform κ)

Author : Aniket (EE25BTECH11007)
Course : EE1204 — Engineering Electromagnetics, IIT Hyderabad
Date   : March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

# Use non-interactive backend so the script works on headless servers
mpl.use('Agg')

# Import shared style (sets rcParams + defines colour constants)
import style  # noqa: F401,E402

# ═══════════════════════════════════════════════════════════════
#  1.  SIMULATION PARAMETERS
# ═══════════════════════════════════════════════════════════════

N      = 120              # number of grid points along each axis
L      = 5.0e-3           # domain side length [m]  →  5 mm × 5 mm
h      = L / (N - 1)      # grid spacing ≈ 42 µm
ITERS  = 5000             # number of Gauss–Seidel iterations

# Plate voltages
VA = +5.0                 # Plate A (top plate) voltage [V]
VB = -5.0                 # Plate B (bottom plate) voltage [V]

# Plate y-positions  →  ±5/3 mm  ≈  ±1.667 mm
yA =  5.0 / 3.0 * 1e-3   # Plate A y-coordinate [m]
yB = -5.0 / 3.0 * 1e-3   # Plate B y-coordinate [m]

# Plate x-extent  →  full width of domain
xlo = -2.5e-3             # left edge of plates [m]
xhi =  2.5e-3             # right edge of plates [m]

# ═══════════════════════════════════════════════════════════════
#  2.  GRID CONSTRUCTION
# ═══════════════════════════════════════════════════════════════

x = np.linspace(-L / 2, L / 2, N)   # x-coordinates [m]
y = np.linspace(-L / 2, L / 2, N)   # y-coordinates [m]
X, Y = np.meshgrid(x, y)            # 2D coordinate arrays

# Helper: find the grid index nearest to a given coordinate value
def idx(arr, val):
    """Return index of element in *arr* closest to *val*."""
    return int(np.argmin(np.abs(arr - val)))

# Grid-row indices for the two plates
jA = idx(y, yA)   # row index of Plate A
jB = idx(y, yB)   # row index of Plate B

# Grid-column range for the plate extent
iL = idx(x, xlo)  # left-most column of plates
iR = idx(x, xhi)  # right-most column of plates

# Create output directory
os.makedirs('figures/q1', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  3.  GAUSS–SEIDEL SOLVER  —  UNIFORM DIELECTRIC  (κ = 1)
# ═══════════════════════════════════════════════════════════════
#
#  Update rule (five-point stencil, Eq. 3 in the report):
#
#      V[j, i] = ¼ ( V[j, i+1] + V[j, i-1] + V[j+1, i] + V[j-1, i] )
#
#  Implemented row-by-row in a vectorised fashion so that left and
#  bottom neighbours use the *current* (already-updated) values
#  → Gauss–Seidel, not Jacobi.

V = np.zeros((N, N))                   # initial guess: V = 0 everywhere
V[jA, iL:iR + 1] = VA                  # enforce Plate A voltage
V[jB, iL:iR + 1] = VB                  # enforce Plate B voltage

residuals = []                          # store max|ΔV| every 100 iterations

for k in range(ITERS):
    V_old = V.copy()                    # snapshot for convergence check

    # --- Row-wise vectorised Gauss–Seidel sweep ---
    for j in range(1, N - 1):
        V[j, 1:-1] = 0.25 * (
            V[j, 2:]      +             # right neighbour (old)
            V[j, :-2]     +             # left  neighbour (already updated)
            V[j + 1, 1:-1] +            # upper neighbour (old)
            V[j - 1, 1:-1]              # lower neighbour (already updated)
        )

    # --- Re-apply Dirichlet boundary conditions ---
    V[0, :]  = 0.0                      # bottom wall  → grounded
    V[-1, :] = 0.0                      # top wall     → grounded
    V[:, 0]  = 0.0                      # left wall    → grounded
    V[:, -1] = 0.0                      # right wall   → grounded
    V[jA, iL:iR + 1] = VA              # Plate A
    V[jB, iL:iR + 1] = VB              # Plate B

    # --- Convergence tracking (every 100 iterations) ---
    if k % 100 == 0:
        residuals.append(np.max(np.abs(V - V_old)))

# ═══════════════════════════════════════════════════════════════
#  4.  ELECTRIC FIELD COMPUTATION  (uniform κ)
# ═══════════════════════════════════════════════════════════════
#
#  E = -∇V   →   Ex = -∂V/∂x ,   Ey = -∂V/∂y
#  numpy.gradient returns (∂V/∂y, ∂V/∂x) for a 2D array

Ey_u, Ex_u = np.gradient(V, h)
Ex_u *= -1
Ey_u *= -1
Em_u = np.sqrt(Ex_u**2 + Ey_u**2)     # field magnitude

# ═══════════════════════════════════════════════════════════════
#  5.  DIELECTRIC INTERFACE  (κ₁ = 4 above y = 0, κ₂ = 1 below)
# ═══════════════════════════════════════════════════════════════
#
#  At the interface row the update rule changes to enforce
#  continuity of the normal component of D:
#
#      κ₁ ∂V/∂y|₀₊ = κ₂ ∂V/∂y|₀₋
#
#  Discretising → V_m = (κ₁ V_{m+1} + κ₂ V_{m-1}) / (κ₁ + κ₂)

Vd = np.zeros((N, N))                 # fresh grid for dielectric case
Vd[jA, iL:iR + 1] = VA
Vd[jB, iL:iR + 1] = VB

k1, k2 = 4.0, 1.0                     # relative permittivities
jI = idx(y, 0.0)                       # interface row index

# Boolean mask: True at plate grid points (these must not be overwritten)
plate = np.zeros((N, N), dtype=bool)
plate[jA, iL:iR + 1] = True
plate[jB, iL:iR + 1] = True

for iteration in range(ITERS):
    for j in range(1, N - 1):
        # Columns to update (skip plate nodes)
        cols = np.arange(1, N - 1)
        if plate[j].any():
            cols = cols[~plate[j, 1:-1]]

        if j == jI:
            # --- Interface row: modified stencil (Eq. 7 in report) ---
            Vd[j, cols] = (k1 * Vd[j + 1, cols] + k2 * Vd[j - 1, cols]) / (k1 + k2)
        else:
            # --- Standard four-neighbour average ---
            Vd[j, cols] = 0.25 * (
                Vd[j, cols + 1] + Vd[j, cols - 1] +
                Vd[j + 1, cols] + Vd[j - 1, cols]
            )

    # Re-apply BCs
    Vd[0, :] = 0; Vd[-1, :] = 0; Vd[:, 0] = 0; Vd[:, -1] = 0
    Vd[jA, iL:iR + 1] = VA
    Vd[jB, iL:iR + 1] = VB

# Electric field for dielectric case
Ey_d, Ex_d = np.gradient(Vd, h)
Ex_d *= -1; Ey_d *= -1
Em_d = np.sqrt(Ex_d**2 + Ey_d**2)

# ═══════════════════════════════════════════════════════════════
#  6.  PLOTTING
# ═══════════════════════════════════════════════════════════════

mm = 1e3   # multiplier to convert metres → millimetres for axis labels


def draw_plates(ax):
    """Draw the two plates as thick coloured lines on *ax*."""
    ax.plot([xlo * mm, xhi * mm], [y[jA] * mm] * 2,
            color=style.PLATE_POS, lw=4, solid_capstyle='butt',
            label=f'Plate A (+{VA:.0f} V)')
    ax.plot([xlo * mm, xhi * mm], [y[jB] * mm] * 2,
            color=style.PLATE_NEG, lw=4, solid_capstyle='butt',
            label=f'Plate B ({VB:.0f} V)')


levs = np.linspace(-5, 5, 21)   # contour levels for potential

# ── Fig 1 : Equipotential contours + field streamlines (uniform κ) ──
fig, ax = plt.subplots(figsize=(7.5, 6.2))
cf = ax.contourf(X * mm, Y * mm, V, levels=levs, cmap=style.CMAP_POT)
cs = ax.contour(X * mm, Y * mm, V, levels=levs, colors='k',
                linewidths=0.3, alpha=0.4)
ax.clabel(cs, cs.levels[::2], fontsize=7, fmt='%.1f V')

# Streamlines coloured by |E| on a log scale
lnorm = mpl.colors.LogNorm(vmin=max(100, Em_u[Em_u > 0].min()),
                            vmax=Em_u.max())
ax.streamplot(X * mm, Y * mm, Ex_u, Ey_u, color=Em_u,
              cmap=style.CMAP_FIELD, norm=lnorm,
              density=1.8, linewidth=0.65, arrowsize=0.7)
draw_plates(ax)
fig.colorbar(cf, ax=ax, label='Electric Potential  $V$  (V)', pad=0.02)
ax.set_xlabel('$x$  (mm)'); ax.set_ylabel('$y$  (mm)')
ax.set_title(r'Equipotential Contours and Electric Field Lines  ($\kappa = 1$)')
ax.legend(loc='upper right', framealpha=0.92, fontsize=8)
ax.set_aspect('equal')
fig.savefig('figures/q1/fig1_equipotential_field_uniform.png')
plt.close()

# ── Fig 2 : |E| along vertical midline (uniform κ) ──
mid = N // 2                       # midline column index
E_mid_u = Em_u[:, mid]             # field magnitude along x = 0
d_gap = y[jA] - y[jB]             # plate separation [m]
E0 = (VA - VB) / d_gap            # ideal uniform field [V/m]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(y * mm, E_mid_u, color=style.PLATE_POS, lw=1.5,
        label='Simulated $|\\mathbf{E}|$')
ax.axhline(E0, color=style.ACCENT2, ls=':', lw=1.3,
           label=f'Ideal  $E_0 = \\Delta V / d \\approx {E0:.0f}$  V/m')
ax.axvline(y[jA] * mm, color=style.PLATE_NEG, ls='--', lw=0.9,
           alpha=0.6, label='Plate A')
ax.axvline(y[jB] * mm, color=style.ACCENT, ls='--', lw=0.9,
           alpha=0.6, label='Plate B')
ax.set_xlabel('$y$  (mm)'); ax.set_ylabel('$|\\mathbf{E}|$  (V/m)')
ax.set_title(r'Electric Field Magnitude Along Vertical Midline  '
             r'($x = 0$,  $\kappa = 1$)')
ax.legend(fontsize=9); ax.grid(True); ax.set_xlim(-2.5, 2.5)
fig.savefig('figures/q1/fig2_midline_field_uniform.png')
plt.close()

# ── Fig 3 : Equipotential + field lines (dielectric interface) ──
fig, ax = plt.subplots(figsize=(7.5, 6.2))
cf = ax.contourf(X * mm, Y * mm, Vd, levels=levs, cmap=style.CMAP_POT)
cs = ax.contour(X * mm, Y * mm, Vd, levels=levs, colors='k',
                linewidths=0.3, alpha=0.4)
ax.clabel(cs, cs.levels[::2], fontsize=7, fmt='%.1f V')
ax.streamplot(X * mm, Y * mm, Ex_d, Ey_d, color='k',
              density=1.5, linewidth=0.55, arrowsize=0.7)
# Shade dielectric regions
ax.axhspan(0, 2.5, alpha=0.07, color='blue')
ax.axhspan(-2.5, 0, alpha=0.07, color='orange')
ax.axhline(0, color=style.ACCENT, ls='--', lw=1.5,
           label='Dielectric interface  ($y=0$)')
draw_plates(ax)
fig.colorbar(cf, ax=ax, label='Electric Potential  $V$  (V)', pad=0.02)
ax.set_xlabel('$x$  (mm)'); ax.set_ylabel('$y$  (mm)')
ax.set_title(r'Equipotential Contours and Field Lines  '
             r'($\kappa_1=4$,  $\kappa_2=1$)')
ax.legend(loc='upper right', framealpha=0.92, fontsize=7.5)
ax.set_aspect('equal')
fig.savefig('figures/q1/fig3_equipotential_dielectric.png')
plt.close()

# ── Fig 4 : |E| comparison — uniform vs dielectric ──
E_mid_d = Em_d[:, mid]

fig, ax = plt.subplots(figsize=(7.5, 4.8))
ax.plot(y * mm, E_mid_u, color=style.PLATE_POS, lw=1.5,
        label=r'Uniform  $\kappa=1$')
ax.plot(y * mm, E_mid_d, color=style.PLATE_NEG, ls='--', lw=1.5,
        label=r'Interface  $\kappa_1\!=\!4$ / $\kappa_2\!=\!1$')
ax.axvline(0, color=style.ACCENT, ls='--', lw=1,
           label='Interface  $y=0$')
ax.axhline(E0, color=style.ACCENT2, ls=':', lw=1,
           label=f'$E_0 \\approx {E0:.0f}$  V/m')
ax.set_xlabel('$y$  (mm)'); ax.set_ylabel('$|\\mathbf{E}|$  (V/m)')
ax.set_title(r'Field Magnitude Comparison: Uniform vs.\ Dielectric Interface')
ax.legend(fontsize=9); ax.grid(True); ax.set_xlim(-2.5, 2.5)
fig.savefig('figures/q1/fig4_midline_field_comparison.png')
plt.close()

# ── Fig 5 : Convergence history ──
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogy(np.arange(len(residuals)) * 100, residuals,
            color=style.PLATE_POS, lw=1.3)
ax.set_xlabel('Iteration')
ax.set_ylabel('Max residual  $|\\Delta V|$  (V)')
ax.set_title(r'Convergence of Gauss--Seidel Solver  ($\kappa = 1$)')
ax.grid(True)
fig.savefig('figures/q1/fig5_convergence.png')
plt.close()

# ── Fig 6 : |E| heat map (log scale) ──
fig, ax = plt.subplots(figsize=(7, 5.8))
im = ax.pcolormesh(X * mm, Y * mm, np.log10(Em_u + 1),
                   cmap=style.CMAP_HEAT, shading='auto')
draw_plates(ax)
fig.colorbar(im, ax=ax,
             label=r'$\log_{10}(|\mathbf{E}|+1)$  [V/m]', pad=0.02)
ax.set_xlabel('$x$  (mm)'); ax.set_ylabel('$y$  (mm)')
ax.set_title(r'Electric Field Magnitude  (log scale,  $\kappa = 1$)')
ax.legend(fontsize=8, loc='upper right'); ax.set_aspect('equal')
fig.savefig('figures/q1/fig6_field_heatmap.png')
plt.close()

print("Q1 complete — 6 figures saved to figures/q1/")
