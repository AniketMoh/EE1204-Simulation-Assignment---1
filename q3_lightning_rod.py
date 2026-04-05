"""
q3_lightning_rod.py
===================
EE1204 Simulation Assignment 1 — Question 3
Electric Field Enhancement at a Sharp Conductor (Lightning Rod)

Simulates the electrostatic field produced by a grounded conducting
needle placed between a ground plane (V = 0) and a charged cloud
(V = 100 V), using Gauss–Seidel relaxation on a 100×100 grid.

The needle occupies the centre column from the ground (row 0) up to
the middle of the grid (row 50), fixed at V = 0.  The solver iterates
until the maximum residual falls below 10⁻⁶ V.

Key result:  Field enhancement factor β ≈ 7.5
             (theory for prolate spheroid: β ≈ 10.9)

Figures produced (saved to ./figures/q3/):
    fig1 — Potential heat map + field vectors
    fig2 — log₁₀|E| heat map
    fig3 — |E| along vertical centre line
    fig4 — Streamlines on potential contours
    fig5 — Convergence plot

Author : Aniket (EE25BTECH11007)
Course : EE1204 — Engineering Electromagnetics, IIT Hyderabad
Date   : March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.use('Agg')

import style  # noqa: F401,E402

# ═══════════════════════════════════════════════════════════════
#  1.  PARAMETERS
# ═══════════════════════════════════════════════════════════════

N        = 100          # grid size  →  100 × 100
V_GND    = 0.0          # ground potential  [V]
V_CLOUD  = 100.0        # cloud potential   [V]
MAX_ITER = 10000        # maximum iterations
TOL      = 1e-6         # convergence tolerance  [V]
CHK      = 500          # check convergence every CHK iterations

centre   = N // 2       # needle column index  = 50
tip_row  = N // 2       # needle tip row (matrix row; 0 = top = cloud)

os.makedirs('figures/q3', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  2.  INITIAL CONDITIONS AND BOUNDARY SETUP
# ═══════════════════════════════════════════════════════════════
#
#  Matrix convention:
#      Row 0    =  top of domain  =  cloud  (V = 100 V)
#      Row N-1  =  bottom         =  ground (V = 0 V)
#
#  Side boundaries carry a linear ramp from V_GND to V_CLOUD,
#  simulating the ambient uniform field that would exist without
#  the needle.
#
#  The needle is a vertical conducting line at column = centre,
#  from row = tip_row (mid-height) down to row = N−1 (ground),
#  all fixed at V = 0.

V = np.zeros((N, N))

# Top boundary → cloud voltage
V[0, :] = V_CLOUD

# Bottom boundary → ground voltage
V[-1, :] = V_GND

# Side boundaries → linear ramp
for j in range(N):
    frac = (N - 1 - j) / (N - 1)   # 0 at bottom (j=N-1), 1 at top (j=0)
    V[j, 0]    = V_GND + frac * (V_CLOUD - V_GND)
    V[j, -1]   = V_GND + frac * (V_CLOUD - V_GND)

# Needle: column = centre, from tip_row to bottom, all at V = 0
needle_rows = list(range(tip_row, N))
for r in needle_rows:
    V[r, centre] = V_GND

# ═══════════════════════════════════════════════════════════════
#  3.  GAUSS–SEIDEL SOLVER
# ═══════════════════════════════════════════════════════════════
#
#  Standard five-point stencil (Eq. 5 in report), skipping
#  needle nodes which are held at V = 0.  After each sweep,
#  all Dirichlet BCs are reimposed.

residuals = []          # store (iteration, max_residual) pairs
conv_iter = MAX_ITER    # will be updated if convergence is reached

print("Running Gauss–Seidel solver (100×100, lightning rod)...")
for k in range(1, MAX_ITER + 1):
    V_prev = V.copy()

    # Interior update (pure Python loop — acceptable for 100×100)
    for j in range(1, N - 1):
        for i in range(1, N - 1):
            # Skip needle nodes
            if i == centre and j >= tip_row:
                continue
            V[j, i] = 0.25 * (
                V[j, i + 1] +       # right
                V[j, i - 1] +       # left
                V[j + 1, i] +       # below
                V[j - 1, i]         # above
            )

    # Re-apply all Dirichlet boundary conditions
    V[0, :]  = V_CLOUD
    V[-1, :] = V_GND
    for j in range(N):
        frac = (N - 1 - j) / (N - 1)
        V[j, 0]  = V_GND + frac * (V_CLOUD - V_GND)
        V[j, -1] = V_GND + frac * (V_CLOUD - V_GND)
    for r in needle_rows:
        V[r, centre] = V_GND

    # Convergence check
    if k % CHK == 0:
        max_delta = np.max(np.abs(V - V_prev))
        residuals.append((k, max_delta))
        print(f"  iter {k:5d}   max|ΔV| = {max_delta:.3e}")
        if max_delta < TOL:
            conv_iter = k
            print(f"  >>> Converged at iteration {k}")
            break

# ═══════════════════════════════════════════════════════════════
#  4.  ELECTRIC FIELD COMPUTATION
# ═══════════════════════════════════════════════════════════════
#
#  E = −∇V   computed via numpy.gradient.
#
#  Note on sign conventions:
#      Matrix row j increases downward (cloud→ground).
#      Physical y increases upward.
#      So E_y(up) = +∂V/∂j  (gradient w.r.t. row index).

dVdj, dVdi = np.gradient(V, 1.0, 1.0)
Ex  = -dVdi          # x-component of E
Ey  =  dVdj          # y-component (upward positive)
Emag = np.hypot(Ex, Ey)

# ═══════════════════════════════════════════════════════════════
#  5.  FLIP FOR DISPLAY  (ground at bottom)
# ═══════════════════════════════════════════════════════════════
#
#  The matrix has row 0 = top (cloud), but we want to display
#  with ground at the bottom of the figure.

def flip(A):
    """Flip a 2D array vertically (top ↔ bottom)."""
    return A[::-1, :]

Vf   = flip(V)
Exf  = flip(Ex)
Eyf  = flip(-Ey)     # flip sign because y-axis is inverted
Emf  = flip(Emag)

# Display coordinates
needle_top_disp = (N - 1) - tip_row   # = 49

# ═══════════════════════════════════════════════════════════════
#  6.  ENHANCEMENT FACTOR CALCULATION
# ═══════════════════════════════════════════════════════════════

# Ambient uniform field (without needle)
E0 = V_CLOUD / (N - 1)               # ≈ 1.01 V/grid-unit

# Field along the vertical centre line (flipped for display coords)
E_centre = Emag[:, centre][::-1]

# Field at one cell above the tip
E_tip = E_centre[needle_top_disp + 1]

# Numerical enhancement factor
beta_num = E_tip / E0

# Theoretical prolate-spheroid estimate:  β = a / (b ln(2a/b))
a_needle = 50.0     # half-length of needle  [grid units]
b_tip    = 1.0      # effective tip radius   [grid units]
beta_theory = a_needle / (b_tip * np.log(2 * a_needle / b_tip))

print(f"\n  E_tip       = {E_tip:.3f} V/grid-unit")
print(f"  E_0         = {E0:.3f} V/grid-unit")
print(f"  β_numerical = {beta_num:.1f}")
print(f"  β_theory    = {beta_theory:.1f}")

# ═══════════════════════════════════════════════════════════════
#  7.  PLOTTING
# ═══════════════════════════════════════════════════════════════

ext = [0, N - 1, 0, N - 1]   # extent for imshow


def draw_needle(ax):
    """Draw the needle and its tip on *ax*."""
    ax.plot([centre, centre], [0, needle_top_disp],
            color=style.ACCENT, lw=3.5, solid_capstyle='butt',
            label='Needle ($V=0$)')
    ax.plot(centre, needle_top_disp, '^', color='lime', ms=9,
            zorder=6, label='Needle tip')


# ── Fig 1 : Potential heat map + field vectors ──
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(Vf, extent=ext, cmap='plasma', origin='lower',
               aspect='equal')
# White equipotential contour lines
clines = ax.contour(Vf, levels=np.arange(0, 101, 10), colors='white',
                     linewidths=0.5, alpha=0.6, extent=ext)
ax.clabel(clines, clines.levels, fontsize=7, fmt='%.0f V')
# Quiver arrows (subsampled)
sk = 5
xs = np.arange(0, N, sk); ys = np.arange(0, N, sk)
Xq, Yq = np.meshgrid(xs, ys)
Exq = Exf[::sk, ::sk]; Eyq = Eyf[::sk, ::sk]
Eq = np.maximum(np.hypot(Exq, Eyq), 1e-10)
ax.quiver(Xq, Yq, Exq / Eq, Eyq / Eq, color='cyan',
          scale=30, width=0.003, alpha=0.7)
draw_needle(ax)
fig.colorbar(im, ax=ax, label='Electric Potential  $V$  (V)', pad=0.02)
ax.set_xlabel('Grid column ($x$)')
ax.set_ylabel('Grid row ($y$)')
ax.set_title('Potential Distribution with Electric Field Vectors')
ax.legend(loc='upper right', fontsize=9, framealpha=0.9)
fig.savefig('figures/q3/fig1_potential_field.png')
plt.close()

# ── Fig 2 : log₁₀|E| heat map ──
fig, ax = plt.subplots(figsize=(7.5, 6.5))
im = ax.imshow(np.log10(Emf + 1e-10), extent=ext,
               cmap=style.CMAP_HEAT, origin='lower', aspect='equal')
draw_needle(ax)
fig.colorbar(im, ax=ax,
             label=r'$\log_{10}|\mathbf{E}|$  [V / grid unit]', pad=0.02)
ax.set_xlabel('Grid column'); ax.set_ylabel('Grid row')
ax.set_title(r'Electric Field Magnitude  ($\log_{10}$ scale)')
ax.legend(fontsize=9, loc='upper right')
fig.savefig('figures/q3/fig2_field_heatmap.png')
plt.close()

# ── Fig 3 : |E| along centre line ──
fig, ax = plt.subplots(figsize=(7.5, 4.8))
yp = np.arange(N)
ax.plot(yp, E_centre, color=style.PLATE_NEG, lw=1.4,
        label=r'$|\mathbf{E}|$ on centre line')
ax.axvline(needle_top_disp, color=style.PLATE_POS, ls='--', lw=1.1,
           label='Needle tip')
ax.axvspan(0, needle_top_disp, alpha=0.06, color=style.ACCENT,
           label='Needle body')
ax.set_xlabel('Grid row  ($y$,  0 = ground)')
ax.set_ylabel(r'$|\mathbf{E}|$  (V / grid unit)')
ax.set_title('Field Magnitude Along Vertical Centre Line')
ax.legend(fontsize=9); ax.grid(True)
fig.savefig('figures/q3/fig3_centreline_field.png')
plt.close()

# ── Fig 4 : Streamlines on potential contours ──
fig, ax = plt.subplots(figsize=(7.5, 6.5))
cf = ax.contourf(Vf, levels=np.linspace(0, 100, 21), cmap='plasma',
                 alpha=0.5, extent=ext)
fig.colorbar(cf, ax=ax, label='$V$  (V)', pad=0.02)
ax.streamplot(np.arange(N, dtype=float), np.arange(N, dtype=float),
              Exf, Eyf, color='k', density=2.0,
              linewidth=0.55, arrowsize=0.75)
draw_needle(ax)
ax.set_xlabel('$x$  (grid)'); ax.set_ylabel('$y$  (grid)')
ax.set_title('Potential Contours and E-field Streamlines')
ax.legend(fontsize=9, loc='upper right')
fig.savefig('figures/q3/fig4_streamlines.png')
plt.close()

# ── Fig 5 : Convergence plot ──
iters_r, vals_r = zip(*residuals)

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogy(iters_r, vals_r, 'o-', color=style.PLATE_POS, ms=4, lw=1.2)
ax.axhline(TOL, color=style.PLATE_NEG, ls='--', lw=0.9,
           label=f'Tolerance = $10^{{-6}}$ V')
ax.set_xlabel('Iteration')
ax.set_ylabel(r'Max residual  $|\Delta V|$  (V)')
ax.set_title('Convergence of Gauss\u2013Seidel Solver  (Lightning Rod)')
ax.legend(fontsize=9); ax.grid(True)
fig.savefig('figures/q3/fig5_convergence.png')
plt.close()

print("Q3 complete — 5 figures saved to figures/q3/")
