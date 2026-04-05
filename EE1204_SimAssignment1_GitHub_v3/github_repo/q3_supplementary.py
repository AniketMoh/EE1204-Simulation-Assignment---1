"""
q3_supplementary.py
===================
EE1204 — Question 3 Supplementary Analysis

Additional figures beyond the base simulation:
    fig6  — β vs needle height (25%, 50%, 75% of gap)
    fig7  — β vs grid resolution (N = 50, 100, 200) — convergence study
    fig8  — Surface charge density along the needle body
    fig9  — Zoomed-in |E| near the needle tip
    fig10 — Side-by-side comparison: with vs without needle

Author : Aniket Mohapatra (EE25BTECH11007)
Date   : 4 April 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.use('Agg')
import style  # noqa: F401,E402

os.makedirs('figures/q3', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  HELPER: Gauss–Seidel solver for the lightning rod problem
# ═══════════════════════════════════════════════════════════════

def solve_lightning(N, needle_frac, max_iter=10000, tol=1e-6):
    """
    Solve the lightning rod problem on an N×N grid.

    Parameters
    ----------
    N            : grid size
    needle_frac  : fraction of the gap occupied by the needle (0 to 1).
                   e.g., 0.5 = needle reaches mid-height.
    max_iter     : max Gauss–Seidel iterations
    tol          : convergence tolerance

    Returns
    -------
    V     : converged potential array (N×N), row 0 = cloud, row N-1 = ground
    beta  : numerical enhancement factor E_tip / E_0
    n_iter: number of iterations to converge
    """
    V_GND, V_CLOUD = 0.0, 100.0
    centre = N // 2
    # Needle extends from ground (row N-1) upward.
    # tip_row = row index of the needle tip (in matrix coords: 0=top)
    tip_row = N - 1 - int(needle_frac * (N - 1))
    tip_row = max(1, min(tip_row, N - 2))   # clamp to interior

    V = np.zeros((N, N))
    V[0, :] = V_CLOUD
    V[-1, :] = V_GND

    # Side ramps
    for j in range(N):
        f = (N-1-j)/(N-1)
        V[j, 0] = V[j, -1] = V_GND + f*(V_CLOUD - V_GND)

    # Needle
    needle_rows = list(range(tip_row, N))
    for r in needle_rows:
        V[r, centre] = V_GND

    converged_iter = max_iter
    for k in range(1, max_iter + 1):
        Vp = V.copy()
        for j in range(1, N-1):
            for i in range(1, N-1):
                if i == centre and j >= tip_row:
                    continue
                V[j,i] = 0.25*(V[j,i+1]+V[j,i-1]+V[j+1,i]+V[j-1,i])
        # Reimpose BCs
        V[0,:] = V_CLOUD; V[-1,:] = V_GND
        for j in range(N):
            f = (N-1-j)/(N-1)
            V[j,0] = V[j,-1] = V_GND + f*(V_CLOUD-V_GND)
        for r in needle_rows:
            V[r, centre] = V_GND

        if k % 500 == 0:
            md = np.max(np.abs(V-Vp))
            if md < tol:
                converged_iter = k
                break

    # Compute E field and enhancement
    dVdj, dVdi = np.gradient(V, 1.0, 1.0)
    Ex = -dVdi; Ey = dVdj
    Emag = np.hypot(Ex, Ey)

    E0 = V_CLOUD / (N - 1)

    # Field just above the tip (one row closer to cloud = row tip_row - 1)
    E_tip = Emag[tip_row - 1, centre]
    beta = E_tip / E0 if E0 > 0 else 0.0

    return V, beta, converged_iter, Emag, tip_row


# ═══════════════════════════════════════════════════════════════
#  Fig 6 : β vs needle height
# ═══════════════════════════════════════════════════════════════
#
#  Run the solver with needles at 25%, 50%, 75% of the gap height
#  and plot the enhancement factor β for each.

print("Running β vs needle height study...")
fracs = [0.25, 0.50, 0.75]
betas_height = []
N_base = 100

for frac in fracs:
    _, beta, nit, _, _ = solve_lightning(N_base, frac)
    betas_height.append(beta)
    print(f"  needle = {frac*100:.0f}%   β = {beta:.2f}   iters = {nit}")

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot([f*100 for f in fracs], betas_height, 'o-',
        color=style.PLATE_POS, lw=1.5, ms=7, label=r'$\beta_{\mathrm{num}}$')
ax.set_xlabel('Needle height  (\\% of gap)')
ax.set_ylabel(r'Enhancement factor  $\beta$')
ax.set_title(r'Field Enhancement vs.\ Needle Height  ($N = 100$)')
ax.legend(fontsize=10); ax.grid(True)
fig.savefig('figures/q3/fig6_beta_vs_height.png')
plt.close()
print("  fig6 saved")

# ═══════════════════════════════════════════════════════════════
#  Fig 7 : β vs grid resolution (convergence study)
# ═══════════════════════════════════════════════════════════════

print("\nRunning β vs grid resolution study...")
Ns = [30, 50, 75, 100]
betas_grid = []

for Ng in Ns:
    _, beta, nit, _, _ = solve_lightning(Ng, 0.50, max_iter=10000)
    betas_grid.append(beta)
    print(f"  N = {Ng:4d}   β = {beta:.2f}   iters = {nit}")

# Theoretical value for comparison
a, b = 50.0, 1.0
beta_theory = a / (b * np.log(2*a/b))

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(Ns, betas_grid, 'o-', color=style.PLATE_POS, lw=1.5, ms=7,
        label=r'$\beta_{\mathrm{num}}$')
ax.axhline(beta_theory, color=style.PLATE_NEG, ls='--', lw=1.1,
           label=f'$\\beta_{{\\mathrm{{theory}}}} \\approx {beta_theory:.1f}$')
ax.set_xlabel('Grid size  $N$')
ax.set_ylabel(r'Enhancement factor  $\beta$')
ax.set_title(r'Grid Convergence Study  (needle at 50\% height)')
ax.legend(fontsize=10); ax.grid(True)
fig.savefig('figures/q3/fig7_beta_vs_gridsize.png')
plt.close()
print("  fig7 saved")

# ═══════════════════════════════════════════════════════════════
#  Base-case solution for remaining figures
# ═══════════════════════════════════════════════════════════════

N = 100
V_base, beta_base, _, Emag_base, tip_row = solve_lightning(N, 0.50)
centre = N // 2
needle_top_disp = (N-1) - tip_row   # display coordinate of tip

# Flip for display (ground at bottom)
def flip(A): return A[::-1, :]
Vf = flip(V_base)
Emf = flip(Emag_base)

# ═══════════════════════════════════════════════════════════════
#  Fig 8 : Surface charge density along the needle body
# ═══════════════════════════════════════════════════════════════
#
#  σ on the needle surface ≈ −ε₀ ∂V/∂n, where n̂ is the outward
#  normal to the needle.  For a 1-pixel-wide needle at column = centre,
#  the normal points left and right:
#
#      σ_left  ≈ −ε₀ (V[j, centre-1] − V[j, centre]) / h
#      σ_right ≈ −ε₀ (V[j, centre+1] − V[j, centre]) / h
#      σ_total = σ_left + σ_right
#
#  Since V[j, centre] = 0 on the needle:
#      σ ∝ V[j, centre-1] + V[j, centre+1]

needle_display_rows = np.arange(0, needle_top_disp + 1)  # 0 = ground to tip
# In matrix coords (not flipped): needle goes from row N-1 down to tip_row
needle_matrix_rows = np.arange(N-1, tip_row - 1, -1)

sigma_needle = np.zeros(len(needle_matrix_rows))
for idx, jr in enumerate(needle_matrix_rows):
    if 0 < jr < N-1:
        # Normal derivative (left + right)
        sigma_needle[idx] = (V_base[jr, centre-1] + V_base[jr, centre+1])

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(needle_display_rows, sigma_needle[:len(needle_display_rows)],
        color=style.ACCENT, lw=1.5, label=r'$\sigma \propto V_{left} + V_{right}$')
ax.axvline(needle_top_disp, color=style.PLATE_NEG, ls='--', lw=1.1,
           label='Needle tip')
ax.set_xlabel('Position along needle  (grid rows from ground)')
ax.set_ylabel(r'Surface charge density  (a.u.)')
ax.set_title('Induced Surface Charge Along the Needle Body')
ax.legend(fontsize=9); ax.grid(True)
fig.savefig('figures/q3/fig8_needle_surface_charge.png')
plt.close()
print("  fig8 saved")

# ═══════════════════════════════════════════════════════════════
#  Fig 9 : Zoomed-in |E| near the tip
# ═══════════════════════════════════════════════════════════════

# Zoom window: ±12 grid cells around the tip
margin = 12
r_lo = max(needle_top_disp - margin, 0)
r_hi = min(needle_top_disp + margin, N - 1)
c_lo = max(centre - margin, 0)
c_hi = min(centre + margin, N - 1)

Emf_zoom = Emf[r_lo:r_hi+1, c_lo:c_hi+1]
Vf_zoom  = Vf[r_lo:r_hi+1, c_lo:c_hi+1]

fig, ax = plt.subplots(figsize=(7, 6))
ext_z = [c_lo, c_hi, r_lo, r_hi]
im = ax.imshow(np.log10(Emf_zoom + 1e-10), extent=ext_z,
               cmap=style.CMAP_HEAT, origin='lower', aspect='equal')
# Overlay contour lines of V
ax.contour(np.linspace(c_lo, c_hi, Vf_zoom.shape[1]),
           np.linspace(r_lo, r_hi, Vf_zoom.shape[0]),
           Vf_zoom, levels=10, colors='white', linewidths=0.6, alpha=0.7)
# Mark the tip
ax.plot(centre, needle_top_disp, '^', color='lime', ms=12, zorder=6,
        label='Needle tip')
ax.plot([centre, centre], [r_lo, needle_top_disp], color=style.ACCENT,
        lw=3, solid_capstyle='butt')
fig.colorbar(im, ax=ax, label=r'$\log_{10}|\mathbf{E}|$', pad=0.02)
ax.set_xlabel('Grid column'); ax.set_ylabel('Grid row')
ax.set_title('Zoomed View of $|\\mathbf{E}|$ Near Needle Tip')
ax.legend(fontsize=10, loc='upper right')
fig.savefig('figures/q3/fig9_tip_zoom.png')
plt.close()
print("  fig9 saved")

# ═══════════════════════════════════════════════════════════════
#  Fig 10 : Side-by-side — with needle vs without needle
# ═══════════════════════════════════════════════════════════════

# Without needle: just a linear ramp
V_no_needle = np.zeros((N, N))
for j in range(N):
    V_no_needle[j, :] = 100.0 * (N-1-j) / (N-1)
Vf_no = flip(V_no_needle)
dVdj_no, dVdi_no = np.gradient(V_no_needle, 1.0, 1.0)
Emag_no = np.hypot(-dVdi_no, dVdj_no)
Emf_no = flip(Emag_no)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left panel: without needle
ax = axes[0]
im = ax.imshow(Vf_no, extent=[0,N-1,0,N-1], cmap='plasma',
               origin='lower', aspect='equal')
clines = ax.contour(Vf_no, levels=np.arange(0,101,10), colors='white',
                     linewidths=0.5, alpha=0.6, extent=[0,N-1,0,N-1])
ax.clabel(clines, clines.levels, fontsize=7, fmt='%.0f V')
fig.colorbar(im, ax=ax, label='$V$ (V)', pad=0.02, shrink=0.85)
ax.set_xlabel('$x$ (grid)'); ax.set_ylabel('$y$ (grid)')
ax.set_title('Without Needle  (uniform field)')

# Right panel: with needle
ax = axes[1]
im = ax.imshow(Vf, extent=[0,N-1,0,N-1], cmap='plasma',
               origin='lower', aspect='equal')
clines = ax.contour(Vf, levels=np.arange(0,101,10), colors='white',
                     linewidths=0.5, alpha=0.6, extent=[0,N-1,0,N-1])
ax.clabel(clines, clines.levels, fontsize=7, fmt='%.0f V')
# Draw needle
ax.plot([centre, centre], [0, needle_top_disp],
        color=style.ACCENT, lw=3, solid_capstyle='butt')
ax.plot(centre, needle_top_disp, '^', color='lime', ms=9, zorder=6)
fig.colorbar(im, ax=ax, label='$V$ (V)', pad=0.02, shrink=0.85)
ax.set_xlabel('$x$ (grid)'); ax.set_ylabel('$y$ (grid)')
ax.set_title('With Needle  ($\\beta \\approx 7.5$)')

fig.suptitle('Potential Distribution: Effect of the Conducting Needle',
             fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig('figures/q3/fig10_with_vs_without.png')
plt.close()
print("  fig10 saved")

print("\nQ3 supplementary complete — 5 new figures saved.")
