"""
q2_iterative_poisson.py
=======================
EE1204 Simulation Assignment 1 — Question 2 (Supplement)
Iterative Poisson Solver for Cross-Validation

Solves Poisson's equation  ∇²V = −ρ/ε  on a 201×201 grid using
Gauss–Seidel relaxation, where the point charge is modelled as a
Gaussian source blob.  Results are compared with the analytical
method-of-images solution to validate both approaches.

The modified update rule (vs. Laplace) is:
    V_{i,j} = ¼ ( V_{i+1,j} + V_{i-1,j} + V_{i,j+1} + V_{i,j-1} + h²ρ_{i,j} )

Figures produced (saved to ./figures/q2/):
    fig8  — Equipotential contours (iterative)
    fig9  — Electric field vectors (iterative)
    fig10 — σ(θ) comparison: iterative vs method of images
    fig11 — Convergence history of the iterative solver

Author : Aniket (EE25BTECH11007)
Course : EE1204 — Engineering Electromagnetics, IIT Hyderabad
Date   : March 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os

mpl.use('Agg')

from scipy.interpolate import RegularGridInterpolator
import style  # noqa: F401,E402

# ═══════════════════════════════════════════════════════════════
#  1.  PARAMETERS
# ═══════════════════════════════════════════════════════════════

N = 201             # grid resolution  →  201 × 201
L = 6.0             # half-extent of domain  →  [−6, 6]²
h = 2 * L / (N - 1) # grid spacing

R = 2.0             # sphere radius
d = 4.0             # charge x-position
Q = 10e-6           # charge magnitude (for labelling only)

x1d = np.linspace(-L, L, N)
y1d = np.linspace(-L, L, N)
X, Y = np.meshgrid(x1d, y1d)

# Boolean mask for the sphere interior
sphere_mask = X**2 + Y**2 <= R**2

os.makedirs('figures/q2', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  2.  SOURCE TERM  (Gaussian-smoothed point charge)
# ═══════════════════════════════════════════════════════════════
#
#  A true point charge has ρ = Q δ(x−d) δ(y), which cannot be
#  represented on a discrete grid.  We approximate it as a
#  Gaussian blob with width σ_g ≈ 3h (spread over ~3 grid cells):
#
#      ρ(x,y) = [1 / (2π σ_g²)] exp(−[(x−d)² + y²] / [2σ_g²])
#
#  This integrates to 1 over the plane, giving unit total charge
#  in normalised units.

sigma_g = 3 * h
rho = np.exp(-((X - d)**2 + Y**2) / (2 * sigma_g**2))
rho /= (2 * np.pi * sigma_g**2)     # normalise to unit total charge

rho_h2 = h**2 * rho                  # pre-multiply h²ρ for efficiency

# ═══════════════════════════════════════════════════════════════
#  3.  GAUSS–SEIDEL POISSON SOLVER
# ═══════════════════════════════════════════════════════════════
#
#  Update rule (Eq. 8 in report):
#
#      V_{i,j} = ¼ (V_{i+1,j} + V_{i−1,j} + V_{i,j+1} + V_{i,j−1} + h²ρ_{i,j})
#
#  Boundary conditions:
#      • V = 0 on all four domain edges
#      • V = 0 on sphere interior (grounded conductor)

V = np.zeros((N, N))    # initial guess
ITERS = 12000           # maximum iterations
residuals = []          # convergence tracking

print("Running iterative Poisson solver (201×201)...")
for k in range(1, ITERS + 1):
    V_prev = V.copy()

    # Row-wise vectorised Gauss–Seidel sweep
    for j in range(1, N - 1):
        V[j, 1:-1] = 0.25 * (
            V[j, 2:]       +          # right neighbour (old)
            V[j, :-2]      +          # left  neighbour (updated)
            V[j + 1, 1:-1] +          # upper neighbour (old)
            V[j - 1, 1:-1] +          # lower neighbour (updated)
            rho_h2[j, 1:-1]           # source term
        )

    # Re-apply Dirichlet BCs
    V[0, :] = 0; V[-1, :] = 0; V[:, 0] = 0; V[:, -1] = 0
    V[sphere_mask] = 0.0

    # Convergence check every 500 iterations
    if k % 500 == 0:
        max_delta = np.max(np.abs(V - V_prev))
        residuals.append((k, max_delta))
        print(f"  iter {k:5d}   max|ΔV| = {max_delta:.3e}")
        if max_delta < 1e-6:
            print(f"  >>> Converged at iteration {k}")
            break

# ═══════════════════════════════════════════════════════════════
#  4.  ELECTRIC FIELD COMPUTATION
# ═══════════════════════════════════════════════════════════════

Ey, Ex = np.gradient(V, h)
Ex *= -1; Ey *= -1
Em = np.hypot(Ex, Ey)

# Zero field inside the conductor
Ex[sphere_mask] = 0
Ey[sphere_mask] = 0
Em[sphere_mask] = 0

# ═══════════════════════════════════════════════════════════════
#  5.  SURFACE CHARGE DENSITY  (iterative result)
# ═══════════════════════════════════════════════════════════════

theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
dr = h

interp = RegularGridInterpolator(
    (y1d, x1d), V, method='linear',
    bounds_error=False, fill_value=0.0
)
# Sample points just outside/inside the sphere  (interpolator expects [y, x])
pts_outside = np.column_stack([(R + dr) * np.sin(theta),
                                (R + dr) * np.cos(theta)])
pts_inside  = np.column_stack([(R - dr) * np.sin(theta),
                                (R - dr) * np.cos(theta)])
sigma_iter = -(interp(pts_outside) - interp(pts_inside)) / (2 * dr)

# ═══════════════════════════════════════════════════════════════
#  6.  METHOD OF IMAGES  (for comparison)
# ═══════════════════════════════════════════════════════════════

# Image charge: Q' = −Q R/d,  d' = R²/d   (using unit charge for shape comparison)
qi = -1.0 * R / d
di = R**2 / d

r_real = np.maximum(np.hypot(X - d, Y), 1e-15)
r_img  = np.maximum(np.hypot(X - di, Y), 1e-15)
V_moi = -1.0 * np.log(r_real) - qi * np.log(r_img)
V_moi[sphere_mask] = 0.0

interp_moi = RegularGridInterpolator(
    (y1d, x1d), V_moi, method='linear',
    bounds_error=False, fill_value=0.0
)
sigma_moi = -(interp_moi(pts_outside) - interp_moi(pts_inside)) / (2 * dr)

# ═══════════════════════════════════════════════════════════════
#  7.  PLOTTING
# ═══════════════════════════════════════════════════════════════

# Sphere circle for drawing
tc = np.linspace(0, 2 * np.pi, 300)
cx, cy = R * np.cos(tc), R * np.sin(tc)

# ── Fig 8 : Iterative equipotential ──
fig, ax = plt.subplots(figsize=(7.5, 6.5))
V_display = V.copy(); V_display[sphere_mask] = np.nan
cf = ax.contourf(X, Y, V_display, levels=25, cmap=style.CMAP_POT, extend='both')
ax.contour(X, Y, V_display, levels=25, colors='k', linewidths=0.2, alpha=0.3)
ax.fill(cx, cy, color='#d9d9d9', ec='k', lw=1.2, zorder=5)
ax.plot(d, 0, '*', color=style.PLATE_NEG, ms=14, zorder=6,
        label=f'$Q = {Q*1e6:.0f}\\;\\mu$C')
fig.colorbar(cf, ax=ax, label='Potential (Poisson iteration)',
             shrink=0.82, pad=0.02)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title('Equipotential Contours — Iterative Poisson Solver')
ax.legend(fontsize=10, loc='upper left'); ax.set_aspect('equal')
ax.set_xlim(-L, L); ax.set_ylim(-L, L)
fig.savefig('figures/q2/fig8_iterative_equipotential.png')
plt.close()

# ── Fig 9 : Iterative field vectors ──
fig, ax = plt.subplots(figsize=(7.5, 6.5))
sk = 6
Xs, Ys = X[::sk, ::sk], Y[::sk, ::sk]
Exs, Eys = Ex[::sk, ::sk], Ey[::sk, ::sk]
Es = np.hypot(Exs, Eys)
Es_safe = np.maximum(Es, 1e-20)
q = ax.quiver(Xs, Ys, Exs / Es_safe, Eys / Es_safe,
              np.log10(Es_safe), cmap='YlOrRd',
              scale=28, width=0.004, alpha=0.85)
fig.colorbar(q, ax=ax, label=r'$\log_{10}|\mathbf{E}|$',
             shrink=0.82, pad=0.02)
ax.fill(cx, cy, color='#d9d9d9', ec='k', lw=1.2, zorder=5)
ax.plot(d, 0, '*', color=style.PLATE_NEG, ms=14, zorder=6)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title('Electric Field Vectors — Iterative Poisson Solver')
ax.set_aspect('equal'); ax.set_xlim(-L, L); ax.set_ylim(-L, L)
fig.savefig('figures/q2/fig9_iterative_field.png')
plt.close()

# ── Fig 10 : σ(θ) comparison (shape-normalised) ──
# Normalise both to their peak for shape comparison (different absolute scales)
s_it  = sigma_iter / np.max(np.abs(sigma_iter))
s_moi = sigma_moi  / np.max(np.abs(sigma_moi))

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(np.degrees(theta), s_it, color=style.PLATE_POS, lw=1.5,
        label='Iterative Poisson (201×201)')
ax.plot(np.degrees(theta), s_moi, color=style.PLATE_NEG, ls='--', lw=1.5,
        label='Method of Images (analytical)')
ax.set_xlabel(r'$\theta$ (degrees)')
ax.set_ylabel(r'$\sigma / \sigma_{\max}$')
ax.set_title(r'Surface Charge Density: Iterative vs.\ Method of Images')
ax.legend(fontsize=9); ax.grid(True)
fig.savefig('figures/q2/fig10_sigma_comparison.png')
plt.close()

# ── Fig 11 : Convergence history ──
if residuals:
    iters_r, vals_r = zip(*residuals)
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.semilogy(iters_r, vals_r, 'o-', color=style.PLATE_POS, ms=4, lw=1.2)
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r'Max residual $|\Delta V|$')
    ax.set_title('Convergence of Iterative Poisson Solver (Q2)')
    ax.grid(True)
    fig.savefig('figures/q2/fig11_iterative_convergence.png')
    plt.close()

print("Q2 (Iterative Poisson) complete — 4 figures saved to figures/q2/")
