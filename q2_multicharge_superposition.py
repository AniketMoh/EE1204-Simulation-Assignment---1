"""
q2_multicharge_superposition.py
===============================
EE1204 Simulation Assignment 1 — Question 2 (Extension)
Multiple Charges Around a Grounded Sphere — Superposition

Demonstrates the principle of superposition by placing 6 external
charges (3 positive, 3 negative) around the grounded sphere.  For
each real charge, the image charge is computed independently using:

    d'_k = R² / d_k ,     Q'_k = −Q_k R / d_k

The total potential and field are then:

    V_total = Σ V_k  (scalar sum)
    E_total = Σ E_k  (vector sum)

Figures produced (saved to ./figures/q2/):
    fig12 — Individual equipotential panels (2×3)
    fig13 — Combined (superposed) equipotential map
    fig14 — Combined electric field vectors
    fig15 — Combined surface charge density σ(θ)

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

R = 2.0              # sphere radius
GRID_HALF = 8.0      # domain half-extent  →  [−8, 8]²
N = 500              # grid resolution (finer than base case)

x1d = np.linspace(-GRID_HALF, GRID_HALF, N)
y1d = np.linspace(-GRID_HALF, GRID_HALF, N)
X, Y = np.meshgrid(x1d, y1d)

sphere_mask = X**2 + Y**2 <= R**2
delta = (2 * GRID_HALF) / (N - 1)     # grid spacing

# Sphere circle for drawing
tc = np.linspace(0, 2 * np.pi, 300)
cx, cy = R * np.cos(tc), R * np.sin(tc)

os.makedirs('figures/q2', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  2.  DEFINE THE 6 EXTERNAL CHARGES
# ═══════════════════════════════════════════════════════════════
#  Format: (x_position, y_position, charge_in_Coulombs)

charges = [
    ( 4.0,  1.5,  +10e-6),   # +10 µC
    (-3.5,  3.0,   +7e-6),   # + 7 µC
    ( 1.5, -4.5,   +5e-6),   # + 5 µC
    (-4.5, -2.0,   -8e-6),   # − 8 µC
    ( 3.0,  4.5,   -6e-6),   # − 6 µC
    (-2.0,  4.0,   -4e-6),   # − 4 µC
]

# ═══════════════════════════════════════════════════════════════
#  3.  IMAGE CHARGE COMPUTATION
# ═══════════════════════════════════════════════════════════════


def image_charge(cx, cy, Q, R):
    """
    Compute the image charge for a line charge Q at (cx, cy) outside
    a grounded cylinder of radius R centred at the origin.

    Returns
    -------
    ix, iy : image position  =  (cx, cy) × R² / d²
    Qi     : image charge    =  −Q R / d
    """
    d_sq = cx**2 + cy**2          # squared distance from origin
    d = np.sqrt(d_sq)
    scale = R**2 / d_sq           # inversion factor
    ix = cx * scale               # image x-position
    iy = cy * scale               # image y-position
    Qi = -Q * R / d               # image charge
    return ix, iy, Qi


# ═══════════════════════════════════════════════════════════════
#  4.  COMPUTE TOTAL POTENTIAL AND FIELD BY SUPERPOSITION
# ═══════════════════════════════════════════════════════════════

V_total  = np.zeros_like(X)      # total potential (scalar sum)
Ex_total = np.zeros_like(X)      # total E_x (vector sum)
Ey_total = np.zeros_like(X)      # total E_y (vector sum)

for (qx, qy, Q) in charges:
    ix, iy, Qi = image_charge(qx, qy, Q, R)

    # --- Potential contribution ---
    r_real = np.maximum(np.hypot(X - qx, Y - qy), 1e-15)
    r_img  = np.maximum(np.hypot(X - ix, Y - iy), 1e-15)
    V_total += -Q * np.log(r_real) - Qi * np.log(r_img)

    # --- Field contribution ---
    dx1, dy1 = X - qx, Y - qy
    r1sq = np.maximum(dx1**2 + dy1**2, 1e-15)
    dx2, dy2 = X - ix, Y - iy
    r2sq = np.maximum(dx2**2 + dy2**2, 1e-15)

    Ex_total += Q * dx1 / r1sq + Qi * dx2 / r2sq
    Ey_total += Q * dy1 / r1sq + Qi * dy2 / r2sq

# Enforce zero inside conductor
V_total[sphere_mask]  = 0.0
Ex_total[sphere_mask] = 0.0
Ey_total[sphere_mask] = 0.0
Em_total = np.hypot(Ex_total, Ey_total)

# ═══════════════════════════════════════════════════════════════
#  5.  SURFACE CHARGE DENSITY  (combined)
# ═══════════════════════════════════════════════════════════════

theta = np.linspace(0, 2 * np.pi, 500, endpoint=False)
dr = delta

interp = RegularGridInterpolator(
    (y1d, x1d), V_total, method='linear',
    bounds_error=False, fill_value=0.0
)
pts_outside = np.column_stack([(R + dr) * np.sin(theta),
                                (R + dr) * np.cos(theta)])
pts_inside  = np.column_stack([(R - dr) * np.sin(theta),
                                (R - dr) * np.cos(theta)])
sigma_total = -(interp(pts_outside) - interp(pts_inside)) / (2 * dr)

# ═══════════════════════════════════════════════════════════════
#  6.  PLOTTING
# ═══════════════════════════════════════════════════════════════

levs = np.linspace(-8, 8, 33)

# ── Fig 12 : Individual charge panels (2×3) ──
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
for idx_c, ((qx, qy, Q), ax) in enumerate(zip(charges, axes.flat)):
    ix, iy, Qi = image_charge(qx, qy, Q, R)
    r_real = np.maximum(np.hypot(X - qx, Y - qy), 1e-15)
    r_img  = np.maximum(np.hypot(X - ix, Y - iy), 1e-15)
    Vi = -Q * np.log(r_real) - Qi * np.log(r_img)
    Vi[sphere_mask] = 0.0

    cf = ax.contourf(X, Y, np.clip(Vi, -8, 8), levels=levs,
                     cmap=style.CMAP_POT, extend='both')
    ax.fill(cx, cy, color='#d9d9d9', ec='k', lw=1.0, zorder=5)

    # Mark real charge (star for +, triangle for −)
    marker = '*' if Q > 0 else 'v'
    color = style.PLATE_NEG if Q > 0 else style.PLATE_POS
    ax.plot(qx, qy, marker, color=color, ms=12, zorder=6)
    ax.plot(ix, iy, 'x', color='gray', ms=8, zorder=6)  # image

    sign = '+' if Q > 0 else ''
    ax.set_title(f'$Q_{{{idx_c+1}}} = {sign}{Q*1e6:.0f}\\;\\mu$C  '
                 f'at ({qx},{qy})', fontsize=10)
    ax.set_aspect('equal')
    ax.set_xlim(-GRID_HALF, GRID_HALF)
    ax.set_ylim(-GRID_HALF, GRID_HALF)
    fig.colorbar(cf, ax=ax, shrink=0.75, pad=0.02)

fig.suptitle('Equipotential Maps for Six Individual External Charges',
             fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig('figures/q2/fig12_multi_individual.png')
plt.close()

# ── Fig 13 : Combined equipotential ──
fig, ax = plt.subplots(figsize=(8, 7))
Vclip = np.clip(V_total, -15, 15)
Vclip[sphere_mask] = np.nan
levs_comb = np.linspace(-15, 15, 41)
cf = ax.contourf(X, Y, Vclip, levels=levs_comb, cmap=style.CMAP_POT,
                 extend='both')
ax.contour(X, Y, Vclip, levels=levs_comb, colors='k',
           linewidths=0.15, alpha=0.25)
ax.fill(cx, cy, color='#d9d9d9', ec='k', lw=1.2, zorder=5)

for idx_c, (qx, qy, Q) in enumerate(charges):
    marker = '*' if Q > 0 else 'v'
    color = style.PLATE_NEG if Q > 0 else style.PLATE_POS
    sign = '+' if Q > 0 else ''
    ax.plot(qx, qy, marker, color=color, ms=12, zorder=6,
            label=f'$Q_{{{idx_c+1}}}={sign}{Q*1e6:.0f}\\;\\mu$C')

fig.colorbar(cf, ax=ax, label='Potential (normalised)',
             shrink=0.82, pad=0.02)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title('Superposed Equipotential Map — 6 External Charges')
ax.legend(fontsize=7, loc='lower left', ncol=2, framealpha=0.9)
ax.set_aspect('equal')
ax.set_xlim(-GRID_HALF, GRID_HALF)
ax.set_ylim(-GRID_HALF, GRID_HALF)
fig.savefig('figures/q2/fig13_multi_combined.png')
plt.close()

# ── Fig 14 : Combined field vectors ──
fig, ax = plt.subplots(figsize=(8, 7))
sk = 12
Xs, Ys = X[::sk, ::sk], Y[::sk, ::sk]
Exs, Eys = Ex_total[::sk, ::sk], Ey_total[::sk, ::sk]
Es = np.hypot(Exs, Eys)
Es_safe = np.maximum(Es, 1e-20)
q = ax.quiver(Xs, Ys, Exs / Es_safe, Eys / Es_safe,
              np.log10(Es_safe), cmap='YlOrRd',
              scale=28, width=0.004, alpha=0.85)
fig.colorbar(q, ax=ax, label=r'$\log_{10}|\mathbf{E}|$',
             shrink=0.82, pad=0.02)
ax.fill(cx, cy, color='#d9d9d9', ec='k', lw=1.2, zorder=5)
for qx, qy, Q in charges:
    marker = '*' if Q > 0 else 'v'
    color = style.PLATE_NEG if Q > 0 else style.PLATE_POS
    ax.plot(qx, qy, marker, color=color, ms=12, zorder=6)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title('Superposed Electric Field — 6 External Charges')
ax.set_aspect('equal')
ax.set_xlim(-GRID_HALF, GRID_HALF)
ax.set_ylim(-GRID_HALF, GRID_HALF)
fig.savefig('figures/q2/fig14_multi_field.png')
plt.close()

# ── Fig 15 : Combined surface charge density ──
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.fill_between(np.degrees(theta), sigma_total, 0,
                where=sigma_total < 0,
                color='#ef8a62', alpha=0.45, label=r'Negative $\sigma$')
ax.fill_between(np.degrees(theta), sigma_total, 0,
                where=sigma_total >= 0,
                color='#67a9cf', alpha=0.45, label=r'Positive $\sigma$')
ax.plot(np.degrees(theta), sigma_total, 'k-', lw=0.7)
ax.axhline(0, color=style.NEUTRAL, lw=0.4)
ax.set_xlabel(r'$\theta$  (degrees from $+x$ axis)')
ax.set_ylabel(r'$\sigma$  (normalised)')
ax.set_title('Induced Surface Charge Density — 6 Superposed External Charges')
ax.legend(fontsize=9); ax.grid(True)
fig.savefig('figures/q2/fig15_multi_sigma.png')
plt.close()

print("Q2 (Multi-Charge Superposition) complete — 4 figures saved to figures/q2/")
