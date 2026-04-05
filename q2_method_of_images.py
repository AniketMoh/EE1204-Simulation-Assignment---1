"""
q2_method_of_images.py
======================
EE1204 Simulation Assignment 1 — Question 2
Point Charge Near a Grounded Conducting Sphere (2D)

Uses the **method of images** with the 2D logarithmic potential on a
300×300 grid.  The grounded sphere (radius R = 2) is centred at the origin
and the point charge Q = 10 µC is placed at (d, 0) = (4, 0).

Image-charge formulas (for a grounded cylinder of radius R):
    Image position :  d' = R² / d
    Image charge   :  Q' = −Q R / d

Figures produced (saved to ./figures/q2/):
    fig1 — Equipotential contours (base case)
    fig2 — Electric field vectors (quiver, log-coloured)
    fig3 — Induced surface charge density σ(θ) (base case)
    fig4 — σ(θ) for varying charge magnitudes Q
    fig5 — σ(θ) for varying distances d
    fig6 — Equipotential panels for 4 distances
    fig7 — Electric field streamlines (base case)

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

R  = 2.0            # sphere (cylinder) radius  [simulation units]
Q0 = 10e-6          # base-case charge magnitude  = 10 µC
d0 = 4.0            # base-case charge position  x = 4

# Simulation domain
XMIN, XMAX = -6.0, 6.0
YMIN, YMAX = -6.0, 6.0
NG = 300             # grid resolution  →  300 × 300
delta = (XMAX - XMIN) / (NG - 1)   # grid spacing ≈ 0.04

# Grid arrays
x = np.linspace(XMIN, XMAX, NG)
y = np.linspace(YMIN, YMAX, NG)
X, Y = np.meshgrid(x, y)

os.makedirs('figures/q2', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  2.  METHOD OF IMAGES — CORE FUNCTIONS
# ═══════════════════════════════════════════════════════════════


def image_params(Q, d):
    """
    Compute the image charge and its position for a line charge Q
    at distance d from the axis of a grounded cylinder of radius R.

    Returns
    -------
    Q_img : float   — image charge magnitude  Q' = −Q R / d
    d_img : float   — image position          d' = R² / d
    """
    Q_img = -Q * R / d
    d_img = R**2 / d
    return Q_img, d_img


def potential(X, Y, Q, d):
    """
    2D logarithmic potential at every grid point (X, Y) due to a real
    charge Q at (d, 0) and its image inside the sphere.

    V = −Q ln(r_q) − Q' ln(r_q')

    The potential inside the sphere is set to zero (conductor interior).
    """
    Q_img, d_img = image_params(Q, d)

    # Distance from each grid point to the real charge
    r_real = np.maximum(np.hypot(X - d, Y), 1e-15)
    # Distance from each grid point to the image charge
    r_img  = np.maximum(np.hypot(X - d_img, Y), 1e-15)

    V = -Q * np.log(r_real) - Q_img * np.log(r_img)

    # Enforce V = 0 inside the conductor
    V[X**2 + Y**2 <= R**2] = 0.0
    return V


def electric_field(X, Y, Q, d):
    """
    Analytical electric field at every grid point from the real charge
    at (d, 0) and its image.

    For a line charge at (x₀, y₀), the field contribution is:
        E_x = Q (x − x₀) / r²
        E_y = Q (y − y₀) / r²
    """
    Q_img, d_img = image_params(Q, d)

    # Real charge contribution
    dx1, dy1 = X - d, Y
    r1sq = np.maximum(dx1**2 + dy1**2, 1e-15)

    # Image charge contribution
    dx2, dy2 = X - d_img, Y
    r2sq = np.maximum(dx2**2 + dy2**2, 1e-15)

    Ex = Q * dx1 / r1sq + Q_img * dx2 / r2sq
    Ey = Q * dy1 / r1sq + Q_img * dy2 / r2sq

    # Zero field inside the conductor
    mask = X**2 + Y**2 <= R**2
    Ex[mask] = 0.0
    Ey[mask] = 0.0
    return Ex, Ey


def surface_charge(Q, d, V_grid, n_pts=360):
    """
    Numerical surface-charge density on the sphere, computed from
    the radial derivative of V at r = R:

        σ(θ) = −(V_outside − V_inside) / (2 Δr)

    Uses bilinear interpolation on the potential grid.

    Parameters
    ----------
    Q, d     : charge parameters (used only to label; V_grid already computed)
    V_grid   : 2D potential array  (N × N)
    n_pts    : number of angular sample points

    Returns
    -------
    theta : array of angles  [0, 2π)
    sigma : surface-charge density at each angle
    """
    interp = RegularGridInterpolator(
        (y, x), V_grid, method='linear',
        bounds_error=False, fill_value=0.0
    )
    dr = delta   # radial step = one grid spacing
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)

    # Sample points just outside and just inside the sphere
    pts_outside = np.column_stack([
        (R + dr) * np.sin(theta),   # y-coordinate
        (R + dr) * np.cos(theta)    # x-coordinate
    ])
    pts_inside = np.column_stack([
        (R - dr) * np.sin(theta),
        (R - dr) * np.cos(theta)
    ])

    # σ ≈ −∂V/∂r  via central difference
    sigma = -(interp(pts_outside) - interp(pts_inside)) / (2 * dr)
    return theta, sigma


# ═══════════════════════════════════════════════════════════════
#  3.  BASE-CASE COMPUTATION
# ═══════════════════════════════════════════════════════════════

V0 = potential(X, Y, Q0, d0)
Ex0, Ey0 = electric_field(X, Y, Q0, d0)
Em0 = np.hypot(Ex0, Ey0)

# Circle for drawing the sphere boundary
tc = np.linspace(0, 2 * np.pi, 300)
cx, cy = R * np.cos(tc), R * np.sin(tc)


def draw_sphere(ax):
    """Fill the sphere as a grey disc with a black outline."""
    ax.fill(cx, cy, color='#d9d9d9', ec='k', lw=1.2, zorder=5)


levs = np.linspace(-8, 8, 33)   # contour levels

# ═══════════════════════════════════════════════════════════════
#  4.  FIGURE GENERATION
# ═══════════════════════════════════════════════════════════════

# ── Fig 1 : Equipotential contours (base case) ──
fig, ax = plt.subplots(figsize=(7.5, 6.5))
cf = ax.contourf(X, Y, np.clip(V0, -8, 8), levels=levs,
                 cmap=style.CMAP_POT, extend='both')
ax.contour(X, Y, np.clip(V0, -8, 8), levels=levs,
           colors='k', linewidths=0.2, alpha=0.3)
draw_sphere(ax)
ax.plot(d0, 0, '*', color=style.PLATE_NEG, ms=14, zorder=6,
        label=f'$Q = {Q0*1e6:.0f}\\;\\mu$C')
fig.colorbar(cf, ax=ax, label='Potential  $V$  (normalised)',
             shrink=0.82, pad=0.02)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title(f'Equipotential Contours  '
             f'($Q = {Q0*1e6:.0f}\\;\\mu$C  at  $d = {d0:.0f}$,  $R = {R:.0f}$)')
ax.legend(fontsize=10, loc='upper left', framealpha=0.9)
ax.set_aspect('equal'); ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
fig.savefig('figures/q2/fig1_equipotential_base.png')
plt.close()

# ── Fig 2 : Field vectors (quiver, log-coloured) ──
fig, ax = plt.subplots(figsize=(7.5, 6.5))
sk = 8   # skip factor for quiver arrows
Xs, Ys = X[::sk, ::sk], Y[::sk, ::sk]
Exs, Eys = Ex0[::sk, ::sk], Ey0[::sk, ::sk]
Es = np.hypot(Exs, Eys)
Es_safe = np.maximum(Es, 1e-20)
q = ax.quiver(Xs, Ys, Exs / Es_safe, Eys / Es_safe,
              np.log10(Es_safe), cmap='YlOrRd',
              scale=28, width=0.004, alpha=0.85)
fig.colorbar(q, ax=ax, label=r'$\log_{10}|\mathbf{E}|$',
             shrink=0.82, pad=0.02)
draw_sphere(ax)
ax.plot(d0, 0, '*', color=style.PLATE_NEG, ms=14, zorder=6,
        label=f'$Q = {Q0*1e6:.0f}\\;\\mu$C')
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title('Electric Field Distribution  (normalised arrows, log-coloured)')
ax.legend(fontsize=10); ax.set_aspect('equal')
ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
fig.savefig('figures/q2/fig2_field_vectors.png')
plt.close()

# ── Fig 3 : Surface charge density σ(θ) (base case) ──
th_s, sig_s = surface_charge(Q0, d0, V0)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.fill_between(np.degrees(th_s), sig_s, 0, where=sig_s < 0,
                color='#ef8a62', alpha=0.45, label=r'Negative $\sigma$')
ax.fill_between(np.degrees(th_s), sig_s, 0, where=sig_s >= 0,
                color='#67a9cf', alpha=0.45, label=r'Positive $\sigma$')
ax.plot(np.degrees(th_s), sig_s, 'k-', lw=0.7)
ax.axhline(0, color=style.NEUTRAL, lw=0.4)
ax.set_xlabel(r'$\theta$  (degrees from $+x$ axis)')
ax.set_ylabel(r'$\sigma$  (normalised)')
ax.set_title(f'Induced Surface Charge Density  '
             f'($Q = {Q0*1e6:.0f}\\;\\mu$C,  $d = {d0:.0f}$)')
ax.legend(fontsize=9); ax.grid(True)
fig.savefig('figures/q2/fig3_surface_charge_base.png')
plt.close()

# ── Fig 4 : Varying Q ──
Qs = [1e-6, 5e-6, 10e-6, 20e-6]
colours_q = ['#4575b4', '#74add1', '#f46d43', '#d73027']

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for Qv, c in zip(Qs, colours_q):
    Vt = potential(X, Y, Qv, d0)
    th, sig = surface_charge(Qv, d0, Vt)
    ax.plot(np.degrees(th), sig, color=c, lw=1.3,
            label=f'$Q = {Qv*1e6:.0f}\\;\\mu$C')
ax.set_xlabel(r'$\theta$  (degrees)')
ax.set_ylabel(r'$\sigma$  (normalised)')
ax.set_title(f'Surface Charge Density — Varying Charge Magnitude  '
             f'($d = {d0:.0f}$)')
ax.legend(fontsize=9); ax.grid(True)
fig.savefig('figures/q2/fig4_sigma_vary_Q.png')
plt.close()

# ── Fig 5 : Varying d ──
ds = [2.5, 3.0, 4.0, 5.5]
colours_d = ['#762a83', '#af8dc3', '#e7d4e8', '#d9f0d3']

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for dv, c in zip(ds, colours_d):
    Vt = potential(X, Y, Q0, dv)
    th, sig = surface_charge(Q0, dv, Vt)
    ax.plot(np.degrees(th), sig, color=c, lw=1.3,
            label=f'$d = {dv:.1f}$')
ax.set_xlabel(r'$\theta$  (degrees)')
ax.set_ylabel(r'$\sigma$  (normalised)')
ax.set_title(f'Surface Charge Density — Varying Distance  '
             f'($Q = {Q0*1e6:.0f}\\;\\mu$C)')
ax.legend(fontsize=9); ax.grid(True)
fig.savefig('figures/q2/fig5_sigma_vary_d.png')
plt.close()

# ── Fig 6 : Equipotential panels for 4 distances ──
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, dv in zip(axes.flat, ds):
    Vv = np.clip(potential(X, Y, Q0, dv), -8, 8)
    cf = ax.contourf(X, Y, Vv, levels=levs, cmap=style.CMAP_POT,
                     extend='both')
    ax.contour(X, Y, Vv, levels=levs, colors='k',
               linewidths=0.15, alpha=0.25)
    draw_sphere(ax)
    ax.plot(dv, 0, '*', color=style.PLATE_NEG, ms=11, zorder=6)
    ax.set_title(f'$d = {dv:.1f}$', fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
    fig.colorbar(cf, ax=ax, shrink=0.78, pad=0.02)
fig.suptitle(f'Equipotential Maps for Varying Charge Distance  '
             f'($Q = {Q0*1e6:.0f}\\;\\mu$C)', fontsize=14, y=1.01)
plt.tight_layout()
fig.savefig('figures/q2/fig6_equipotential_vary_d.png')
plt.close()

# ── Fig 7 : Streamlines (base case) ──
fig, ax = plt.subplots(figsize=(7.5, 6.5))
cf = ax.contourf(X, Y, np.clip(V0, -8, 8), levels=levs,
                 cmap=style.CMAP_POT, alpha=0.30, extend='both')
lnorm = mpl.colors.LogNorm(vmin=1e-8,
                            vmax=Em0[np.isfinite(Em0)].max())
ax.streamplot(X, Y, Ex0, Ey0, color=Em0, cmap=style.CMAP_FIELD,
              norm=lnorm, density=2.2, linewidth=0.65, arrowsize=0.75)
draw_sphere(ax)
ax.plot(d0, 0, '*', color=style.PLATE_NEG, ms=14, zorder=6)
fig.colorbar(cf, ax=ax, label='$V$ (normalised)', shrink=0.82, pad=0.02)
ax.set_xlabel('$x$'); ax.set_ylabel('$y$')
ax.set_title('Electric Field Streamlines  (base case)')
ax.set_aspect('equal')
ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
fig.savefig('figures/q2/fig7_streamlines.png')
plt.close()

print("Q2 (Method of Images) complete — 7 figures saved to figures/q2/")
