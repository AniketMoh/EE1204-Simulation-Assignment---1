"""
q1_supplementary.py
===================
EE1204 — Question 1 Supplementary Analysis

Additional figures beyond the base simulation:
    fig7  — 3D surface plot of V(x,y)
    fig8  — |E| along horizontal line at plate height (fringing analysis)
    fig9  — Energy density heat map + computed capacitance correction
    fig10 — Bound surface charge σ_b at the dielectric interface

Requires: q1_parallel_plate_capacitor.py must have been run first (or
          we re-run the solver inline here for self-containedness).

Author : Aniket Mohapatra (EE25BTECH11007)
Date   : 4 April 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import os

mpl.use('Agg')
import style  # noqa: F401,E402

# ═══════════════════════════════════════════════════════════════
#  1.  RE-RUN THE SOLVER (self-contained)
# ═══════════════════════════════════════════════════════════════

N = 120; L = 5.0e-3; h = L/(N-1); ITERS = 5000
VA, VB = 5.0, -5.0
yA = 5.0/3.0*1e-3; yB = -5.0/3.0*1e-3
xlo, xhi = -2.5e-3, 2.5e-3

x = np.linspace(-L/2, L/2, N)
y = np.linspace(-L/2, L/2, N)
X, Y = np.meshgrid(x, y)

idx = lambda a, v: int(np.argmin(np.abs(a - v)))
jA, jB = idx(y, yA), idx(y, yB)
iL, iR = idx(x, xlo), idx(x, xhi)

# --- Uniform dielectric solver ---
V = np.zeros((N, N))
V[jA, iL:iR+1] = VA; V[jB, iL:iR+1] = VB
for k in range(ITERS):
    for j in range(1, N-1):
        V[j, 1:-1] = 0.25*(V[j,2:]+V[j,:-2]+V[j+1,1:-1]+V[j-1,1:-1])
    V[0,:]=0; V[-1,:]=0; V[:,0]=0; V[:,-1]=0
    V[jA, iL:iR+1]=VA; V[jB, iL:iR+1]=VB

Ey_u, Ex_u = np.gradient(V, h); Ex_u *= -1; Ey_u *= -1
Em_u = np.sqrt(Ex_u**2 + Ey_u**2)

# --- Dielectric interface solver ---
Vd = np.zeros((N, N))
Vd[jA, iL:iR+1]=VA; Vd[jB, iL:iR+1]=VB
k1, k2 = 4.0, 1.0; jI = idx(y, 0.0)
plate = np.zeros((N,N), dtype=bool)
plate[jA, iL:iR+1]=True; plate[jB, iL:iR+1]=True

for iteration in range(ITERS):
    for j in range(1, N-1):
        cols = np.arange(1, N-1)
        if plate[j].any():
            cols = cols[~plate[j, 1:-1]]
        if j == jI:
            Vd[j, cols] = (k1*Vd[j+1,cols]+k2*Vd[j-1,cols])/(k1+k2)
        else:
            Vd[j, cols] = 0.25*(Vd[j,cols+1]+Vd[j,cols-1]+Vd[j+1,cols]+Vd[j-1,cols])
    Vd[0,:]=0; Vd[-1,:]=0; Vd[:,0]=0; Vd[:,-1]=0
    Vd[jA,iL:iR+1]=VA; Vd[jB,iL:iR+1]=VB

Ey_d, Ex_d = np.gradient(Vd, h); Ex_d*=-1; Ey_d*=-1
Em_d = np.sqrt(Ex_d**2 + Ey_d**2)

mm = 1e3
os.makedirs('figures/q1', exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  Fig 7 : 3D surface plot of V(x,y)
# ═══════════════════════════════════════════════════════════════

fig = plt.figure(figsize=(9, 6.5))
ax = fig.add_subplot(111, projection='3d')

# Subsample for clarity
sk = 3
Xs, Ys, Vs = X[::sk,::sk]*mm, Y[::sk,::sk]*mm, V[::sk,::sk]
surf = ax.plot_surface(Xs, Ys, Vs, cmap=style.CMAP_POT,
                       edgecolor='none', alpha=0.92, antialiased=True)
fig.colorbar(surf, ax=ax, shrink=0.55, pad=0.08, label='$V$ (V)')

ax.set_xlabel('$x$ (mm)', labelpad=8)
ax.set_ylabel('$y$ (mm)', labelpad=8)
ax.set_zlabel('$V$ (V)', labelpad=8)
ax.set_title('3D Potential Landscape  ($\\kappa = 1$)', pad=12)
ax.view_init(elev=30, azim=-55)
fig.savefig('figures/q1/fig7_3d_potential.png')
plt.close()
print("  fig7 saved")

# ═══════════════════════════════════════════════════════════════
#  Fig 8 : |E| along horizontal line at plate height (fringing)
# ═══════════════════════════════════════════════════════════════
#
#  We plot |E| along y = yA (Plate A row) to show the field
#  spike at the plate edges (x = ±2.5 mm) — the fringing effect.

E_along_plateA = Em_u[jA, :]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(x*mm, E_along_plateA, color=style.PLATE_POS, lw=1.5,
        label='$|\\mathbf{E}|$ along $y = y_A$')
ax.axvline(xlo*mm, color=style.PLATE_NEG, ls='--', lw=0.9, alpha=0.7,
           label='Plate edge (left)')
ax.axvline(xhi*mm, color=style.PLATE_NEG, ls='--', lw=0.9, alpha=0.7,
           label='Plate edge (right)')
ax.axvspan(xlo*mm, xhi*mm, alpha=0.06, color=style.ACCENT,
           label='Plate extent')
d_gap = y[jA]-y[jB]
E0 = (VA-VB)/d_gap
ax.axhline(E0, color=style.ACCENT2, ls=':', lw=1.1,
           label=f'$E_0 \\approx {E0:.0f}$ V/m')
ax.set_xlabel('$x$ (mm)')
ax.set_ylabel('$|\\mathbf{E}|$ (V/m)')
ax.set_title('Field Magnitude Along Plate A Height — Fringing Analysis')
ax.legend(fontsize=8, loc='upper right'); ax.grid(True)
ax.set_xlim(-2.5, 2.5)
fig.savefig('figures/q1/fig8_fringing_analysis.png')
plt.close()
print("  fig8 saved")

# ═══════════════════════════════════════════════════════════════
#  Fig 9 : Energy density + capacitance correction factor
# ═══════════════════════════════════════════════════════════════
#
#  Energy density:   u = ½ ε₀ |E|²
#  Total energy:     U = ∫∫ u dA  =  Σ ½ ε₀ |E|² h²
#  Ideal capacitor:  U_ideal = ½ C V² = ½ (ε₀ w/d) (ΔV)²
#  Correction:       C_eff / C_ideal = U / U_ideal

eps0 = 8.854e-12
u_density = 0.5 * eps0 * Em_u**2            # energy density [J/m²]
U_total = np.sum(u_density) * h**2           # total energy [J/m] (per unit z-length)

# Ideal parallel-plate: C = ε₀ w / d  (per unit z-length)
plate_width = xhi - xlo                      # 5 mm
C_ideal = eps0 * plate_width / d_gap
U_ideal = 0.5 * C_ideal * (VA - VB)**2
C_ratio = U_total / U_ideal

print(f"\n  U_total  = {U_total:.4e} J/m")
print(f"  U_ideal  = {U_ideal:.4e} J/m")
print(f"  C_eff / C_ideal = {C_ratio:.3f}")

fig, ax = plt.subplots(figsize=(7.5, 6))
# Plot log of energy density for dynamic range
u_plot = np.log10(u_density + 1e-20)
im = ax.pcolormesh(X*mm, Y*mm, u_plot, cmap='hot', shading='auto')
fig.colorbar(im, ax=ax,
             label='$\\log_{10}(u)$  [J/m$^2$]', pad=0.02)

# Draw plates
ax.plot([xlo*mm,xhi*mm],[y[jA]*mm]*2, color='cyan', lw=3,
        solid_capstyle='butt', label=f'Plate A (+{VA:.0f} V)')
ax.plot([xlo*mm,xhi*mm],[y[jB]*mm]*2, color='lime', lw=3,
        solid_capstyle='butt', label=f'Plate B ({VB:.0f} V)')

ax.set_xlabel('$x$ (mm)'); ax.set_ylabel('$y$ (mm)')
ax.set_title(f'Electrostatic Energy Density  '
             f'($U = {U_total:.2e}$ J/m,  '
             f'$C_{{\\mathrm{{eff}}}}/C_{{\\mathrm{{ideal}}}} = {C_ratio:.3f}$)')
ax.legend(fontsize=8, loc='upper right'); ax.set_aspect('equal')
fig.savefig('figures/q1/fig9_energy_density.png')
plt.close()
print("  fig9 saved")

# ═══════════════════════════════════════════════════════════════
#  Fig 10 : Bound surface charge at the dielectric interface
# ═══════════════════════════════════════════════════════════════
#
#  At the interface y = 0 between κ₁ = 4 (above) and κ₂ = 1 (below):
#
#      σ_b = ε₀ (E_{1n} − E_{2n})  =  ε₀ (Ey_above − Ey_below)
#
#  where E_{n} is the upward normal component of E just above/below.

# Ey just above the interface (row jI-1) and just below (row jI+1)
Ey_above = -Ey_d[jI - 1, :]   # Ey_d already has the -∂V/∂y sign; we want upward
Ey_below = -Ey_d[jI + 1, :]

# Bound surface charge: σ_b = ε₀ (κ₁ E_above − κ₂ E_below) ... 
# More precisely: σ_b = P₁ₙ − P₂ₙ = ε₀(κ₁−1)E₁ₙ − ε₀(κ₂−1)E₂ₙ
# Using the correct Ey from the gradient (which is -∂V/∂y):
Ey_a = Ey_d[jI - 1, :]   # just above interface
Ey_b = Ey_d[jI + 1, :]   # just below interface

# σ_bound = ε₀[(κ₁-1)E_{y,above} - (κ₂-1)E_{y,below}]
# But simpler: σ_b = D₁ₙ - D₂ₙ = ε₀(κ₁ Ey_above - κ₂ Ey_below) ← this is σ_free+bound
# For bound charge only: σ_b = ε₀(Ey_above - Ey_below) × (appropriate factor)
# Cleanest: σ_b = (D_above - D_below)/1 ... but there's no free charge at interface
# so D is continuous => σ_free = 0, and σ_b = ε₀(E_below - E_above) (since n̂ = ŷ)

# Let's just compute the discontinuity in E_y across the interface
E_discont = Ey_d[jI+1, :] - Ey_d[jI-1, :]   # [V/m]
sigma_b = eps0 * E_discont * 1e6              # convert to µC/m² for readability

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(x*mm, sigma_b, color=style.ACCENT, lw=1.5)
ax.fill_between(x*mm, sigma_b, 0, where=sigma_b > 0,
                color='#67a9cf', alpha=0.35, label='Positive $\\sigma_b$')
ax.fill_between(x*mm, sigma_b, 0, where=sigma_b <= 0,
                color='#ef8a62', alpha=0.35, label='Negative $\\sigma_b$')
ax.axvline(xlo*mm, color=style.NEUTRAL, ls=':', lw=0.7)
ax.axvline(xhi*mm, color=style.NEUTRAL, ls=':', lw=0.7)
ax.set_xlabel('$x$ (mm)')
ax.set_ylabel('$\\sigma_b$ ($\\mu$C/m$^2$)')
ax.set_title('Bound Surface Charge at Dielectric Interface  '
             '($\\kappa_1=4$ / $\\kappa_2=1$,  $y=0$)')
ax.legend(fontsize=9); ax.grid(True)
fig.savefig('figures/q1/fig10_bound_surface_charge.png')
plt.close()
print("  fig10 saved")

print("\nQ1 supplementary complete — 4 new figures saved.")
