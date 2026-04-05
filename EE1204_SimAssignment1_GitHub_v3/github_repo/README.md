# EE1204 — Engineering Electromagnetics: Simulation Assignment 1

**Author:** Aniket Mohapatra (EE25BTECH11007)  
**Institute:** Indian Institute of Technology Hyderabad  
**Course:** EE1204 — Engineering Electromagnetics  
**Date:** 4 April 2026

---

## Overview

Numerical solutions to three two-dimensional electrostatic boundary-value problems:

1. **Parallel-Plate Capacitor** — Gauss–Seidel relaxation on a 120×120 grid with optional dielectric interface
2. **Point Charge Near a Grounded Sphere** — Method of images + iterative Poisson solver + multi-charge superposition
3. **Lightning Rod Field Enhancement** — Gauss–Seidel relaxation on a 100×100 grid with convergence analysis

## Repository Structure

```
.
├── style.py                          # Shared plotting style (colours, fonts, cmaps)
├── q1_parallel_plate_capacitor.py    # Question 1: Parallel-plate capacitor
├── q2_method_of_images.py            # Question 2: Method of images (7 figures)
├── q2_iterative_poisson.py           # Question 2: Iterative Poisson solver (4 figures)
├── q2_multicharge_superposition.py   # Question 2: Multi-charge superposition (4 figures)
├── q3_lightning_rod.py               # Question 3: Lightning rod (5 figures)
├── q1_supplementary.py               # Question 1: 3D plot, fringing, energy, bound charge (4 figures)
├── q3_supplementary.py               # Question 3: Parameter studies, zoom, comparison (5 figures)
├── report.tex                        # LaTeX source for the report
├── EE1204_SimAssignment1_Report.pdf  # Compiled report
└── figures/                          # All generated figures (auto-created)
    ├── q1/
    ├── q2/
    └── q3/
```

## Requirements

- Python 3.8+
- NumPy
- Matplotlib
- SciPy

Install dependencies:
```bash
pip install numpy matplotlib scipy
```

## Running the Simulations

Each script is standalone and can be run independently. Figures are saved to `./figures/q{1,2,3}/`.

```bash
# Question 1: Parallel-plate capacitor (6 figures)
python q1_parallel_plate_capacitor.py

# Question 2: Method of images (7 figures)
python q2_method_of_images.py

# Question 2: Iterative Poisson solver — cross-validation (4 figures)
python q2_iterative_poisson.py

# Question 2: Multi-charge superposition (4 figures)
python q2_multicharge_superposition.py

# Question 3: Lightning rod (5 figures)
python q3_lightning_rod.py
```

## Key Results

| Problem | Grid | Method | Key Validation |
|---------|------|--------|----------------|
| Q1: Capacitor | 120×120 | Gauss–Seidel (5000 iters) | E₀ ≈ 3000 V/m ✓ |
| Q2: Sphere | 300×300 | Method of Images | V = 0 on sphere (< 10⁻⁵) ✓ |
| Q2: Sphere | 201×201 | Iterative Poisson | Matches MoI profile ✓ |
| Q3: Lightning Rod | 100×100 | Gauss–Seidel (7000 iters) | β_num = 7.5 (theory: 10.9) ✓ |

## License

This code is submitted as part of the EE1204 coursework at IIT Hyderabad.
