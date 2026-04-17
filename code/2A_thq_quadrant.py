"""
2A_thq_quadrant.py — Quadrant illustration for THQ amine in the 2A state.

Shows two complementary views of the steric environment:

  Panel A (left)  : 2D steric map with quadrant dividers, per-quadrant
                    footprint coverage (%), projected amine atom positions,
                    and the N→Ru interaction axis overlay.

  Panel B (right) : Side-view cross-section (xz-plane) showing the buried-
                    volume sphere, Ru metal centre, the N→Ru dative-bond
                    arrow, THQ atom projections in the xz plane, vdW radius
                    circles for atoms intersecting the sphere, and a
                    labelled "steric intersection" region.

Usage:
    python 2A_thq_quadrant.py

Output:
    ../output/steric_maps/2A/2A_thq_quadrant.png
"""

import os, sys, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.lines import Line2D

from morfeus import BuriedVolume, read_xyz
from morfeus.buried_volume import rotate_coordinates

# ── locate code/ directory and import shared config ───────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR   = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, SCRIPT_DIR)

from config import (
    RU_INDEX, Z_AXIS_ATOMS, XZ_PLANE_ATOMS,
    SPHERE_RADIUS, INCLUDE_H, RADII_TYPE, RADII_SCALE,
    N_CATALYST_ATOMS,
)

XYZ_FILE = os.path.join(BASE_DIR, "xyz_opt", "2A",
                         "opt_thq_2_limcat_b3lypd3bs1.xyz")
OUT_FILE = os.path.join(BASE_DIR, "output", "steric_maps", "2A",
                         "2A_thq_quadrant.png")

R      = SPHERE_RADIUS      # 3.5 Å
GRID   = 160                # steric map grid resolution
LEVELS = 200                # contour levels
CMAP   = "viridis"

# Bondi radii (Å) used for drawing vdW circles in side view
BONDI_RADII = {"H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52,
               "P": 1.80, "Ru": 1.90}

def vdw_scaled(sym: str) -> float:
    return BONDI_RADII.get(sym, 1.70) * RADII_SCALE


# ═══════════════════════════════════════════════════════════════════════════════
# Steric map grid — replicates morfeus internals, same as steric_map_2A_grid.py
# Returns additionally the rotated atom coords and rotation z-vector.
# ═══════════════════════════════════════════════════════════════════════════════

def compute_steric_grid(bv: BuriedVolume, grid: int = GRID):
    """
    Compute the 2D steric map surface and return rotated atom positions.

    Returns
    -------
    x_, y_      : 1-D coordinate arrays  (length = grid)
    z_arr       : 2-D surface array, shape (grid, grid)
                  z_arr[iy, ix] is the maximum amine-atom vdW height at (x_[ix], y_[iy])
    rot_coords  : (N_amine, 3) amine atom coords in the steric-map frame
    atoms       : list of morfeus Atom objects (amine only)
    z_vec       : unit vector  Ru → C_CO  (rotation reference)
    all_coords_c: (N_all, 3) ALL atom coords centered at Ru (pre-rotation)
    """
    atoms       = bv._atoms
    center      = np.array(bv._sphere.center)          # Ru position
    all_coords  = bv._all_coordinates.copy()
    coords      = np.array([a.coordinates for a in atoms])
    r           = bv._sphere.radius

    # Translate everything to Ru origin
    all_coords -= center
    coords     -= center

    # z-axis: direction from Ru toward C_CO (atom 18)
    z_ax_idx   = np.array(bv._z_axis_atoms) - 1        # 0-indexed
    z_ax_mean  = np.mean(all_coords[z_ax_idx], axis=0)
    z_vec      = z_ax_mean / np.linalg.norm(z_ax_mean)

    # Rotate so that z_vec (Ru→CO) aligns with [0,0,-1].
    # After rotation: CO is at −z, amine is at +z.
    rot_coords = rotate_coordinates(coords, z_vec, np.array([0, 0, -1]))

    # Build steric surface grid
    x_ = np.linspace(-r, r, grid)
    y_ = np.linspace(-r, r, grid)

    z_flat = []
    for xy in np.dstack(np.meshgrid(x_, y_)).reshape(-1, 2):
        xp, yp = xy
        if np.linalg.norm(xy) > r:
            z_flat.append(np.nan)
            continue
        z_list = []
        for i, atom in enumerate(atoms):
            xs, ys, zs = rot_coords[i]
            test = atom.radius**2 - (xp - xs)**2 - (yp - ys)**2
            if test >= 0:
                z_list.append(math.sqrt(test) + zs)
        if z_list:
            z_max = max(z_list)
            if z_max < 0 and np.linalg.norm([xp, yp, z_max]) >= r:
                z_max = np.nan
        else:
            z_max = np.nan
        z_flat.append(z_max)

    # shape: (grid, grid) — z_arr[iy, ix] corresponds to (x_[ix], y_[iy])
    z_arr = np.array(z_flat).reshape(grid, grid)
    return x_, y_, z_arr, rot_coords, atoms, z_vec, all_coords


# ═══════════════════════════════════════════════════════════════════════════════
# Per-quadrant 2D footprint coverage (fraction of sphere area covered by amine)
# ═══════════════════════════════════════════════════════════════════════════════

def quadrant_coverage(x_: np.ndarray, y_: np.ndarray, z_arr: np.ndarray):
    """
    Returns {'NE': %, 'NW': %, 'SW': %, 'SE': %}

    Coverage = (grid cells in quadrant with amine atom projection) /
               (total grid cells in quadrant within sphere)  ×  100
    """
    counts = {q: [0, 0] for q in ('NE', 'NW', 'SW', 'SE')}   # [covered, total]

    for iy, yv in enumerate(y_):
        for ix, xv in enumerate(x_):
            if xv == 0.0 or yv == 0.0:
                continue
            if xv**2 + yv**2 > R**2:
                continue
            if   xv > 0 and yv > 0:  q = 'NE'
            elif xv < 0 and yv > 0:  q = 'NW'
            elif xv < 0 and yv < 0:  q = 'SW'
            else:                    q = 'SE'
            counts[q][1] += 1
            if not np.isnan(z_arr[iy, ix]):
                counts[q][0] += 1

    return {q: (c / t * 100 if t > 0 else 0.0) for q, (c, t) in counts.items()}


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)

    elements, coordinates = read_xyz(XYZ_FILE)
    n_atoms   = len(elements)

    # 2A state: no trailing fragment — amine = atoms 64..84 (1-indexed)
    cat_atoms = list(range(1, N_CATALYST_ATOMS + 1))   # 1-indexed
    excluded  = cat_atoms

    bv = BuriedVolume(
        elements, coordinates, RU_INDEX,
        excluded_atoms=excluded,
        z_axis_atoms=Z_AXIS_ATOMS,
        xz_plane_atoms=XZ_PLANE_ATOMS,
        radius=R,
        include_hs=INCLUDE_H,
        radii_type=RADII_TYPE,
        radii_scale=RADII_SCALE,
    )
    vbur_total = bv.fraction_buried_volume * 100
    print(f"THQ  %V_bur (total) = {vbur_total:.2f}%")

    # ── compute steric map + rotated coordinates ──────────────────────────────
    x_, y_, z_arr, rot_coords, atoms, z_vec, all_coords_c = \
        compute_steric_grid(bv, GRID)

    cov = quadrant_coverage(x_, y_, z_arr)
    print(f"Quadrant 2D coverage:  NE={cov['NE']:.1f}%  NW={cov['NW']:.1f}%  "
          f"SW={cov['SW']:.1f}%  SE={cov['SE']:.1f}%")

    # Amine element symbols (atoms 64..84, 0-indexed 63..83 in elements list)
    amine_syms = [elements[i] for i in range(N_CATALYST_ATOMS, n_atoms)]

    # ── rotate reference atoms into the steric-map frame ─────────────────────
    # C_CO = atom 18 (1-indexed) → index 17;  P_LEFT = atom 2 → index 1
    def rot_single(idx0: int) -> np.ndarray:
        """Rotate atom at 0-indexed position into steric-map frame."""
        return rotate_coordinates(
            all_coords_c[idx0:idx0+1], z_vec, np.array([0, 0, -1])
        )[0]

    co_rot = rot_single(17)   # C of CO ligand (atom 18)
    p_rot  = rot_single(1)    # P_LEFT (atom 2)

    # Locate amine N in rotated frame (first N in amine atom list)
    N_amine_rot = None
    N_amine_sym_idx = None
    for i, sym in enumerate(amine_syms):
        if sym == 'N':
            N_amine_rot    = rot_coords[i]
            N_amine_sym_idx = i
            break

    # ── Ru–N bond distance ────────────────────────────────────────────────────
    bond_len = 0.0   # default; overwritten when N is found
    if N_amine_rot is not None:
        ru_pos   = np.array(coordinates[RU_INDEX - 1])
        n_pos    = np.array(coordinates[N_CATALYST_ATOMS + N_amine_sym_idx])
        bond_len = np.linalg.norm(ru_pos - n_pos)
        print(f"Ru–N(amine) distance = {bond_len:.3f} Å")

    # ══════════════════════════════════════════════════════════════════════════
    # Figure — two panels
    # ══════════════════════════════════════════════════════════════════════════
    fig = plt.figure(figsize=(16, 8.0), facecolor='white')

    # axes positions: [left, bottom, width, height]
    ax_A  = fig.add_axes([0.04,  0.14, 0.43, 0.78])   # Panel A: steric map
    ax_B  = fig.add_axes([0.55,  0.08, 0.41, 0.84])   # Panel B: side view
    ax_cb = fig.add_axes([0.04,  0.08, 0.43, 0.028])  # colorbar for Panel A

    # ─────────────────────────────────────────────────────────────────────────
    # Panel A  :  2D steric map + quadrant overlay
    # ─────────────────────────────────────────────────────────────────────────

    # 1. Subtle quadrant background tints (drawn BEFORE contourf)
    theta_q = np.linspace(0, np.pi / 2, 200)
    q_tints = {
        'NE': ('#ffdddd', +1, +1),
        'NW': ('#ddeeff', -1, +1),
        'SW': ('#e8e8ff', -1, -1),
        'SE': ('#fffadd', +1, -1),
    }
    for q, (col, sx, sy) in q_tints.items():
        # Build a quarter-circle wedge polygon for each quadrant
        t  = theta_q                                    # angles 0 → π/2
        wx = sx * np.cos(t) * R                         # x values (signed)
        wy = sy * np.sin(t) * R                         # y values (signed)
        xs = np.concatenate([[0], wx, [0]])
        ys = np.concatenate([[0], wy, [0]])
        ax_A.fill(xs, ys, color=col, alpha=0.18, zorder=1)

    # 2. Filled contour steric map
    cf = ax_A.contourf(x_, y_, z_arr, LEVELS, cmap=CMAP, vmin=-R, vmax=R, zorder=2)

    # 3. Sphere boundary circle
    theta_c = np.linspace(0, 2 * np.pi, 400)
    ax_A.plot(R * np.cos(theta_c), R * np.sin(theta_c),
              color='white', lw=2.2, zorder=5)

    # 4. Quadrant dividers
    ax_A.axhline(0, color='white', lw=1.3, ls='--', alpha=0.85, zorder=6)
    ax_A.axvline(0, color='white', lw=1.3, ls='--', alpha=0.85, zorder=6)

    # 5. Quadrant label + coverage boxes
    q_label_pos = {
        'NE': ( R * 0.62,  R * 0.62),
        'NW': (-R * 0.62,  R * 0.62),
        'SW': (-R * 0.62, -R * 0.62),
        'SE': ( R * 0.62, -R * 0.62),
    }
    for q, (qx, qy) in q_label_pos.items():
        ax_A.text(qx, qy, f'{q}\n{cov[q]:.0f}%',
                  color='white', fontsize=10.5, ha='center', va='center',
                  fontweight='bold', zorder=9,
                  bbox=dict(boxstyle='round,pad=0.3', fc='#000000',
                            alpha=0.48, ec='none'))

    # 6. Projected amine atom positions (xy projection)
    a_color = {'N': '#00cfff', 'C': '#cccccc', 'H': '#eeeeee'}
    a_size  = {'N': 130,       'C': 65,        'H': 18}
    for sym, rc in zip(amine_syms, rot_coords):
        ax_A.scatter(rc[0], rc[1],
                     s=a_size.get(sym, 40),
                     c=a_color.get(sym, '#cccccc'),
                     alpha=0.72, zorder=8,
                     edgecolors='k', linewidths=0.3)

    # 7. Amine N — special star marker
    if N_amine_rot is not None:
        ax_A.scatter(N_amine_rot[0], N_amine_rot[1],
                     s=200, c='cyan', marker='*', zorder=11,
                     edgecolors='navy', linewidths=0.9)

    # 8. P_LEFT projection — orientation reference (defines xz-plane → x-axis)
    ax_A.scatter(p_rot[0], p_rot[1],
                 s=100, c='#ff9900', marker='D', zorder=9,
                 edgecolors='k', linewidths=0.6)
    ax_A.text(p_rot[0] + 0.12, p_rot[1] + 0.12, 'P',
              color='#ff9900', fontsize=9, fontweight='bold', zorder=10)

    # 9. Ru centre
    ax_A.scatter(0, 0, s=170, c='gold', marker='o', zorder=10,
                 edgecolors='black', linewidths=1.1)
    ax_A.text(0.08, -0.22, 'Ru', color='gold', fontsize=9,
              fontweight='bold', zorder=11)

    # 10. N→Ru direction arrow (from N projected position toward Ru)
    if N_amine_rot is not None:
        nx2d = np.array([N_amine_rot[0], N_amine_rot[1]])
        # draw from 85% of N position to 10% (near Ru)
        ax_A.annotate(
            '',
            xy   =(nx2d[0] * 0.10, nx2d[1] * 0.10),
            xytext=(nx2d[0] * 0.82, nx2d[1] * 0.82),
            arrowprops=dict(arrowstyle='->', color='#ff4da6',
                            lw=2.4, mutation_scale=16),
            zorder=12,
        )
        # Label near midpoint
        mid  = nx2d * 0.46
        angle_deg = np.degrees(np.arctan2(nx2d[1], nx2d[0]))
        ax_A.text(mid[0], mid[1] - 0.22, 'N→Ru',
                  color='#ff4da6', fontsize=9, fontweight='bold',
                  ha='center', rotation=angle_deg, zorder=12)

    # 11. "Into-page" z-axis symbol at Ru (the steric map looks along +z, i.e. from N toward Ru)
    # Draw a small ⊗ to indicate the z-axis points into the page at Ru centre
    ax_A.text(-R * 1.05, R * 1.00,
              '⊗ z-axis\n(N→Ru, into page)',
              color='white', fontsize=8, va='top',
              bbox=dict(boxstyle='round,pad=0.25', fc='#222222',
                        alpha=0.7, ec='none'),
              zorder=13)

    ax_A.set_xlim(-R * 1.14, R * 1.14)
    ax_A.set_ylim(-R * 1.14, R * 1.14)
    ax_A.set_aspect('equal')
    ax_A.set_xlabel('x (Å)', fontsize=12)
    ax_A.set_ylabel('y (Å)', fontsize=12)
    ax_A.set_title(
        f'THQ — 2A Steric Map\n'
        f'%V$_{{bur}}$ = {vbur_total:.1f}%   ·   '
        f'R = {R} Å   ·   Bondi × {RADII_SCALE}',
        fontsize=12, fontweight='bold', pad=7,
    )

    # Legend
    leg_A = [
        Line2D([0],[0], marker='o',  ls='none', markerfacecolor='gold',
               ms=10, mec='k', label='Ru'),
        Line2D([0],[0], marker='*',  ls='none', markerfacecolor='cyan',
               ms=12, mec='navy', label='N (amine)'),
        Line2D([0],[0], marker='o',  ls='none', markerfacecolor='#cccccc',
               ms=8,  mec='k', label='C (amine)'),
        Line2D([0],[0], marker='D',  ls='none', markerfacecolor='#ff9900',
               ms=7,  mec='k', label='P (xz ref)'),
        Line2D([0],[0], ls='--', color='white', lw=1.2, label='Quadrant lines'),
    ]
    #ax_A.legend(handles=leg_A, loc='lower right', fontsize=8, framealpha=0.75)

    # Colorbar
    cbar = fig.colorbar(cf, cax=ax_cb, orientation='horizontal')
    #cbar.set_label('z (Å) — amine height above equatorial plane', fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    # Panel letter
    #ax_A.text(-R * 1.10, R * 1.10, 'A', fontsize=17, fontweight='bold', va='top')

    # ─────────────────────────────────────────────────────────────────────────
    # Panel B  :  Side-view cross-section  (xz plane)
    # ─────────────────────────────────────────────────────────────────────────
    BG = '#f7f7f7'
    ax_B.set_facecolor(BG)

    # --- sphere (filled circle = cross-section)
    sphere_fill = Circle((0, 0), R, facecolor='#141430',
                          edgecolor='#4cc9f0', linewidth=2.5, zorder=2)
    ax_B.add_patch(sphere_fill)

    # Equatorial dashed line (y=0 in steric-map frame → x-axis in xz plane)
    ax_B.axhline(0, color='#4cc9f0', lw=0.8, ls=':', alpha=0.45, zorder=3)
    ax_B.axvline(0, color='#4cc9f0', lw=0.8, ls=':', alpha=0.45, zorder=3)

    # --- z-axis arrow (CO side = −z at bottom, amine side = +z at top)
    z_ext = R * 1.35
    ax_B.annotate('',
                  xy=(0, z_ext), xytext=(0, -z_ext),
                  arrowprops=dict(arrowstyle='->', color='#aaaacc',
                                  lw=1.3, mutation_scale=13),
                  zorder=4)
    ax_B.text(0.10, z_ext * 0.88, 'z', color='#aaaacc',
              fontsize=11, fontweight='bold', zorder=5)

    # Side labels for z-axis
    ax_B.text(0.12, -z_ext * 0.70,
              '← CO  (−z)', color='#ff8855', fontsize=8, va='center', zorder=5)
    ax_B.text(0.12,  z_ext * 0.50,
              'amine  (+z) →', color='#66ddff', fontsize=8, va='center', zorder=5)

    # --- C_CO atom in side view (xz projection: x=co_rot[0], z=co_rot[2])
    ax_B.scatter(co_rot[0], co_rot[2],
                 s=110, c='#ff6633', marker='s', zorder=10,
                 edgecolors='k', linewidths=0.8)
    ax_B.text(co_rot[0] + 0.17, co_rot[2], 'C(CO)',
              color='#ff6633', fontsize=8, va='center', zorder=11)

    # --- vdW radius dashed circles for amine atoms near/within sphere
    for sym, rc in zip(amine_syms, rot_coords):
        rv = vdw_scaled(sym)
        d_xz = math.sqrt(rc[0]**2 + rc[2]**2)     # distance in xz plane
        if d_xz < R + rv:
            ec = {'N': '#00cfff', 'C': '#999999', 'H': '#bbbbbb'}.get(sym, '#999999')
            ax_B.add_patch(Circle((rc[0], rc[2]), rv,
                                  fill=False, edgecolor=ec,
                                  lw=0.75, ls='--', alpha=0.40, zorder=6))

    # --- amine atom scatter in xz plane
    b_color = {'N': '#00cfff', 'C': '#888888', 'H': '#aaaaaa'}
    b_size  = {'N': 130,       'C': 60,        'H': 14}
    for sym, rc in zip(amine_syms, rot_coords):
        ax_B.scatter(rc[0], rc[2],
                     s=b_size.get(sym, 40),
                     c=b_color.get(sym, '#888888'),
                     alpha=0.80, zorder=7,
                     edgecolors='k', linewidths=0.3)

    # --- amine N: large star + N→Ru interaction arrow
    if N_amine_rot is not None:
        nx, ny_, nz = N_amine_rot    # ny_ unused (y-component)

        # Arrow from N position → Ru (0,0) in xz frame
        ax_B.annotate(
            '',
            xy    =(nx * 0.09, nz * 0.09),    # near Ru
            xytext=(nx * 0.88, nz * 0.88),    # near N
            arrowprops=dict(arrowstyle='->', color='#ff4da6',
                            lw=2.8, mutation_scale=20),
            zorder=13,
        )
        # Label along arrow
        ax_B.text(nx * 0.50, nz * 0.50 + 0.54,
                  'N→Ru\ninteraction',
                  color='#ff4da6', fontsize=9, ha='center', fontweight='bold',
                  zorder=14,
                  bbox=dict(boxstyle='round,pad=0.22', fc='black',
                            alpha=0.55, ec='none'))

        # N atom star marker (on top)
        ax_B.scatter(nx, nz, s=220, c='cyan', marker='*', zorder=15,
                     edgecolors='navy', linewidths=0.9)
        ax_B.text(nx + 0.14, nz, 'N', color='cyan',
                  fontsize=9, fontweight='bold', va='center', zorder=15)

        # Ru–N bond length annotation
        bond_len_xz = math.sqrt(nx**2 + nz**2)
        ax_B.annotate('',
                      xy=(0.0, 0.0), xytext=(nx * 0.85, nz * 0.85),
                      arrowprops=dict(arrowstyle='<->', color='#ff4da6',
                                      lw=1.0, mutation_scale=10),
                      zorder=12)
        ax_B.text(nx * 0.44 - 0.35, nz * 0.44 - 0.15,
                  f'd(Ru–N)\n≈ {bond_len:.2f} Å',
                  color='#ff4da6', fontsize=7.5, ha='center', zorder=12,
                  bbox=dict(boxstyle='round,pad=0.18', fc='black',
                            alpha=0.50, ec='none'))

    # --- Ru metal centre
    ax_B.scatter(0, 0, s=230, c='gold', marker='o', zorder=11,
                 edgecolors='black', linewidths=1.3)
    ax_B.text(0.14, 0.14, 'Ru', color='gold',
              fontsize=9, fontweight='bold', zorder=12)

    # --- sphere radius dimension line
    ax_B.annotate('',
                  xy=(R, 0.02), xytext=(0.02, 0.02),
                  arrowprops=dict(arrowstyle='<->', color='#4cc9f0',
                                  lw=1.5, mutation_scale=11),
                  zorder=8)
    ax_B.text(R / 2, 0.20, f'R = {R} Å',
              color='#4cc9f0', fontsize=8.5, ha='center', fontweight='bold',
              zorder=8)

    # --- steric intersection annotation
    # Find amine atom that is closest to the sphere surface (|r| ≈ R)
    # and annotate that region
    best = None
    best_dist = np.inf
    for sym, rc in zip(amine_syms, rot_coords):
        rv = vdw_scaled(sym)
        d_full = np.linalg.norm(rc)
        gap = abs(d_full + rv - R)      # how close vdW surface is to sphere boundary
        if gap < best_dist:
            best_dist = gap
            best = (sym, rc, rv)

    if best is not None:
        sym_b, rc_b, rv_b = best
        # Sphere surface point in the direction of rc_b (xz projection)
        angle_b = math.atan2(rc_b[2], rc_b[0])
        sx = R * math.cos(angle_b)
        sz = R * math.sin(angle_b)
        ax_B.annotate(
            'Sphere \nintersection',
            xy=(sx * 0.96, sz * 0.96),
            xytext=(sx * 0.55 + R * 0.70, sz * 0.55 + R * 0.28),
            arrowprops=dict(arrowstyle='->', color='#ff6b6b',
                            lw=1.7, mutation_scale=12),
            color='#ff6b6b', fontsize=8.5, ha='center', zorder=14,
            bbox=dict(boxstyle='round,pad=0.22', fc='#2a0000',
                      alpha=0.70, ec='none'),
        )
        # Draw the intersection "arc" highlight — a small arc near the sphere surface
        # spanning ±15° around the intersection angle
        arc_theta = np.linspace(angle_b - np.radians(18), angle_b + np.radians(18), 60)
        ax_B.plot(R * np.cos(arc_theta), R * np.sin(arc_theta),
                  color='#ff6b6b', lw=3.0, solid_capstyle='round', zorder=9)

    # --- P_LEFT reference atom in side view
    ax_B.scatter(p_rot[0], p_rot[2],
                 s=100, c='#ff9900', marker='D', zorder=9,
                 edgecolors='k', linewidths=0.6)
    ax_B.text(p_rot[0] + 0.17, p_rot[2], 'P (Ph$_2$)',
              color='#ff9900', fontsize=8, va='center', zorder=10)

    # --- "Steric map view direction" annotation (looking along −z in map frame
    #      means looking from amine side toward Ru, i.e., downward in this plot)
    #ax_B.annotate('',
    #              xy=(-R * 1.05, -R * 0.70),
    #              xytext=(-R * 1.05, R * 0.55),
    #              arrowprops=dict(arrowstyle='->', color='#ffee55',
    #                              lw=2.0, mutation_scale=14),
    #              zorder=14)
    #ax_B.text(-R * 1.08, -R * 0.85,
    #          'Steric map\nview direction\n(along −z)',
    #          color='#ffee55', fontsize=7.5, ha='center', va='top', zorder=14)

    # Styling
    ax_B.set_xlim(-R * 1.60, R * 2.20)
    ax_B.set_ylim(-R * 1.60, R * 1.60)
    ax_B.set_aspect('equal')
    ax_B.set_xlabel('x (Å)', fontsize=11, color='white')
    ax_B.set_ylabel('z (Å)   [CO ← Ru → N]', fontsize=11, color='white')
    #ax_B.set_title(
    #    'Side View: N→Ru Interaction & Steric Sphere Intersection\n'
    #    '(xz cross-section, x-axis defined by P reference)',
    #    fontsize=11, fontweight='bold', color='white', pad=8,
    #)
    ax_B.tick_params(colors='white', labelsize=9)
    for spine in ax_B.spines.values():
        spine.set_edgecolor('#4cc9f0')
        spine.set_linewidth(1.6)

    # Legend Panel B
    leg_B = [
        Line2D([0],[0], marker='o',  ls='none', c='gold',    ms=10, mec='k',
               label='Ru'),
        Line2D([0],[0], marker='*',  ls='none', c='cyan',    ms=12, mec='navy',
               label='N (amine)'),
        Line2D([0],[0], marker='o',  ls='none', c='#888888', ms=7,  mec='k',
               label='C (amine)'),
        Line2D([0],[0], marker='s',  ls='none', c='#ff6633', ms=7,  mec='k',
               label='C of CO'),
        Line2D([0],[0], marker='D',  ls='none', c='#ff9900', ms=7,  mec='k',
               label='P (xz ref)'),
        Line2D([0],[0], ls='--', c='#999999', lw=1.2, label='vdW radii'),
        Line2D([0],[0], ls='-',  c='#4cc9f0', lw=2.0, label='Sphere boundary'),
        Line2D([0],[0], ls='-',  c='#ff6b6b', lw=2.5, label='Steric intersection'),
        Line2D([0],[0], ls='-',  c='#ff4da6', lw=2.0, label='N→Ru interaction'),
    ]
    ax_B.legend(handles=leg_B, loc='lower right', fontsize=7.5,
                framealpha=0.50, labelcolor='white',
                facecolor='#111133', edgecolor='#4cc9f0', ncol=1)

    # Panel B letter
    #ax_B.text(-R * 1.55, R * 1.55, 'B', fontsize=17, fontweight='bold',
    #          color='white', va='top')

    # ── overall title ─────────────────────────────────────────────────────────
    #fig.suptitle(
    #    'THQ (1,2,3,4-Tetrahydroquinoline) — Steric Quadrant Analysis  [2A State]',
    #    fontsize=14, fontweight='bold', y=0.995,
    #)

    fig.savefig(OUT_FILE, dpi=220, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Saved → {OUT_FILE}")


if __name__ == "__main__":
    main()
