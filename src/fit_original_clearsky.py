#!/usr/bin/env python3
"""
Step 1: Fit per-channel (R,G,B) clear-sky coefficients from a cloudless all-sky image

Uses the Chauvin et al. model:
  L_ch(θ,γ) ≈ G_ch(θ) * S_ch(γ)
  G(θ) = A * (1 + C * cos(θ)^β) / (1 + B * cos(θ))             [β fixed ~ 0.32]
  S(γ) = D + E * γ^(-F) + H * cos(γ)

We:
  - Rebuild (θ,γ) on the SAME K-tan stereo geometry you used to unwrap
  - Select clear-sky pixels (exclude sun cone, near-horizon, saturation)
  - Fit G on γ≈90° band, then S on the rest
  - Repeat for channels R,G,B in *linear* RGB
  - Save JSON: {"R":[A,B,C,D,E,F,H], "G":[...], "B":[...]}

Outputs:
  - ../output/parameters/clearsky_RGB_coefficients.json
  - ../output/diagnostics/fit_diagnostics.png
"""

from pathlib import Path
import json
import numpy as np
from PIL import Image
import cv2
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))
from colorspace_utils import srgb_to_linear
from geometry_utils import compute_sun_pixel_angle

from scipy.optimize import least_squares
from scipy.ndimage import binary_erosion

# ----------------- CONFIG -----------------
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent

IMG = BASE_DIR / "input/clearsky_128px.png"
OUT_JSON = BASE_DIR / "output/parameters/clearsky_RGB_coefficients.json"
OUT_PLOT = BASE_DIR / "output/diagnostics/fit_diagnostics.png"

# Geometry settings (must match how the image was unwrapped)
K_STEREO = 1.4                # STEREO_K
THETA_MAX_DEG = 90.0          # fit on full hemisphere
BETA = 0.32                   # cos(theta)^beta (paper uses 0.32)

# Exclusions / bands
SUN_EXCL_DEG = 30.0           # exclude pixels closer than this to the sun
BAND_GAMMA_MIN = 88.0         # for gradation fit (γ≈90°)
BAND_GAMMA_MAX = 92.0
ZE_HORIZON_CUTOFF = 85.0      # exclude near-horizon for stability
SAT_THRESH = 0.98              # drop top ~2% brightest per-channel as possibly saturated

# ----------------- helpers -----------------
def k_tan_theta_from_radius(r, R, K=1.4):
    """Inverse K-tan stereographic: radius r → zenith θ (radians)."""
    denom = np.tan(K * np.pi / 4.0)
    t = (r / R) * denom
    t = np.clip(t, 0.0, 1e6)
    return (2.0 / K) * np.arctan(t)

def fit_disk_safe(alpha_or_gray):
    """Robust disk center/radius estimation."""
    a = alpha_or_gray.astype(np.float32)
    if a.ndim == 3:
        a = a.mean(-1)
    mask = a > (5.0 if a.max() > 1.5 else 0.02)
    H, W = a.shape
    if not mask.any():
        return (W-1)/2.0, (H-1)/2.0, min(W, H)/2.05
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean(), ys.mean()
    edge = mask & (~binary_erosion(mask, iterations=1))
    ys_e, xs_e = np.nonzero(edge)
    if xs_e.size < 30:
        R = min(W, H)/2.05
    else:
        R = np.median(np.hypot(xs_e - cx, ys_e - cy))
    return cx, cy, R

def circular_mask(image):
    """Perfect central circular mask."""
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, (w // 2, h // 2), min(h, w) // 2, 255, -1)
    if image.ndim == 3:
        return cv2.bitwise_and(image, image, mask=mask)
    return cv2.bitwise_and(image, mask)

def G_model(theta_rad, A, B, C, beta=BETA):
    ct = np.clip(np.cos(theta_rad), 0.0, 1.0)
    return A * (1.0 + C * np.power(ct, beta)) / (1.0 + B * ct + 1e-8)

def S_model(gamma_rad, D, E, F, H):
    g = np.clip(gamma_rad, 1e-3, np.pi)
    return D + E * np.power(g, -F) + H * np.cos(gamma_rad)

def percent_clip(x, p=98):
    s = np.sort(x.reshape(-1))
    k = int(np.clip(len(s) * p / 100.0, 0, len(s)-1))
    return s[k]

# ----------------- main -----------------
def main():
    print("="*70)
    print("STEP 1: Fitting per-channel RGB clear-sky coefficients")
    print("="*70)
    
    if not IMG.exists():
        print(f"ERROR: Input image not found: {IMG}")
        print("Please place your clear-sky image at: input/clearsky_128px.png")
        return
    
    # Load image (RGBA)
    im_rgba = Image.open(IMG).convert("RGBA")
    arr = np.array(im_rgba, np.uint8)
    rgb8 = arr[..., :3]
    alpha = arr[..., 3]
    H, W = alpha.shape
    print(f"Image: {W}×{H}")
    
    rgb = rgb8.astype(np.float32) / 255.0
    disk = alpha > 5
    if not disk.any():
        disk = (rgb.mean(-1) > 1/255)

    # Disk geometry → θ, φ
    cx, cy, R = fit_disk_safe(alpha if (alpha > 0).any() else rgb)
    print(f"Disk center=({cx:.2f},{cy:.2f})  R={R:.2f}px")
    
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    x = xx - cx
    y = cy - yy
    r = np.hypot(x, y)
    r_clipped = np.minimum(r, R)

    theta = k_tan_theta_from_radius(r_clipped, R, K=K_STEREO)
    ze_deg = np.degrees(theta)
    phi = np.arctan2(y, x)
    az_math = (np.degrees(phi) % 360.0).astype(np.float32)
    az_nav = (90.0 - az_math) % 360.0

    # Detect sun from image
    print("\n[Detecting Sun Position]")
    rgb_lin = srgb_to_linear(rgb)
    brightness = rgb_lin.mean(axis=-1)
    brightness_valid = brightness.copy()
    brightness_valid[~disk] = 0
    threshold = np.percentile(brightness_valid[disk], 99.9)
    sun_candidates = (brightness_valid > threshold) & disk
    
    if sun_candidates.any():
        ys_sun, xs_sun = np.nonzero(sun_candidates)
        sun_x, sun_y = xs_sun.mean(), ys_sun.mean()
        sun_idx_y, sun_idx_x = int(sun_y), int(sun_x)
        sun_az_math = az_math[sun_idx_y, sun_idx_x]
        sun_ze_deg = ze_deg[sun_idx_y, sun_idx_x]
        sun_az_nav = (90.0 - sun_az_math) % 360.0
        print(f"  Sun pixel: ({sun_x:.1f}, {sun_y:.1f})")
        print(f"  Sun position: az={sun_az_nav:.1f}° (nav), ze={sun_ze_deg:.1f}°")
    else:
        print("  WARNING: Could not detect sun, using default")
        sun_az_nav, sun_ze_deg = 180.0, 45.0

    SPA_deg = compute_sun_pixel_angle(az_nav, ze_deg, sun_az_nav, sun_ze_deg)
    gamma = np.radians(SPA_deg)

    # Clear-sky masks
    fov = (ze_deg <= THETA_MAX_DEG) & disk
    non_horizon = (ze_deg <= ZE_HORIZON_CUTOFF)
    sun_excl = (SPA_deg >= SUN_EXCL_DEG)
    clear_mask = fov & non_horizon & sun_excl
    print(f"\nClear-sky mask: {clear_mask.sum()} pixels ({100*clear_mask.sum()/(H*W):.1f}%)")

    # Per-channel fit
    coeffs = {}
    chan_names = ["R", "G", "B"]
    chans = [rgb_lin[...,0], rgb_lin[...,1], rgb_lin[...,2]]
    band = clear_mask & (SPA_deg >= BAND_GAMMA_MIN) & (SPA_deg <= BAND_GAMMA_MAX)

    print("\n[Fitting Chauvin Model per Channel]")
    for ch_name, I in zip(chan_names, chans):
        I = I.copy()
        sat_cut = percent_clip(I[clear_mask], SAT_THRESH*100.0)
        valid = clear_mask & (I <= sat_cut)

        # Fit G(θ)
        th_band = theta[band]
        I_band = I[band]
        if I_band.size < 200:
            th_band = theta[valid]
            I_band = I[valid]

        A0, B0, C0 = max(np.median(I_band), 1e-3), -0.5, 0.2
        bounds_G = ([0.0, -2.0, -2.0], [np.inf, 0.0, 1.5])
        def resid_G(params):
            A, B, C = params
            return (G_model(th_band, A,B,C) - I_band)
        sol_G = least_squares(resid_G, x0=[A0,B0,C0], bounds=bounds_G, loss='huber', f_scale=0.02)
        A, B, C = sol_G.x.tolist()

        # Fit S(γ)
        G_full = G_model(theta, A,B,C)
        denom = np.where(G_full <= 1e-8, 1e-8, G_full)
        S_samples = np.clip(I / denom, 1e-6, None)
        g_fit = gamma[valid]
        s_fit = S_samples[valid]
        mask_s = (np.degrees(g_fit) > 5.0)
        g_fit = g_fit[mask_s]
        s_fit = s_fit[mask_s]

        D0, E0, F0, H0 = float(np.percentile(s_fit, 10)), 1.0, 1.2, 0.2
        bounds_S = ([0.0, 0.0, 0.2, -1.0], [10.0, 1e4, 3.0, 1.0])
        def resid_S(params):
            D,E,F,H = params
            return (S_model(g_fit, D,E,F,H) - s_fit)
        sol_S = least_squares(resid_S, x0=[D0,E0,F0,H0], bounds=bounds_S, loss='huber', f_scale=0.02)
        D,E,F,H = sol_S.x.tolist()

        coeffs[ch_name] = [float(A), float(B), float(C), float(D), float(E), float(F), float(H)]
        print(f"  {ch_name}: A={A:.4f} B={B:.4f} C={C:.4f} D={D:.4f} E={E:.4f} F={F:.4f} H={H:.4f}")

    # Save JSON
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(coeffs, indent=2))
    print(f"\n✅ Saved coefficients: {OUT_JSON}")

    # Diagnostic plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(3, 3, figsize=(14, 12))
        theta_deg = np.degrees(theta)

        for i,(ch_name,I) in enumerate(zip(chan_names, chans)):
            A,B,C,D,E,F,H = coeffs[ch_name]
            
            im_disp = circular_mask((np.clip(I / percent_clip(I[clear_mask],98),0,1)*255).astype(np.uint8))
            axes[i,0].imshow(im_disp, cmap='gray')
            axes[i,0].set_title(f"{ch_name} channel"); axes[i,0].axis('off')

            band_mask = (SPA_deg >= BAND_GAMMA_MIN) & (SPA_deg <= BAND_GAMMA_MAX) & clear_mask
            tband = theta_deg[band_mask]; Iband = I[band_mask]
            axes[i,1].scatter(tband, Iband, s=4, alpha=0.2)
            tgrid = np.linspace(0, 90, 181)
            axes[i,1].plot(tgrid, G_model(np.radians(tgrid), A,B,C), 'r-', lw=2)
            axes[i,1].set_xlabel("θ (deg)"); axes[i,1].set_ylabel(f"{ch_name}")
            axes[i,1].set_title(f"Gradation G(θ)"); axes[i,1].grid(True, alpha=0.3)

            valid = clear_mask & (I <= percent_clip(I[clear_mask],98))
            g_fit = np.degrees(gamma[valid]); s_fit = np.clip((I/G_model(theta,A,B,C))[valid], 1e-6, None)
            axes[i,2].scatter(g_fit, s_fit, s=4, alpha=0.2)
            ggrid = np.linspace(5, 180, 351)
            axes[i,2].plot(ggrid, S_model(np.radians(ggrid), D,E,F,H), 'r-', lw=2)
            axes[i,2].set_xlabel("γ (deg)"); axes[i,2].set_ylabel("S")
            axes[i,2].set_title(f"Scattering S(γ)"); axes[i,2].grid(True, alpha=0.3)

        plt.tight_layout()
        OUT_PLOT.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT_PLOT, dpi=160)
        print(f"✅ Saved diagnostics: {OUT_PLOT}")
    except Exception as e:
        print(f"Diagnostic plot skipped: {e}")
    
    print("\n" + "="*70)
    print("STEP 1 COMPLETE!")
    print("Next: Run scripts/2_generate_synthetic_rgb.py")
    print("="*70)

if __name__ == "__main__":
    main()

