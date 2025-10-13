#!/usr/bin/env python3
"""
geometry_utils.py

Geometric calculations for fisheye sky imaging:
- Spherical coordinate conversions
- Sun-Pixel Angle (SPA) computation
- Azimuth derivation for synthetic cameras
"""

import numpy as np
import math


def angles_to_unit_vector(az_deg, ze_deg):
    """
    Convert azimuth and zenith angles to unit direction vectors.
    
    Args:
        az_deg: Azimuth angle(s) in degrees (0° = +X, counterclockwise, math convention)
        ze_deg: Zenith angle(s) in degrees (0° = up/zenith, 90° = horizon)
    
    Returns:
        Unit vector(s) with shape (..., 3) where last dimension is [x, y, z]
        z-axis points up (zenith)
    """
    az_rad = np.deg2rad(az_deg)
    ze_rad = np.deg2rad(ze_deg)
    
    # Spherical to Cartesian conversion
    x = np.sin(ze_rad) * np.cos(az_rad)
    y = np.sin(ze_rad) * np.sin(az_rad)
    z = np.cos(ze_rad)
    
    return np.stack([x, y, z], axis=-1)


def compute_sun_pixel_angle(az_pix_deg, ze_pix_deg, sun_az_deg, sun_ze_deg):
    """
    Compute Sun-Pixel Angle (SPA): angular separation between each pixel 
    look direction and the sun direction.
    
    This is the key geometric parameter for clear-sky modeling.
    
    Args:
        az_pix_deg: Pixel azimuth angles (degrees), array of any shape
        ze_pix_deg: Pixel zenith angles (degrees), same shape as az_pix_deg
        sun_az_deg: Solar azimuth (degrees), scalar
        sun_ze_deg: Solar zenith angle (degrees), scalar
    
    Returns:
        SPA (degrees), same shape as input pixel angles
    """
    # Convert to unit vectors
    v_pix = angles_to_unit_vector(az_pix_deg, ze_pix_deg)  # (..., 3)
    v_sun = angles_to_unit_vector(sun_az_deg, sun_ze_deg)  # (3,)
    
    # Dot product (broadcast sun vector)
    dot_product = np.sum(v_pix * v_sun, axis=-1)
    
    # Clip to handle numerical errors
    dot_product = np.clip(dot_product, -1.0, 1.0)
    
    # Angle from dot product
    spa_rad = np.arccos(dot_product)
    
    return np.rad2deg(spa_rad)


def derive_synthetic_azimuth(height, width, center=None, convention='math'):
    """
    Derive azimuth angles for a synthetic fisheye camera.
    
    Assumes a nadir-pointing fisheye camera with radially symmetric projection.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        center: Tuple (cx, cy) for image center. If None, uses geometric center.
        convention: 'math' (0°=+X, CCW) or 'nav' (0°=North, CW)
    
    Returns:
        Azimuth array (height, width) in degrees [0, 360)
    """
    if center is None:
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    else:
        cx, cy = center
    
    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Compute azimuth (math convention: 0° = +X, counterclockwise)
    dx = x_grid - cx
    dy = y_grid - cy
    az_rad = np.arctan2(dy, dx)
    az_deg = np.rad2deg(az_rad) % 360.0
    
    # Convert to navigation convention if requested
    if convention == 'nav':
        az_deg = (90.0 - az_deg) % 360.0
    
    return az_deg


def equidistant_fisheye_zenith(height, width, fov_deg=180.0, center=None):
    """
    Compute zenith angles for an equidistant fisheye projection.
    
    In equidistant projection: r = f * θ, where r is radial distance from center,
    f is a scale factor, and θ is zenith angle.
    
    Args:
        height: Image height in pixels
        width: Image width in pixels
        fov_deg: Full field of view in degrees (default 180° = hemisphere)
        center: Tuple (cx, cy) for image center. If None, uses geometric center.
    
    Returns:
        Zenith angle array (height, width) in degrees [0, fov_deg/2]
    """
    if center is None:
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    else:
        cx, cy = center
    
    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Radial distance from center
    dx = x_grid - cx
    dy = y_grid - cy
    r = np.sqrt(dx**2 + dy**2)
    
    # Maximum radius corresponds to fov_deg/2
    max_radius = min(cx, cy)
    theta_max_rad = np.deg2rad(fov_deg / 2.0)
    
    # Equidistant: θ = (r / r_max) * θ_max
    zenith_rad = (r / max_radius) * theta_max_rad
    zenith_deg = np.rad2deg(zenith_rad)
    
    return np.clip(zenith_deg, 0.0, fov_deg / 2.0)


def stereographic_fisheye_zenith(height, width, fov_deg=180.0, center=None, k=1.0):
    """
    Compute zenith angles for a stereographic fisheye projection.
    
    In stereographic projection: r = 2f * tan(θ/2)
    Modified version: r = f * tan(k*θ/2) for edge magnification control
    
    Args:
        height: Image height in pixels
        width: Image width in pixels  
        fov_deg: Full field of view in degrees
        center: Tuple (cx, cy) for image center
        k: Magnification parameter (1.0 = standard, >1 = more edge magnification)
    
    Returns:
        Zenith angle array (height, width) in degrees
    """
    if center is None:
        cx, cy = (width - 1) / 2.0, (height - 1) / 2.0
    else:
        cx, cy = center
    
    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    # Radial distance from center
    dx = x_grid - cx
    dy = y_grid - cy
    r = np.sqrt(dx**2 + dy**2)
    
    # Maximum radius corresponds to fov_deg/2
    max_radius = min(cx, cy)
    theta_max_rad = np.deg2rad(fov_deg / 2.0)
    
    # Inverse stereographic: θ = (2/k) * arctan(r / (f * tan_factor))
    # where tan_factor ensures r_max maps to theta_max
    tan_factor = max_radius / np.tan(k * theta_max_rad / 2.0)
    
    zenith_rad = (2.0 / k) * np.arctan(r / tan_factor)
    zenith_deg = np.rad2deg(zenith_rad)
    
    return np.clip(zenith_deg, 0.0, 90.0)


def find_sun_in_radiance(radiance, zen_deg, az_deg, search_radius_deg=15.0):
    """
    Locate the sun position from a radiance image by finding the brightest region.
    
    Args:
        radiance: 2D radiance array (W/m²/sr)
        zen_deg: 2D zenith angle array (degrees)
        az_deg: 2D azimuth angle array (degrees)
        search_radius_deg: Radius around peak to average (degrees)
    
    Returns:
        dict with keys:
            - 'sun_az': Solar azimuth (degrees)
            - 'sun_ze': Solar zenith (degrees)
            - 'peak_radiance': Maximum radiance value
            - 'confidence': Quality metric (peak / median ratio)
    """
    # Find peak radiance location
    peak_idx = np.unravel_index(np.argmax(radiance), radiance.shape)
    peak_radiance = radiance[peak_idx]
    
    # Get angles at peak
    sun_ze_initial = zen_deg[peak_idx]
    sun_az_initial = az_deg[peak_idx]
    
    # Refine by averaging nearby pixels (weighted by brightness)
    # This reduces noise and pixel discretization effects
    
    # Compute angular distance from peak
    spa_from_peak = compute_sun_pixel_angle(az_deg, zen_deg, sun_az_initial, sun_ze_initial)
    
    # Select pixels within search radius
    nearby_mask = spa_from_peak < search_radius_deg
    
    if np.sum(nearby_mask) > 0:
        # Weight by radiance
        weights = radiance[nearby_mask]
        weights = weights / np.sum(weights)
        
        # Weighted average of angles (convert to vectors to handle wrap-around)
        az_nearby = az_deg[nearby_mask]
        ze_nearby = zen_deg[nearby_mask]
        
        v_nearby = angles_to_unit_vector(az_nearby, ze_nearby)
        v_avg = np.average(v_nearby, weights=weights, axis=0)
        
        # Normalize
        v_avg = v_avg / np.linalg.norm(v_avg)
        
        # Convert back to angles
        sun_ze_refined = np.rad2deg(np.arccos(np.clip(v_avg[2], -1, 1)))
        sun_az_refined = np.rad2deg(np.arctan2(v_avg[1], v_avg[0])) % 360.0
    else:
        sun_ze_refined = sun_ze_initial
        sun_az_refined = sun_az_initial
    
    # Confidence metric
    median_radiance = np.median(radiance[radiance > 0])
    confidence = peak_radiance / median_radiance if median_radiance > 0 else 0.0
    
    return {
        'sun_az': float(sun_az_refined),
        'sun_ze': float(sun_ze_refined),
        'peak_radiance': float(peak_radiance),
        'confidence': float(confidence)
    }


def resample_angles_to_grid(az_source, ze_source, target_shape):
    """
    Resample angle arrays to a different grid size.
    
    Uses bilinear interpolation to preserve smooth angle transitions.
    
    Args:
        az_source: Source azimuth array
        ze_source: Source zenith array
        target_shape: Tuple (height, width) for output
    
    Returns:
        az_target, ze_target: Resampled angle arrays
    """
    from scipy.ndimage import zoom
    
    h_src, w_src = az_source.shape
    h_tgt, w_tgt = target_shape
    
    zoom_factors = (h_tgt / h_src, w_tgt / w_src)
    
    az_target = zoom(az_source, zoom_factors, order=1)
    ze_target = zoom(ze_source, zoom_factors, order=1)
    
    return az_target, ze_target


if __name__ == "__main__":
    # Quick test
    print("Testing geometry utilities...")
    
    # Test SPA calculation
    sun_az, sun_ze = 180.0, 30.0  # Sun in south, 30° elevation
    
    # Create a simple grid
    az_grid = np.linspace(0, 360, 10)
    ze_grid = np.linspace(0, 90, 10)
    AZ, ZE = np.meshgrid(az_grid, ze_grid)
    
    spa = compute_sun_pixel_angle(AZ, ZE, sun_az, sun_ze)
    
    print(f"Sun position: Az={sun_az}°, Ze={sun_ze}°")
    print(f"SPA range: {np.min(spa):.1f}° to {np.max(spa):.1f}°")
    print(f"Min SPA at: Az={AZ.flat[np.argmin(spa)]:.1f}°, Ze={ZE.flat[np.argmin(spa)]:.1f}°")
    
    # Test synthetic azimuth generation
    az_syn = derive_synthetic_azimuth(224, 224)
    print(f"\nSynthetic azimuth: shape={az_syn.shape}, range=[{np.min(az_syn):.1f}, {np.max(az_syn):.1f}]°")
    
    print("\n✅ Geometry utilities ready!")

