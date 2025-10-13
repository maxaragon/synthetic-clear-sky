#!/usr/bin/env python3
"""
mask_utils.py

Clear-sky detection and masking for model fitting:
- Chromaticity-based sky detection (for RGB/XYZ images)
- Cloud detection from water path or optical depth
- Saturation detection
- Outlier removal
"""

import numpy as np


def build_clear_sky_mask_xyz(xyz, ze_deg=None, spa_deg=None, 
                              sat_threshold=0.98, circumsolar_exclusion_deg=5.0,
                              percentile_trim=98):
    """
    Build clear-sky mask from XYZ image using chromaticity.
    
    Clear sky is identified by:
    1. Blue-ish chromaticity (CIE xy space)
    2. Not saturated
    3. Not in circumsolar region
    4. Not statistical outlier
    
    Args:
        xyz: XYZ array, shape (..., 3) or (3, ...)
        ze_deg: Zenith angles (optional, for horizon exclusion)
        spa_deg: Sun-pixel angles (optional, for circumsolar exclusion)
        sat_threshold: Saturation threshold as fraction of max value
        circumsolar_exclusion_deg: Exclude pixels within this angle of sun
        percentile_trim: Exclude brightest pixels above this percentile
    
    Returns:
        Boolean mask, True = clear sky
    """
    # Extract XYZ channels
    if xyz.shape[0] == 3:
        X, Y, Z = xyz[0], xyz[1], xyz[2]
    else:
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
    
    # 1. Chromaticity filter (clear sky is bluish)
    xyz_sum = X + Y + Z + 1e-6
    x_chrom = X / xyz_sum
    y_chrom = Y / xyz_sum
    
    # Clear sky chromaticity band
    # Typical daylight locus: x ∈ [0.22, 0.40], y ∈ [0.22, 0.48]
    # Blue sky is toward lower x (less red)
    chrom_mask = (
        (x_chrom > 0.20) & (x_chrom < 0.42) &
        (y_chrom > 0.20) & (y_chrom < 0.50)
    )
    
    # 2. Brightness range (exclude very dark and saturated)
    Y_min = np.percentile(Y[Y > 0], 10) if np.any(Y > 0) else 0
    Y_max = np.percentile(Y, 99.5) * sat_threshold
    brightness_mask = (Y > Y_min) & (Y < Y_max)
    
    # 3. Circumsolar exclusion
    if spa_deg is not None:
        circumsolar_mask = spa_deg > circumsolar_exclusion_deg
    else:
        circumsolar_mask = True
    
    # 4. Zenith angle filter (optional: exclude horizon artifacts)
    if ze_deg is not None:
        zenith_mask = ze_deg < 85.0
    else:
        zenith_mask = True
    
    # Combine all criteria
    combined_mask = chrom_mask & brightness_mask & circumsolar_mask & zenith_mask
    
    # 5. Outlier removal: trim brightest pixels
    if np.any(combined_mask) and percentile_trim < 100:
        Y_clear = Y[combined_mask]
        outlier_threshold = np.percentile(Y_clear, percentile_trim)
        combined_mask &= (Y <= outlier_threshold)
    
    return combined_mask


def build_clear_sky_mask_radiance(radiance, ze_deg=None, spa_deg=None,
                                   sat_threshold_value=None,
                                   circumsolar_exclusion_deg=5.0,
                                   percentile_trim=98):
    """
    Build clear-sky mask from radiance image.
    
    Simpler than XYZ version - uses only intensity.
    
    Args:
        radiance: 2D radiance array (W/m²/sr)
        ze_deg: Zenith angles (optional)
        spa_deg: Sun-pixel angles (optional)
        sat_threshold_value: Absolute saturation threshold (if None, uses percentile)
        circumsolar_exclusion_deg: Exclude pixels within this angle of sun
        percentile_trim: Exclude brightest pixels above this percentile
    
    Returns:
        Boolean mask, True = clear sky
    """
    # 1. Brightness range
    rad_min = np.percentile(radiance[radiance > 0], 10) if np.any(radiance > 0) else 0
    
    if sat_threshold_value is not None:
        rad_max = sat_threshold_value
    else:
        rad_max = np.percentile(radiance, 99.5) * 0.98
    
    brightness_mask = (radiance > rad_min) & (radiance < rad_max)
    
    # 2. Circumsolar exclusion
    if spa_deg is not None:
        circumsolar_mask = spa_deg > circumsolar_exclusion_deg
    else:
        circumsolar_mask = True
    
    # 3. Zenith angle filter
    if ze_deg is not None:
        zenith_mask = ze_deg < 85.0
    else:
        zenith_mask = True
    
    # Combine
    combined_mask = brightness_mask & circumsolar_mask & zenith_mask
    
    # 4. Outlier removal
    if np.any(combined_mask) and percentile_trim < 100:
        rad_clear = radiance[combined_mask]
        outlier_threshold = np.percentile(rad_clear, percentile_trim)
        combined_mask &= (radiance <= outlier_threshold)
    
    return combined_mask


def build_cloud_mask_from_lwp(liq_ice_wp, threshold=0.01, log_threshold=None):
    """
    Build cloud mask from liquid/ice water path.
    
    Args:
        liq_ice_wp: Liquid + ice water path (kg/m² or g/m²)
        threshold: LWP threshold (linear scale)
        log_threshold: Alternative threshold on log10(LWP + 1)
    
    Returns:
        Boolean mask, True = cloudy
    """
    if log_threshold is not None:
        lwp_log = np.log10(liq_ice_wp + 1)
        cloud_mask = lwp_log > log_threshold
    else:
        cloud_mask = liq_ice_wp > threshold
    
    return cloud_mask


def build_cloud_mask_from_tau(tau_cloud, threshold=0.5):
    """
    Build cloud mask from cloud optical depth.
    
    Args:
        tau_cloud: Cloud optical depth (dimensionless)
        threshold: Optical depth threshold
    
    Returns:
        Boolean mask, True = cloudy
    """
    return tau_cloud > threshold


def detect_saturation(image, threshold=0.95, dtype_max=None):
    """
    Detect saturated pixels.
    
    Args:
        image: Image array (any shape)
        threshold: Fraction of maximum value to consider saturated
        dtype_max: Maximum value for data type (if None, inferred from dtype)
    
    Returns:
        Boolean mask, True = saturated
    """
    if dtype_max is None:
        if image.dtype == np.uint8:
            dtype_max = 255
        elif image.dtype == np.uint16:
            dtype_max = 65535
        else:
            dtype_max = np.max(image)
    
    sat_threshold = dtype_max * threshold
    
    # For multi-channel, saturated if ANY channel is saturated
    if len(image.shape) > 2:
        sat_mask = np.any(image > sat_threshold, axis=-1)
    else:
        sat_mask = image > sat_threshold
    
    return sat_mask


def refine_mask_morphology(mask, operation='opening', kernel_size=3):
    """
    Refine binary mask using morphological operations.
    
    Useful for cleaning up noisy masks (remove small holes/islands).
    
    Args:
        mask: Boolean mask
        operation: 'opening', 'closing', 'erosion', 'dilation'
        kernel_size: Size of structuring element
    
    Returns:
        Refined boolean mask
    """
    try:
        from scipy.ndimage import binary_opening, binary_closing, binary_erosion, binary_dilation
        
        if operation == 'opening':
            return binary_opening(mask, structure=np.ones((kernel_size, kernel_size)))
        elif operation == 'closing':
            return binary_closing(mask, structure=np.ones((kernel_size, kernel_size)))
        elif operation == 'erosion':
            return binary_erosion(mask, structure=np.ones((kernel_size, kernel_size)))
        elif operation == 'dilation':
            return binary_dilation(mask, structure=np.ones((kernel_size, kernel_size)))
        else:
            return mask
    except ImportError:
        print("Warning: scipy not available, skipping morphological operation")
        return mask


def mask_statistics(mask, labels=None):
    """
    Compute statistics about a mask.
    
    Args:
        mask: Boolean mask
        labels: Optional list of labels for multi-class mask
    
    Returns:
        Dictionary with statistics
    """
    total_pixels = mask.size
    masked_pixels = np.sum(mask)
    fraction = masked_pixels / total_pixels if total_pixels > 0 else 0
    
    stats = {
        'total_pixels': int(total_pixels),
        'masked_pixels': int(masked_pixels),
        'fraction': float(fraction),
        'percentage': float(fraction * 100)
    }
    
    return stats


def combine_masks(masks, operation='and'):
    """
    Combine multiple masks with logical operations.
    
    Args:
        masks: List of boolean masks (same shape)
        operation: 'and', 'or', 'xor', 'not'
    
    Returns:
        Combined boolean mask
    """
    if len(masks) == 0:
        raise ValueError("Need at least one mask")
    
    result = masks[0].copy()
    
    for mask in masks[1:]:
        if operation == 'and':
            result &= mask
        elif operation == 'or':
            result |= mask
        elif operation == 'xor':
            result ^= mask
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    return result


def visualize_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """
    Create visualization of mask overlaid on image.
    
    Args:
        image: RGB image, shape (H, W, 3), values in [0, 255] or [0, 1]
        mask: Boolean mask, shape (H, W)
        color: RGB color for mask overlay
        alpha: Transparency of overlay
    
    Returns:
        RGB image with mask overlay
    """
    # Normalize image to [0, 255]
    if np.max(image) <= 1.0:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    
    # Create overlay
    overlay = img.copy()
    overlay[mask] = (1 - alpha) * overlay[mask] + alpha * np.array(color)
    
    return overlay.astype(np.uint8)


if __name__ == "__main__":
    # Quick test
    print("Testing mask utilities...")
    
    # Create synthetic XYZ data
    np.random.seed(42)
    xyz_test = np.random.rand(224, 224, 3) * 1000
    
    # Add some "sky" pixels (blue chromaticity)
    sky_indices = np.random.choice(224*224, size=10000, replace=False)
    sky_y, sky_x = np.unravel_index(sky_indices, (224, 224))
    xyz_test[sky_y, sky_x, 0] = 200  # Low X (less red)
    xyz_test[sky_y, sky_x, 1] = 250  # Medium Y
    xyz_test[sky_y, sky_x, 2] = 350  # High Z (more blue)
    
    # Build mask
    clear_mask = build_clear_sky_mask_xyz(xyz_test)
    
    # Statistics
    stats = mask_statistics(clear_mask)
    print(f"Clear sky pixels: {stats['masked_pixels']} / {stats['total_pixels']} ({stats['percentage']:.1f}%)")
    
    # Test radiance mask
    radiance_test = np.random.rand(224, 224) * 100
    radiance_mask = build_clear_sky_mask_radiance(radiance_test)
    stats_rad = mask_statistics(radiance_mask)
    print(f"Clear radiance pixels: {stats_rad['percentage']:.1f}%")
    
    print("\n✅ Mask utilities ready!")

