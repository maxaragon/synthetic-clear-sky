#!/usr/bin/env python3
"""
colorspace_utils.py

Color space conversions for sky imaging:
- sRGB ↔ Linear RGB
- Linear RGB ↔ XYZ
- XYZ ↔ sRGB (full pipeline)
- Radiance scaling and normalization
"""

import numpy as np

try:
    import colour
except ImportError:
    colour = None


def srgb_to_linear(rgb_srgb):
    """
    Remove sRGB gamma encoding to get scene-linear RGB.
    
    Args:
        rgb_srgb: sRGB values in [0, 1] or [0, 255], shape (..., 3) or (3, ...)
    
    Returns:
        Linear RGB in [0, 1], same shape as input
    """
    # Normalize to [0, 1] if in [0, 255]
    if rgb_srgb.dtype == np.uint8 or np.max(rgb_srgb) > 1.0:
        rgb_norm = rgb_srgb.astype(np.float32) / 255.0
    else:
        rgb_norm = rgb_srgb.astype(np.float32)
    
    # sRGB inverse gamma
    linear = np.where(
        rgb_norm <= 0.04045,
        rgb_norm / 12.92,
        np.power((rgb_norm + 0.055) / 1.055, 2.4)
    )
    
    return linear


def linear_to_srgb(rgb_linear):
    """
    Apply sRGB gamma encoding to linear RGB.
    
    Args:
        rgb_linear: Linear RGB in [0, 1], shape (..., 3) or (3, ...)
    
    Returns:
        sRGB values in [0, 1], same shape as input
    """
    rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
    
    # sRGB gamma
    srgb = np.where(
        rgb_linear <= 0.0031308,
        12.92 * rgb_linear,
        1.055 * np.power(rgb_linear, 1.0 / 2.4) - 0.055
    )
    
    return np.clip(srgb, 0.0, 1.0)


def srgb_to_xyz(rgb_srgb, illuminant='D65'):
    """
    Convert sRGB to CIE XYZ tristimulus values.
    
    Uses standard sRGB→XYZ matrix (D65 white point by default).
    
    Args:
        rgb_srgb: sRGB values, shape (..., 3) with channels in last dimension
        illuminant: White point ('D65', 'E', etc.)
    
    Returns:
        XYZ values, same shape as input
    """
    # First convert to linear
    rgb_linear = srgb_to_linear(rgb_srgb)
    
    # sRGB→XYZ matrix (D65)
    # From: https://www.color.org/srgb.pdf
    M_sRGB_to_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ])
    
    # Apply matrix
    # Handle different input shapes
    original_shape = rgb_linear.shape
    
    if rgb_linear.shape[-1] == 3:
        # (..., 3) format
        rgb_flat = rgb_linear.reshape(-1, 3)
        xyz_flat = rgb_flat @ M_sRGB_to_XYZ.T
        xyz = xyz_flat.reshape(original_shape)
    else:
        # Assume (3, ...) format
        rgb_flat = rgb_linear.reshape(3, -1).T
        xyz_flat = rgb_flat @ M_sRGB_to_XYZ.T
        xyz = xyz_flat.T.reshape(original_shape)
    
    return xyz


def xyz_to_srgb_simple(xyz, clip=True):
    """
    Convert CIE XYZ to sRGB using direct matrix transform.
    
    Args:
        xyz: XYZ values, shape (..., 3) with channels in last dimension
        clip: If True, clip RGB to [0, 1]
    
    Returns:
        sRGB values in [0, 1], same shape as input
    """
    # XYZ→sRGB matrix (D65)
    M_XYZ_to_sRGB = np.array([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252]
    ])
    
    # Apply matrix to get linear RGB
    original_shape = xyz.shape
    
    if xyz.shape[-1] == 3:
        xyz_flat = xyz.reshape(-1, 3)
        rgb_linear_flat = xyz_flat @ M_XYZ_to_sRGB.T
        rgb_linear = rgb_linear_flat.reshape(original_shape)
    else:
        xyz_flat = xyz.reshape(3, -1).T
        rgb_linear_flat = xyz_flat @ M_XYZ_to_sRGB.T
        rgb_linear = rgb_linear_flat.T.reshape(original_shape)
    
    if clip:
        rgb_linear = np.clip(rgb_linear, 0.0, 1.0)
    
    # Apply sRGB gamma
    rgb_srgb = linear_to_srgb(rgb_linear)
    
    return rgb_srgb


def xyz_to_srgb_colour(xyz, illuminant='E', cat='CAT02', clip=True):
    """
    Convert XYZ to sRGB using the colour-science library.
    
    This is the same approach as in your notebook code.
    Provides chromatic adaptation if needed.
    
    Args:
        xyz: XYZ values, shape (3, N) or (..., 3)
        illuminant: Source illuminant ('D65', 'E', etc.)
        cat: Chromatic adaptation transform
        clip: If True, clip RGB to [0, 1]
    
    Returns:
        sRGB values in [0, 1]
    """
    if colour is None:
        raise ImportError("colour-science is required for xyz_to_srgb_colour but is not installed")

    sRGB = colour.RGB_COLOURSPACES['sRGB']
    illuminant_values = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer'][illuminant]
    
    # Handle different input shapes
    if xyz.shape[0] == 3 and len(xyz.shape) == 2:
        # (3, N) format - transpose for colour library
        xyz_input = xyz.T
    else:
        # (..., 3) format
        original_shape = xyz.shape
        xyz_input = xyz.reshape(-1, 3)
    
    # Convert
    rgb = colour.XYZ_to_RGB(
        xyz_input,
        colourspace=sRGB,
        illuminant=illuminant_values,
        chromatic_adaptation_transform=cat,
        apply_cctf_encoding=True  # Apply sRGB gamma
    )
    
    if clip:
        rgb = np.clip(rgb, 0.0, 1.0)
    
    # Restore original shape
    if xyz.shape[0] == 3 and len(xyz.shape) == 2:
        rgb = rgb.T
    else:
        rgb = rgb.reshape(original_shape)
    
    return rgb


def normalize_xyz_by_luminance(xyz, percentile=98, target_value=1.0):
    """
    Normalize XYZ values by Y channel (luminance) percentile.
    
    This is the normalization approach from your notebook.
    
    Args:
        xyz: XYZ array, shape (3, ...) or (..., 3)
        percentile: Percentile for normalization (default 98)
        target_value: Target value for the percentile (default 1.0)
    
    Returns:
        Normalized XYZ, normalization factor
    """
    # Extract Y channel (luminance)
    if xyz.shape[0] == 3:
        Y = xyz[1]
    else:
        Y = xyz[..., 1]
    
    # Compute normalization factor
    lum_norm = np.percentile(Y[Y > 0], percentile)
    
    # Normalize
    xyz_normalized = xyz / lum_norm * target_value
    
    return xyz_normalized, lum_norm


def scale_xyz_to_radiance_units(xyz_normalized, radiance_scale=1000.0):
    """
    Scale normalized XYZ to approximate radiance units (W/m²/sr).
    
    This is an empirical scaling to match synthetic XYZ ranges.
    
    Args:
        xyz_normalized: XYZ in [0, 1] range
        radiance_scale: Scaling factor (adjust based on calibration)
    
    Returns:
        XYZ in radiance units
    """
    return xyz_normalized * radiance_scale


def radiance_to_xyz_chromaticity(radiance, chromaticity_xy, clip=True):
    """
    Convert radiance (single channel) to XYZ using chromaticity coordinates.
    
    Given Y (luminance/radiance) and CIE xy chromaticity, reconstruct X and Z.
    
    Args:
        radiance: Y channel values (luminance/radiance)
        chromaticity_xy: Tuple (x, y) or arrays of shape matching radiance
        clip: If True, ensure non-negative XYZ
    
    Returns:
        XYZ array with shape (..., 3)
    """
    if isinstance(chromaticity_xy, tuple):
        x, y = chromaticity_xy
    else:
        x = chromaticity_xy[..., 0]
        y = chromaticity_xy[..., 1]
    
    # XYZ from Y and xy
    # X = (x/y) * Y
    # Z = ((1-x-y)/y) * Y
    
    y_safe = np.maximum(y, 1e-6)  # Avoid division by zero
    
    X = (x / y_safe) * radiance
    Y = radiance
    Z = ((1.0 - x - y) / y_safe) * radiance
    
    if clip:
        X = np.maximum(X, 0.0)
        Z = np.maximum(Z, 0.0)
    
    # Stack into XYZ
    xyz = np.stack([X, Y, Z], axis=-1)
    
    return xyz


def compute_chromaticity(xyz):
    """
    Compute CIE xy chromaticity coordinates from XYZ.
    
    Args:
        xyz: XYZ values, shape (..., 3) or (3, ...)
    
    Returns:
        xy chromaticity, shape (..., 2) or (2, ...)
    """
    if xyz.shape[0] == 3:
        X, Y, Z = xyz[0], xyz[1], xyz[2]
        xyz_sum = X + Y + Z + 1e-6
        x = X / xyz_sum
        y = Y / xyz_sum
        return np.stack([x, y], axis=0)
    else:
        X, Y, Z = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        xyz_sum = X + Y + Z + 1e-6
        x = X / xyz_sum
        y = Y / xyz_sum
        return np.stack([x, y], axis=-1)


def rgb_to_grayscale(rgb, weights=None):
    """
    Convert RGB to grayscale using luminance weighting.
    
    Args:
        rgb: RGB array, shape (..., 3)
        weights: Custom weights [R, G, B]. If None, uses standard [0.2126, 0.7152, 0.0722]
    
    Returns:
        Grayscale array, shape (...)
    """
    if weights is None:
        weights = np.array([0.2126, 0.7152, 0.0722])
    
    return np.dot(rgb, weights)


if __name__ == "__main__":
    # Quick test
    print("Testing colorspace utilities...")
    
    # Test sRGB → XYZ → sRGB round-trip
    rgb_test = np.array([0.5, 0.3, 0.7])
    xyz_test = srgb_to_xyz(rgb_test)
    rgb_back = xyz_to_srgb_simple(xyz_test)
    
    print(f"Original RGB: {rgb_test}")
    print(f"XYZ: {xyz_test}")
    print(f"Round-trip RGB: {rgb_back}")
    print(f"Error: {np.max(np.abs(rgb_test - rgb_back)):.6f}")
    
    # Test normalization
    xyz_test_array = np.random.rand(3, 100, 100) * 1000
    xyz_norm, factor = normalize_xyz_by_luminance(xyz_test_array, percentile=98)
    print(f"\nNormalization factor: {factor:.2f}")
    print(f"Y range before: [{np.min(xyz_test_array[1]):.2f}, {np.max(xyz_test_array[1]):.2f}]")
    print(f"Y range after: [{np.min(xyz_norm[1]):.4f}, {np.max(xyz_norm[1]):.4f}]")
    
    print("\n✅ Colorspace utilities ready!")

