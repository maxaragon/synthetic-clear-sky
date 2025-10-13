#!/usr/bin/env python3
"""
clearsky_model.py

Parametric clear-sky radiance model based on:
"Cloud Detection Methodology Based on a Sky-imaging System"
https://www.sciencedirect.com/science/article/pii/S1876610215005044

Model: Y(θ, γ) = G(θ) × S(γ)
Where:
  - G(θ) = gradation function (zenith angle dependence)
  - S(γ) = scattering function (sun-pixel angle dependence)
  - θ = PZA (pixel zenith angle)
  - γ = SPA (sun-pixel angle)
"""

import numpy as np
import json
from pathlib import Path


class ClearSkyModel:
    """
    Parametric clear-sky radiance model.
    
    Factorized form: Radiance = G(zenith) × S(sun_angle)
    """
    
    def __init__(self, params_G=None, params_S=None):
        """
        Initialize model with parameters.
        
        Args:
            params_G: Dict with keys 'A', 'B', 'C' for gradation function
            params_S: Dict with keys 'D', 'E', 'F', 'H' for scattering function
        """
        if params_G is not None:
            self.A = params_G['A']
            self.B = params_G['B']
            self.C = params_G['C']
        else:
            # Default parameters (will be overwritten by fitting)
            self.A, self.B, self.C = 1.0, 0.0, 1.0
        
        if params_S is not None:
            self.D = params_S['D']
            self.E = params_S['E']
            self.F = params_S['F']
            self.H = params_S['H']
        else:
            # Default parameters
            self.D, self.E, self.F, self.H = 1.0, 0.0, -1.0, 0.0
    
    def G(self, theta_deg):
        """
        Gradation function: zenith angle dependence.
        
        G(θ) = A × cos(θ)^C + B
        
        Args:
            theta_deg: Zenith angle(s) in degrees
        
        Returns:
            Gradation value(s)
        """
        theta_rad = np.deg2rad(theta_deg)
        cos_theta = np.cos(theta_rad)
        return self.A * (cos_theta ** self.C) + self.B
    
    def S(self, gamma_deg):
        """
        Scattering function: sun-pixel angle dependence.
        
        S(γ) = D + E × γ^F + H × cos(γ)
        
        Args:
            gamma_deg: Sun-pixel angle(s) in degrees
        
        Returns:
            Scattering value(s)
        """
        gamma_rad = np.deg2rad(gamma_deg)
        # Handle negative exponents: avoid division by zero for gamma=0
        gamma_safe = np.maximum(gamma_deg, 0.01)
        return (self.D + 
                self.E * (gamma_safe ** self.F) + 
                self.H * np.cos(gamma_rad))
    
    def evaluate(self, pza_deg, spa_deg):
        """
        Evaluate full clear-sky model.
        
        Y = G(θ) × S(γ)
        
        Args:
            pza_deg: Pixel zenith angle(s) in degrees
            spa_deg: Sun-pixel angle(s) in degrees
        
        Returns:
            Clear-sky radiance/luminance values (same shape as inputs)
        """
        return np.maximum(0.0, self.G(pza_deg) * self.S(spa_deg))
    
    def get_parameters(self):
        """Return all parameters as a dictionary."""
        return {
            'G': {'A': float(self.A), 'B': float(self.B), 'C': float(self.C)},
            'S': {'D': float(self.D), 'E': float(self.E), 'F': float(self.F), 'H': float(self.H)}
        }
    
    def save(self, filepath):
        """Save parameters to JSON file."""
        params = self.get_parameters()
        with open(filepath, 'w') as f:
            json.dump(params, f, indent=2)
    
    @classmethod
    def load(cls, filepath):
        """Load parameters from JSON file."""
        with open(filepath, 'r') as f:
            params = json.load(f)
        return cls(params_G=params['G'], params_S=params['S'])


def fit_gradation_function(PZA_deg, Y_observed, SPA_deg, clear_mask, verbose=True):
    """
    Fit gradation function G(θ) = A × cos(θ)^C + B
    
    Strategy: Isolate zenith dependence by using pixels where sun influence is minimal
    (SPA ≈ 90° ± 2°, i.e., anti-solar region).
    
    Uses grid search on C, linear least squares for A, B.
    
    Args:
        PZA_deg: Pixel zenith angles (degrees), any shape
        Y_observed: Observed radiance/luminance, same shape as PZA_deg
        SPA_deg: Sun-pixel angles (degrees), same shape as PZA_deg
        clear_mask: Boolean mask indicating clear-sky pixels
        verbose: Print fitting progress
    
    Returns:
        Dict with fitted parameters and quality metrics
    """
    # Isolate anti-solar region (minimal sun influence)
    antisolar_mask = clear_mask & (np.abs(SPA_deg - 90.0) <= 2.0)
    
    n_antisolar = np.sum(antisolar_mask)
    if n_antisolar < 50:
        raise ValueError(f"Insufficient clear pixels near SPA=90° ({n_antisolar} found, need ≥50)")
    
    if verbose:
        print(f"  Using {n_antisolar} anti-solar pixels for gradation fit")
    
    # Extract data
    theta_fit = PZA_deg[antisolar_mask]
    y_fit = Y_observed[antisolar_mask]
    
    # Grid search for optimal C
    C_candidates = np.linspace(0.3, 2.0, 35)
    best_fit = None
    
    for C in C_candidates:
        # Build design matrix for linear least squares
        theta_rad = np.deg2rad(theta_fit)
        cos_theta = np.cos(theta_rad)
        
        # Handle potential numerical issues
        cos_theta = np.clip(cos_theta, 0.01, 1.0)
        
        Phi = np.column_stack([
            cos_theta ** C,           # Column for A
            np.ones_like(theta_rad)   # Column for B
        ])
        
        # Solve: min ||Phi @ [A, B]^T - y||²
        try:
            params, residuals, rank, s = np.linalg.lstsq(Phi, y_fit, rcond=None)
            A, B = params
            
            # Evaluate fit quality
            y_pred = Phi @ params
            rmse = np.sqrt(np.mean((y_pred - y_fit) ** 2))
            
            # R² coefficient
            ss_res = np.sum((y_fit - y_pred) ** 2)
            ss_tot = np.sum((y_fit - np.mean(y_fit)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            if best_fit is None or rmse < best_fit['rmse']:
                best_fit = {
                    'A': float(A),
                    'B': float(B),
                    'C': float(C),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'n_samples': int(n_antisolar)
                }
        except np.linalg.LinAlgError:
            continue
    
    if best_fit is None:
        raise RuntimeError("Gradation fitting failed (singular matrix)")
    
    if verbose:
        print(f"  Gradation: A={best_fit['A']:.2f}, B={best_fit['B']:.2f}, "
              f"C={best_fit['C']:.3f}, R²={best_fit['r2']:.4f}, RMSE={best_fit['rmse']:.2f}")
    
    return best_fit


def fit_scattering_function(SPA_deg, Y_observed, G_func, PZA_deg, clear_mask, verbose=True):
    """
    Fit scattering function S(γ) = D + E × γ^F + H × cos(γ)
    
    Strategy: Normalize observed data by gradation G(θ), then fit sun-angle dependence.
    
    Uses grid search on F (power-law exponent), linear least squares for D, E, H.
    
    Args:
        SPA_deg: Sun-pixel angles (degrees), any shape
        Y_observed: Observed radiance/luminance, same shape as SPA_deg
        G_func: Function that computes G(theta_deg)
        PZA_deg: Pixel zenith angles (degrees), same shape as SPA_deg
        clear_mask: Boolean mask indicating clear-sky pixels
        verbose: Print fitting progress
    
    Returns:
        Dict with fitted parameters and quality metrics
    """
    spa_fit = SPA_deg[clear_mask]
    y_fit = Y_observed[clear_mask]
    pza_fit = PZA_deg[clear_mask]
    
    n_samples = len(spa_fit)
    if n_samples < 50:
        raise ValueError(f"Insufficient clear pixels for scattering fit ({n_samples} found, need ≥50)")
    
    if verbose:
        print(f"  Using {n_samples} clear-sky pixels for scattering fit")
    
    # Normalize out gradation
    G_values = G_func(pza_fit)
    y_normalized = y_fit / np.maximum(G_values, 1e-6)
    
    # Grid search for optimal F (power-law exponent)
    F_candidates = np.linspace(-2.2, -0.3, 39)
    best_fit = None
    
    for F in F_candidates:
        # Build design matrix
        spa_rad = np.deg2rad(spa_fit)
        
        # Handle negative exponents: avoid division by zero
        spa_safe = np.maximum(spa_fit, 0.01)
        
        Phi = np.column_stack([
            np.ones_like(spa_fit),    # Column for D (constant)
            spa_safe ** F,             # Column for E (power law)
            np.cos(spa_rad)            # Column for H (forward scattering)
        ])
        
        # Solve: min ||Phi @ [D, E, H]^T - y||²
        try:
            params, residuals, rank, s = np.linalg.lstsq(Phi, y_normalized, rcond=None)
            D, E, H = params
            
            # Evaluate fit quality
            y_pred = Phi @ params
            rmse = np.sqrt(np.mean((y_pred - y_normalized) ** 2))
            
            # R² coefficient
            ss_res = np.sum((y_normalized - y_pred) ** 2)
            ss_tot = np.sum((y_normalized - np.mean(y_normalized)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            if best_fit is None or rmse < best_fit['rmse']:
                best_fit = {
                    'D': float(D),
                    'E': float(E),
                    'F': float(F),
                    'H': float(H),
                    'rmse': float(rmse),
                    'r2': float(r2),
                    'n_samples': int(n_samples)
                }
        except np.linalg.LinAlgError:
            continue
    
    if best_fit is None:
        raise RuntimeError("Scattering fitting failed (singular matrix)")
    
    if verbose:
        print(f"  Scattering: D={best_fit['D']:.3f}, E={best_fit['E']:.3f}, "
              f"F={best_fit['F']:.3f}, H={best_fit['H']:.3f}, R²={best_fit['r2']:.4f}, RMSE={best_fit['rmse']:.3f}")
    
    return best_fit


def fit_clearsky_model(Y_observed, PZA_deg, SPA_deg, clear_mask, verbose=True):
    """
    Fit complete clear-sky model (two-stage process).
    
    Stage 1: Fit gradation G(θ) using anti-solar pixels
    Stage 2: Fit scattering S(γ) using all clear pixels, normalized by G
    
    Args:
        Y_observed: Observed radiance/luminance
        PZA_deg: Pixel zenith angles (degrees)
        SPA_deg: Sun-pixel angles (degrees)
        clear_mask: Boolean mask for clear-sky pixels
        verbose: Print progress
    
    Returns:
        ClearSkyModel instance with fitted parameters
    """
    if verbose:
        print("Fitting clear-sky model...")
        print(f"Clear pixels: {np.sum(clear_mask)} / {clear_mask.size} "
              f"({100*np.mean(clear_mask):.1f}%)")
    
    # Stage 1: Fit gradation
    if verbose:
        print("\nStage 1: Fitting gradation G(θ)...")
    params_G = fit_gradation_function(PZA_deg, Y_observed, SPA_deg, clear_mask, verbose)
    
    # Create G function for stage 2
    def G_func(theta_deg):
        theta_rad = np.deg2rad(theta_deg)
        return params_G['A'] * (np.cos(theta_rad) ** params_G['C']) + params_G['B']
    
    # Stage 2: Fit scattering
    if verbose:
        print("\nStage 2: Fitting scattering S(γ)...")
    params_S = fit_scattering_function(SPA_deg, Y_observed, G_func, PZA_deg, clear_mask, verbose)
    
    # Create model
    model = ClearSkyModel(params_G=params_G, params_S=params_S)
    
    if verbose:
        print("\n✅ Model fitting complete!")
    
    return model


def fit_clearsky_model_xyz(XYZ_observed, PZA_deg, SPA_deg, clear_mask, channels=None, verbose=True):
    """
    Fit clear-sky model to all three XYZ channels.
    
    Args:
        XYZ_observed: XYZ array, shape (..., 3) or (3, ...)
        PZA_deg: Pixel zenith angles
        SPA_deg: Sun-pixel angles
        clear_mask: Clear-sky mask
        channels: List of channel names to fit (default: ['X', 'Y', 'Z'])
        verbose: Print progress
    
    Returns:
        Dict with models for each channel: {'X': model_X, 'Y': model_Y, 'Z': model_Z}
    """
    if channels is None:
        channels = ['X', 'Y', 'Z']
    
    # Determine data layout
    if XYZ_observed.shape[0] == 3:
        channel_data = [XYZ_observed[i] for i in range(3)]
    else:
        channel_data = [XYZ_observed[..., i] for i in range(3)]
    
    models = {}
    
    for i, channel_name in enumerate(['X', 'Y', 'Z']):
        if channel_name not in channels:
            continue
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fitting {channel_name} channel")
            print(f"{'='*60}")
        
        model = fit_clearsky_model(
            channel_data[i], PZA_deg, SPA_deg, clear_mask, verbose
        )
        models[channel_name] = model
    
    return models


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing clear-sky model...")
    
    # Create synthetic sky
    np.random.seed(42)
    pza = np.linspace(0, 85, 100)
    spa = np.linspace(0, 180, 100)
    PZA, SPA = np.meshgrid(pza, spa)
    
    # True parameters
    true_model = ClearSkyModel(
        params_G={'A': 100.0, 'B': 20.0, 'C': 0.8},
        params_S={'D': 1.0, 'E': 5.0, 'F': -1.2, 'H': 0.3}
    )
    
    # Generate data with noise
    Y_true = true_model.evaluate(PZA, SPA)
    Y_noisy = Y_true + np.random.normal(0, 5, Y_true.shape)
    
    # Create mask (exclude some regions to simulate clouds)
    clear_mask = np.random.rand(*Y_true.shape) > 0.3
    
    # Fit model
    fitted_model = fit_clearsky_model(Y_noisy, PZA, SPA, clear_mask, verbose=True)
    
    # Compare parameters
    print("\n" + "="*60)
    print("Parameter comparison:")
    print("="*60)
    true_params = true_model.get_parameters()
    fitted_params = fitted_model.get_parameters()
    
    for func in ['G', 'S']:
        print(f"\n{func}:")
        for key in true_params[func]:
            true_val = true_params[func][key]
            fitted_val = fitted_params[func][key]
            error = abs(fitted_val - true_val) / abs(true_val) * 100 if true_val != 0 else 0
            print(f"  {key}: true={true_val:.3f}, fitted={fitted_val:.3f}, error={error:.1f}%")
    
    print("\n✅ Clear-sky model ready!")

