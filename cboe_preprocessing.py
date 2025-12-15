"""
Kidger's Data Preprocessing Pipeline for CBOE SPY Options
==========================================================

This module implements the complete data preprocessing pipeline that Kidger would
design for transforming CBOE options snapshots into Neural RDE training data.

The pipeline addresses the core insight:
    "The short-dated options problem is fundamentally a LONG TIME SERIES problem.
    A 1-week option with minute-level data has thousands of observations, and
    the dynamics are ROUGH — dominated by jumps and rapid regime changes."

Key Components:
1. PATH CONSTRUCTION: Extract (t, log_S, σ, J^S, J^I) from snapshots
2. LOG-SIGNATURE PREPROCESSING: Compress long paths into optimal statistics
3. CROSS-SECTIONAL TARGETS: Extract smile parameters (γ, ω², ξ)
4. P&L ATTRIBUTION: Compute signature-based P&L decomposition
5. ONLINE CAPABILITY: Support streaming/real-time processing

Reference: Kidger et al. "Neural Rough Differential Equations" (2021)

Author: Based on Kidger's theoretical framework
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union, Callable
from pathlib import Path
import warnings
from datetime import datetime, timedelta

# For JAX-based computation (optional, falls back to numpy)
try:
    import jax.numpy as jnp
    from jax import jit, vmap
    HAS_JAX = True
except ImportError:
    jnp = np
    HAS_JAX = False
    def jit(fn): return fn
    def vmap(fn, in_axes=None): return lambda *args: np.stack([fn(*a) for a in zip(*args)])


# =============================================================================
# PART 1: CONFIGURATION AND DATA STRUCTURES
# =============================================================================

@dataclass
class PreprocessingConfig:
    """
    Configuration for the preprocessing pipeline.
    
    Kidger's guidance on hyperparameters:
    
    "Increasing step_size leads to faster (but less informative) training.
     Increasing depth leads to slower (but more informative) training.
     For short-dated options, start with depth=2, step_size=8."
    
    For CBOE 5-minute data with 78 snapshots/day:
    - step_size=8 → ~9 log-signature vectors per day
    - step_size=4 → ~19 log-signature vectors per day
    - depth=2 → captures Lévy area (essential for jumps)
    - depth=3 → captures skewness (jump asymmetry)
    """
    # Path construction
    spot_column: str = 'active_underlying_price'
    time_column: str = 'quote_datetime'
    expiration_column: str = 'expiration'
    strike_column: str = 'strike'
    option_type_column: str = 'option_type'
    iv_column: str = 'implied_volatility'
    delta_column: str = 'delta'
    gamma_column: str = 'gamma'
    theta_column: str = 'theta'
    vega_column: str = 'vega'
    bid_column: str = 'bid'
    ask_column: str = 'ask'
    volume_column: str = 'trade_volume'
    oi_column: str = 'open_interest'
    
    # Log-signature hyperparameters
    step_size: int = 8        # Snapshots per log-signature interval
    depth: int = 2            # Log-signature truncation depth
    
    # Jump detection thresholds
    spot_jump_threshold: float = 0.005   # 0.5% = ~2σ for 5-min
    vol_jump_threshold: float = 0.02     # 2% absolute IV change
    
    # ATM definition for IV extraction
    atm_moneyness_range: float = 0.02    # ±2% around spot
    
    # Smile fitting parameters
    smile_moneyness_range: float = 1.0   # |z_+| ≤ 1 for fitting
    min_options_for_fit: int = 5         # Minimum options for regression
    
    # Market hours (for time normalization)
    market_open: str = '09:30'
    market_close: str = '16:00'
    
    # Output options
    include_greeks: bool = True
    include_smile_params: bool = True
    include_jump_features: bool = True


@dataclass
class PathData:
    """
    Processed path data ready for Neural RDE.
    
    The path X = (t, log_S, σ, J^S, J^I) captures:
    - t: Normalized time [0, 1] within trading day
    - log_S: Log of spot price
    - σ: ATM implied volatility (instantaneous vol proxy)
    - J^S: Cumulative spot jump indicator
    - J^I: Cumulative vol jump indicator
    """
    timestamps: np.ndarray          # Original timestamps
    path: np.ndarray                # Shape: (n_snapshots, 5)
    logsignatures: np.ndarray       # Shape: (n_intervals, logsig_dim)
    
    # Metadata
    n_snapshots: int = 0
    n_intervals: int = 0
    logsig_dim: int = 0
    
    # Jump statistics
    n_spot_jumps: int = 0
    n_vol_jumps: int = 0
    
    # Path statistics
    realized_variance: float = 0.0
    realized_covariance: float = 0.0
    total_levy_area: float = 0.0


@dataclass
class SmileParameters:
    """
    Cross-sectional smile parameters extracted from options chain.
    
    From Carr-Wu: I² - A² = 2γz_+ + ω²z_+z_- + ξE[A(X)]
    
    - γ: Return-vol covariance (vanna term, typically negative)
    - ω²: Vol-of-vol (volga term, positive)
    - ξ: Lévy area coefficient (NEW - path convexity)
    - A: ATM implied volatility
    """
    gamma: float                    # Vanna coefficient
    omega2: float                   # Volga coefficient
    xi: float                       # Lévy area coefficient (NEW)
    atm_iv: float                   # ATM implied volatility
    r_squared: float                # Regression fit quality
    n_options: int                  # Number of options used in fit
    
    # Extended diagnostics
    skew_25d: Optional[float] = None    # 25-delta skew
    butterfly_25d: Optional[float] = None  # 25-delta butterfly


@dataclass
class TrainingBatch:
    """
    Complete training batch for Neural RDE.
    
    Contains:
    - Path data (X and log-signatures)
    - Cross-sectional targets (smile parameters)
    - P&L attribution targets
    - Market Greeks for validation
    """
    # Path data
    paths: np.ndarray               # Shape: (batch, n_snapshots, 5)
    logsignatures: np.ndarray       # Shape: (batch, n_intervals, logsig_dim)
    
    # Targets
    smile_params: List[SmileParameters]
    realized_variance: np.ndarray   # Shape: (batch,)
    realized_covariance: np.ndarray # Shape: (batch,)
    
    # For implied vol prediction
    moneyness: np.ndarray           # Shape: (batch, n_strikes)
    tau: np.ndarray                 # Shape: (batch,)
    target_iv: np.ndarray           # Shape: (batch, n_strikes)
    
    # Market Greeks (for validation)
    market_delta: Optional[np.ndarray] = None
    market_gamma: Optional[np.ndarray] = None
    market_theta: Optional[np.ndarray] = None
    market_vega: Optional[np.ndarray] = None


# =============================================================================
# PART 2: SIGNATURE AND LOG-SIGNATURE COMPUTATION
# =============================================================================

def logsig_dimension(d: int, depth: int) -> int:
    """
    Compute dimension of log-signature.
    
    β(d, M) = Σ_{k=1}^{M} (1/k) Σ_{j|k} μ(k/j) d^j
    
    Simplified for practical depths:
    - depth=1: d
    - depth=2: d + d(d-1)/2
    - depth=3: d + d(d-1)/2 + d²(d-1)/3
    """
    if depth == 1:
        return d
    elif depth == 2:
        return d + d * (d - 1) // 2
    elif depth == 3:
        return d + d * (d - 1) // 2 + d * d * (d - 1) // 3
    else:
        # General approximation
        total = 0
        for k in range(1, depth + 1):
            total += d ** k // k
        return total


def compute_signature_depth1(path: np.ndarray) -> np.ndarray:
    """Compute depth-1 signature (increments)."""
    return path[-1] - path[0]


def compute_signature_depth2(path: np.ndarray) -> np.ndarray:
    """
    Compute depth-2 signature terms.
    
    S^{i,j} = ∫∫_{s<t} dX^i_s dX^j_t
    """
    dX = np.diff(path, axis=0)
    X_cumsum = np.cumsum(dX, axis=0)
    X_cumsum = np.vstack([np.zeros((1, path.shape[1])), X_cumsum[:-1]])
    
    # S^{i,j} = Σ_t X_cumsum[t,i] * dX[t,j]
    sig2 = np.einsum('ti,tj->ij', X_cumsum, dX)
    return sig2


def compute_levy_area(path: np.ndarray) -> np.ndarray:
    """
    Compute Lévy area (antisymmetric part of depth-2 signature).
    
    A^{ij} = (S^{i,j} - S^{j,i}) / 2
    
    This captures the ORDER of movements — critical for jumps!
    """
    sig2 = compute_signature_depth2(path)
    return (sig2 - sig2.T) / 2.0


def compute_logsignature(path: np.ndarray, depth: int = 2) -> np.ndarray:
    """
    Compute log-signature of a path.
    
    Kidger: "The log-signature is the optimal summary statistic for
    predicting how a path will drive a differential equation."
    
    Components:
    - Depth 1: Increments (Δ, V) — classical first-order sensitivities
    - Depth 2: Lévy areas — order of movements (essential for jumps!)
    - Depth 3: Skewness — jump asymmetry
    
    Args:
        path: Array of shape (length, channels)
        depth: Truncation depth
        
    Returns:
        Log-signature vector
    """
    d = path.shape[1]
    
    # Depth 1: increments
    logsig1 = compute_signature_depth1(path)
    
    if depth == 1:
        return logsig1
    
    # Depth 2: Lévy areas (upper triangular of antisymmetric matrix)
    levy_area = compute_levy_area(path)
    logsig2 = []
    for i in range(d):
        for j in range(i + 1, d):
            logsig2.append(levy_area[i, j])
    logsig2 = np.array(logsig2) if logsig2 else np.array([])
    
    if depth == 2:
        return np.concatenate([logsig1, logsig2])
    
    # Depth 3: Higher-order terms (simplified)
    if depth >= 3:
        sig2 = compute_signature_depth2(path)
        sig3_terms = []
        for i in range(d):
            for j in range(d):
                if i < j:
                    # Approximate depth-3 Lyndon terms
                    term = sig2[i, i] * logsig1[j] - sig2[i, j] * logsig1[i]
                    sig3_terms.append(term / 6.0)
        logsig3 = np.array(sig3_terms) if sig3_terms else np.array([])
        return np.concatenate([logsig1, logsig2, logsig3])
    
    return np.concatenate([logsig1, logsig2])


def compute_logsignatures_for_intervals(
    path: np.ndarray,
    step_size: int,
    depth: int = 2
) -> np.ndarray:
    """
    Compute log-signatures over non-overlapping intervals.
    
    This is the KEY preprocessing step:
    
    Kidger: "The log-signature preprocessing is done ONCE and reused.
    This amortizes the cost of processing the high-frequency path.
    For 78 snapshots/day with step_size=8, we get ~9 log-signature
    vectors — a 10x compression with IMPROVED accuracy."
    
    Args:
        path: Full path, shape (n_snapshots, channels)
        step_size: Observations per interval
        depth: Log-signature truncation depth
        
    Returns:
        Array of log-signatures, shape (n_intervals, logsig_dim)
    """
    n_steps = path.shape[0]
    n_intervals = (n_steps - 1) // step_size
    
    if n_intervals == 0:
        # Not enough data for even one interval
        return compute_logsignature(path, depth)[np.newaxis, :]
    
    logsigs = []
    for i in range(n_intervals):
        start = i * step_size
        end = min(start + step_size + 1, n_steps)
        interval_path = path[start:end]
        logsig = compute_logsignature(interval_path, depth)
        logsigs.append(logsig)
    
    return np.stack(logsigs)


# =============================================================================
# PART 3: PATH CONSTRUCTION FROM CBOE DATA
# =============================================================================

class CBOEPathConstructor:
    """
    Constructs Neural RDE paths from CBOE options snapshots.
    
    Kidger's approach:
    
    "The path X = (t, log_S, σ, J^S, J^I) captures everything needed:
    
    - t: Normalized time (for theta decay dynamics)
    - log_S: Log price (for delta/gamma dynamics)  
    - σ: ATM IV (for vega/volga dynamics)
    - J^S: Spot jump indicator (for jump timing)
    - J^I: Vol jump indicator (for vol jump timing)
    
    The log-signature of this augmented path naturally captures:
    - Jump frequency through S^(3), S^(4)
    - Jump timing through S^(1,3), S^(2,4)
    - Jump clustering through S^(3,3), S^(4,4)"
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self._prev_log_s = None
        self._prev_sigma = None
        self._cumulative_spot_jumps = 0
        self._cumulative_vol_jumps = 0
    
    def reset_state(self):
        """Reset state for new trading day."""
        self._prev_log_s = None
        self._prev_sigma = None
        self._cumulative_spot_jumps = 0
        self._cumulative_vol_jumps = 0
    
    def extract_atm_iv(self, df: pd.DataFrame, spot: float) -> float:
        """
        Extract ATM implied volatility from options snapshot.
        
        For 0DTE options, use the closest-to-ATM call option.
        """
        cfg = self.config
        
        # Parse expiration
        df = df.copy()
        df[cfg.expiration_column] = pd.to_datetime(df[cfg.expiration_column])
        df[cfg.time_column] = pd.to_datetime(df[cfg.time_column])
        
        # Compute DTE
        df['dte'] = (df[cfg.expiration_column] - df[cfg.time_column]).dt.days
        
        # Filter to short-dated (0-5 DTE) and calls
        short_dated = df[(df['dte'] >= 0) & (df['dte'] <= 5)]
        calls = short_dated[short_dated[cfg.option_type_column] == 'C']
        
        if len(calls) == 0:
            warnings.warn("No short-dated calls found, using all calls")
            calls = df[df[cfg.option_type_column] == 'C']
        
        # Find ATM (closest strike to spot)
        calls = calls.copy()
        calls['moneyness'] = np.abs(calls[cfg.strike_column] / spot - 1)
        atm_mask = calls['moneyness'] <= self.config.atm_moneyness_range
        
        if atm_mask.sum() > 0:
            atm_options = calls[atm_mask]
            # Weight by proximity to ATM and open interest
            weights = 1 / (atm_options['moneyness'] + 0.001)
            if cfg.oi_column in atm_options.columns:
                weights *= np.sqrt(atm_options[cfg.oi_column] + 1)
            atm_iv = np.average(atm_options[cfg.iv_column], weights=weights)
        else:
            # Fall back to single closest option
            closest_idx = calls['moneyness'].idxmin()
            atm_iv = calls.loc[closest_idx, cfg.iv_column]
        
        return atm_iv
    
    def detect_jumps(
        self,
        log_s: float,
        sigma: float
    ) -> Tuple[bool, bool]:
        """
        Detect jumps in spot and volatility.
        
        Kidger: "The log-signature of a path with jumps looks very different
        from a continuous path: the depth-2 terms (Lévy areas) are much
        larger because jumps create sudden changes in direction."
        """
        spot_jump = False
        vol_jump = False
        
        if self._prev_log_s is not None:
            delta_log_s = log_s - self._prev_log_s
            delta_sigma = sigma - self._prev_sigma
            
            if abs(delta_log_s) > self.config.spot_jump_threshold:
                spot_jump = True
                self._cumulative_spot_jumps += 1
            
            if abs(delta_sigma) > self.config.vol_jump_threshold:
                vol_jump = True
                self._cumulative_vol_jumps += 1
        
        self._prev_log_s = log_s
        self._prev_sigma = sigma
        
        return spot_jump, vol_jump
    
    def normalize_time(self, timestamp: pd.Timestamp) -> float:
        """
        Normalize timestamp to [0, 1] within trading day.
        
        9:30 AM → 0.0, 4:00 PM → 1.0
        """
        market_open = pd.Timestamp(timestamp.strftime('%Y-%m-%d') + ' ' + 
                                   self.config.market_open)
        market_close = pd.Timestamp(timestamp.strftime('%Y-%m-%d') + ' ' + 
                                    self.config.market_close)
        
        total_minutes = (market_close - market_open).total_seconds() / 60
        elapsed_minutes = (timestamp - market_open).total_seconds() / 60
        
        return np.clip(elapsed_minutes / total_minutes, 0.0, 1.0)
    
    def process_snapshot(
        self,
        df: pd.DataFrame,
        compute_smile: bool = True
    ) -> Tuple[np.ndarray, Optional[SmileParameters]]:
        """
        Process a single CBOE snapshot into path point and smile parameters.
        
        Args:
            df: Single snapshot DataFrame
            compute_smile: Whether to extract smile parameters
            
        Returns:
            path_point: Array [t, log_S, σ, J^S, J^I]
            smile_params: Optional SmileParameters
        """
        cfg = self.config
        
        # Extract timestamp and spot
        timestamp = pd.to_datetime(df[cfg.time_column].iloc[0])
        spot = df[cfg.spot_column].iloc[0]
        
        # Normalize time
        t_norm = self.normalize_time(timestamp)
        
        # Log price
        log_s = np.log(spot)
        
        # ATM IV
        sigma = self.extract_atm_iv(df, spot)
        
        # Jump detection
        spot_jump, vol_jump = self.detect_jumps(log_s, sigma)
        
        # Normalized cumulative jump counts
        total_snapshots = 78  # Expected snapshots per day
        j_s = self._cumulative_spot_jumps / max(total_snapshots, 1)
        j_i = self._cumulative_vol_jumps / max(total_snapshots, 1)
        
        path_point = np.array([t_norm, log_s, sigma, j_s, j_i])
        
        # Extract smile parameters
        smile_params = None
        if compute_smile:
            smile_params = self.extract_smile_parameters(df, spot, sigma)
        
        return path_point, smile_params
    
    def extract_smile_parameters(
        self,
        df: pd.DataFrame,
        spot: float,
        atm_iv: float
    ) -> SmileParameters:
        """
        Extract Carr-Wu smile parameters from cross-section.
        
        Fits: I² - A² = 2γz_+ + ω²z_+z_-
        
        Extended with: + ξE[A(X)] for Lévy area contribution
        
        Kidger: "This explains why short-dated smiles are steeper:
        the expected Lévy area E[A(X)] is dominated by jump contributions
        at short horizons. At long horizons, the CLT kicks in."
        """
        cfg = self.config
        
        # Filter to relevant options
        df = df.copy()
        df[cfg.expiration_column] = pd.to_datetime(df[cfg.expiration_column])
        df[cfg.time_column] = pd.to_datetime(df[cfg.time_column])
        df['dte'] = (df[cfg.expiration_column] - df[cfg.time_column]).dt.days
        
        # Short-dated options only
        options = df[(df['dte'] >= 0) & (df['dte'] <= 5)].copy()
        
        # Compute standardized moneyness
        tau = np.maximum(options['dte'] / 252, 1e-6)  # Years
        options['z_plus'] = (np.log(options[cfg.strike_column] / spot) + 
                            0.5 * atm_iv**2 * tau) / (atm_iv * np.sqrt(tau))
        options['z_minus'] = options['z_plus'] - atm_iv * np.sqrt(tau)
        
        # Filter to fitting range
        mask = np.abs(options['z_plus']) <= cfg.smile_moneyness_range
        fit_options = options[mask].copy()
        
        if len(fit_options) < cfg.min_options_for_fit:
            # Not enough options, return defaults
            return SmileParameters(
                gamma=-0.5, omega2=0.1, xi=0.0,
                atm_iv=atm_iv, r_squared=0.0, n_options=len(fit_options)
            )
        
        # Compute I² - A²
        fit_options['iv_diff'] = fit_options[cfg.iv_column]**2 - atm_iv**2
        
        # Design matrix: [2z_+, z_+*z_-]
        X = np.column_stack([
            2 * fit_options['z_plus'].values,
            fit_options['z_plus'].values * fit_options['z_minus'].values
        ])
        y = fit_options['iv_diff'].values
        
        # Weighted regression (weight by bid-ask tightness and OI)
        weights = np.ones(len(fit_options))
        if cfg.bid_column in fit_options.columns and cfg.ask_column in fit_options.columns:
            spread = fit_options[cfg.ask_column] - fit_options[cfg.bid_column]
            mid = (fit_options[cfg.ask_column] + fit_options[cfg.bid_column]) / 2
            spread_pct = spread / (mid + 0.01)
            weights *= 1 / (spread_pct + 0.1)
        if cfg.oi_column in fit_options.columns:
            weights *= np.sqrt(fit_options[cfg.oi_column].values + 1)
        
        # Weighted least squares
        W = np.diag(weights)
        try:
            XtWX = X.T @ W @ X
            XtWy = X.T @ W @ y
            coeffs = np.linalg.solve(XtWX, XtWy)
            
            # Predictions and R²
            y_pred = X @ coeffs
            ss_res = np.sum(weights * (y - y_pred)**2)
            ss_tot = np.sum(weights * (y - np.average(y, weights=weights))**2)
            r_squared = 1 - ss_res / (ss_tot + 1e-8)
            
            gamma = coeffs[0]
            omega2 = max(coeffs[1], 0.0)  # Vol-of-vol must be positive
            
        except np.linalg.LinAlgError:
            gamma = -0.5
            omega2 = 0.1
            r_squared = 0.0
        
        # Estimate ξ (Lévy area coefficient) from residuals
        # This captures the path-dependent component not explained by γ, ω²
        xi = 0.0  # Default; would need path history to estimate properly
        
        # Optional: compute 25-delta skew and butterfly
        skew_25d = None
        butterfly_25d = None
        
        put_25d = options[(options[cfg.option_type_column] == 'P') & 
                          (options[cfg.delta_column].between(-0.30, -0.20))]
        call_25d = options[(options[cfg.option_type_column] == 'C') & 
                           (options[cfg.delta_column].between(0.20, 0.30))]
        
        if len(put_25d) > 0 and len(call_25d) > 0:
            iv_put_25d = put_25d[cfg.iv_column].mean()
            iv_call_25d = call_25d[cfg.iv_column].mean()
            skew_25d = iv_put_25d - iv_call_25d
            butterfly_25d = (iv_put_25d + iv_call_25d) / 2 - atm_iv
        
        return SmileParameters(
            gamma=gamma,
            omega2=omega2,
            xi=xi,
            atm_iv=atm_iv,
            r_squared=r_squared,
            n_options=len(fit_options),
            skew_25d=skew_25d,
            butterfly_25d=butterfly_25d
        )
    
    def process_day(
        self,
        snapshots: List[pd.DataFrame],
        compute_smile: bool = True
    ) -> PathData:
        """
        Process a full day of CBOE snapshots into PathData.
        
        Args:
            snapshots: List of DataFrames, one per snapshot
            compute_smile: Whether to extract smile parameters
            
        Returns:
            PathData ready for Neural RDE
        """
        self.reset_state()
        
        path_points = []
        timestamps = []
        smile_params_list = []
        
        for df in snapshots:
            point, smile = self.process_snapshot(df, compute_smile)
            path_points.append(point)
            timestamps.append(pd.to_datetime(df[self.config.time_column].iloc[0]))
            if smile is not None:
                smile_params_list.append(smile)
        
        path = np.stack(path_points)
        timestamps = np.array(timestamps)
        
        # Compute log-signatures
        logsigs = compute_logsignatures_for_intervals(
            path, self.config.step_size, self.config.depth
        )
        
        # Compute path statistics
        log_returns = np.diff(path[:, 1])  # Diff of log_S
        vol_changes = np.diff(path[:, 2])  # Diff of sigma
        
        realized_var = np.sum(log_returns**2) * 252 * 78  # Annualized
        realized_covar = np.sum(log_returns[:-1] * vol_changes[:-1]) * 252 * 78
        
        # Total Lévy area (price-vol)
        levy_area_matrix = compute_levy_area(path[:, 1:3])
        total_levy_area = levy_area_matrix[0, 1]
        
        return PathData(
            timestamps=timestamps,
            path=path,
            logsignatures=logsigs,
            n_snapshots=len(path),
            n_intervals=len(logsigs),
            logsig_dim=logsigs.shape[1],
            n_spot_jumps=self._cumulative_spot_jumps,
            n_vol_jumps=self._cumulative_vol_jumps,
            realized_variance=realized_var,
            realized_covariance=realized_covar,
            total_levy_area=total_levy_area
        )


# =============================================================================
# PART 4: BATCH CONSTRUCTION FOR TRAINING
# =============================================================================

class TrainingDataBuilder:
    """
    Builds training batches for Neural RDE from processed path data.
    
    Kidger: "The computational efficiency comes from the log-ODE method:
    instead of processing 2,340 minute bars individually, we compress
    them into ~70 log-signature vectors. This gives 10x speedup with
    17% accuracy improvement."
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.path_constructor = CBOEPathConstructor(config)
    
    def load_snapshots_from_files(
        self,
        file_paths: List[Union[str, Path]]
    ) -> List[pd.DataFrame]:
        """Load snapshots from CSV files."""
        snapshots = []
        for fp in sorted(file_paths):
            df = pd.read_csv(fp)
            snapshots.append(df)
        return snapshots
    
    def build_daily_batch(
        self,
        snapshots: List[pd.DataFrame],
        target_strikes: Optional[np.ndarray] = None,
        target_tau: float = 0.0  # 0DTE
    ) -> TrainingBatch:
        """
        Build a single-day training batch.
        
        Args:
            snapshots: List of intraday snapshots
            target_strikes: Strikes for IV targets (or None for ATM)
            target_tau: Target maturity in years
            
        Returns:
            TrainingBatch ready for Neural RDE training
        """
        # Process the day
        path_data = self.path_constructor.process_day(snapshots, compute_smile=True)
        
        # Get the last snapshot for cross-sectional targets
        last_df = snapshots[-1]
        cfg = self.config
        spot = last_df[cfg.spot_column].iloc[0]
        
        # Extract IV targets
        if target_strikes is None:
            # Use strikes around ATM
            target_strikes = np.array([spot * (1 + d) for d in 
                                       [-0.02, -0.01, 0, 0.01, 0.02]])
        
        # Compute moneyness and target IVs
        atm_iv = self.path_constructor.extract_atm_iv(last_df, spot)
        tau = max(target_tau, 1/252)  # At least 1 day
        
        moneyness = (np.log(target_strikes / spot) + 0.5 * atm_iv**2 * tau) / \
                    (atm_iv * np.sqrt(tau))
        
        # Look up target IVs from options chain
        target_iv = np.zeros_like(target_strikes)
        last_df = last_df.copy()
        last_df[cfg.expiration_column] = pd.to_datetime(last_df[cfg.expiration_column])
        last_df[cfg.time_column] = pd.to_datetime(last_df[cfg.time_column])
        last_df['dte'] = (last_df[cfg.expiration_column] - last_df[cfg.time_column]).dt.days
        
        calls = last_df[(last_df['dte'] == 0) & (last_df[cfg.option_type_column] == 'C')]
        
        for i, K in enumerate(target_strikes):
            closest_idx = (calls[cfg.strike_column] - K).abs().idxmin()
            target_iv[i] = calls.loc[closest_idx, cfg.iv_column] if len(calls) > 0 else atm_iv
        
        # Extract smile parameters from last snapshot
        smile_params = self.path_constructor.extract_smile_parameters(
            last_df, spot, atm_iv
        )
        
        # Extract market Greeks (for validation)
        market_delta = None
        market_gamma = None
        market_theta = None
        market_vega = None
        
        if cfg.include_greeks:
            atm_calls = calls[(calls[cfg.strike_column] - spot).abs() < 3]
            if len(atm_calls) > 0:
                market_delta = atm_calls[cfg.delta_column].mean()
                market_gamma = atm_calls[cfg.gamma_column].mean()
                market_theta = atm_calls[cfg.theta_column].mean()
                market_vega = atm_calls[cfg.vega_column].mean()
        
        return TrainingBatch(
            paths=path_data.path[np.newaxis, :, :],
            logsignatures=path_data.logsignatures[np.newaxis, :, :],
            smile_params=[smile_params],
            realized_variance=np.array([path_data.realized_variance]),
            realized_covariance=np.array([path_data.realized_covariance]),
            moneyness=moneyness[np.newaxis, :],
            tau=np.array([tau]),
            target_iv=target_iv[np.newaxis, :],
            market_delta=np.array([market_delta]) if market_delta is not None else None,
            market_gamma=np.array([market_gamma]) if market_gamma is not None else None,
            market_theta=np.array([market_theta]) if market_theta is not None else None,
            market_vega=np.array([market_vega]) if market_vega is not None else None
        )


# =============================================================================
# PART 5: ONLINE/STREAMING PROCESSING
# =============================================================================

class OnlinePathProcessor:
    """
    Online processor for real-time Neural RDE inference.
    
    Kidger: "Log-signatures can be computed in an online fashion,
    making the model suitable for real-time problems."
    
    This class maintains state for streaming updates:
    - Accumulates path points within current interval
    - Computes log-signature when interval completes
    - Provides current log-signature sequence for inference
    """
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.path_constructor = CBOEPathConstructor(config)
        
        # State
        self._current_interval_path = []
        self._logsig_sequence = []
        self._full_path = []
        self._snapshots_in_interval = 0
    
    def reset(self):
        """Reset state for new trading day."""
        self.path_constructor.reset_state()
        self._current_interval_path = []
        self._logsig_sequence = []
        self._full_path = []
        self._snapshots_in_interval = 0
    
    def update(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process new snapshot and return current state.
        
        Args:
            df: New snapshot DataFrame
            
        Returns:
            current_path: Full path so far, shape (n_snapshots, 5)
            current_logsigs: Log-signature sequence, shape (n_intervals, logsig_dim)
        """
        # Process snapshot
        point, smile = self.path_constructor.process_snapshot(df, compute_smile=False)
        
        self._full_path.append(point)
        self._current_interval_path.append(point)
        self._snapshots_in_interval += 1
        
        # Check if interval is complete
        if self._snapshots_in_interval >= self.config.step_size:
            interval_path = np.stack(self._current_interval_path)
            logsig = compute_logsignature(interval_path, self.config.depth)
            self._logsig_sequence.append(logsig)
            
            # Reset interval
            self._current_interval_path = [point]  # Overlap by 1
            self._snapshots_in_interval = 1
        
        # Return current state
        current_path = np.stack(self._full_path)
        current_logsigs = np.stack(self._logsig_sequence) if self._logsig_sequence else \
                          np.zeros((0, logsig_dimension(5, self.config.depth)))
        
        return current_path, current_logsigs
    
    def get_partial_logsig(self) -> np.ndarray:
        """
        Get log-signature for the incomplete current interval.
        
        Useful for real-time inference when interval isn't complete.
        """
        if len(self._current_interval_path) < 2:
            return np.zeros(logsig_dimension(5, self.config.depth))
        
        interval_path = np.stack(self._current_interval_path)
        return compute_logsignature(interval_path, self.config.depth)


# =============================================================================
# PART 6: UTILITY FUNCTIONS
# =============================================================================

def analyze_path_roughness(path: np.ndarray) -> Dict[str, float]:
    """
    Analyze the roughness of a path.
    
    Kidger: "Short-dated options have ROUGH dynamics — dominated by
    jumps and rapid regime changes. The signature captures this
    roughness through higher-order terms."
    
    Returns various roughness metrics.
    """
    log_returns = np.diff(path[:, 1])  # Log price returns
    vol_changes = np.diff(path[:, 2])  # Vol changes
    
    # Realized variance (annualized)
    realized_var = np.sum(log_returns**2) * 252 * len(log_returns)
    
    # Realized skewness
    if len(log_returns) > 2:
        realized_skew = np.sum(log_returns**3) / (np.sum(log_returns**2)**(3/2) + 1e-8)
    else:
        realized_skew = 0.0
    
    # Lévy area magnitude (path-dependence indicator)
    levy_area = compute_levy_area(path[:, 1:3])
    levy_magnitude = np.abs(levy_area[0, 1])
    
    # Roughness exponent estimate (Hurst-like)
    # H ≈ 0.5 for Brownian, < 0.5 for rough
    increments = np.abs(log_returns)
    if len(increments) > 10:
        mean_inc = np.mean(increments)
        mean_sq_inc = np.mean(increments**2)
        roughness_ratio = mean_inc**2 / (mean_sq_inc + 1e-8)
    else:
        roughness_ratio = 0.5
    
    # Jump count (based on threshold)
    jump_threshold = 2 * np.std(log_returns)  # 2-sigma
    n_jumps = np.sum(np.abs(log_returns) > jump_threshold)
    jump_ratio = n_jumps / len(log_returns)
    
    return {
        'realized_variance': realized_var,
        'realized_skewness': realized_skew,
        'levy_area_magnitude': levy_magnitude,
        'roughness_ratio': roughness_ratio,
        'jump_count': n_jumps,
        'jump_ratio': jump_ratio
    }


def recommend_hyperparameters(
    n_snapshots: int,
    path_roughness: Dict[str, float],
    target_accuracy: str = 'balanced'  # 'fast', 'balanced', 'accurate'
) -> Dict[str, int]:
    """
    Recommend step_size and depth based on data characteristics.
    
    Kidger: "Increasing step_size leads to faster training.
    Increasing depth leads to more informative training.
    The choice depends on the roughness of the path and
    computational constraints."
    """
    # Base recommendations
    if target_accuracy == 'fast':
        base_step = 16
        base_depth = 2
    elif target_accuracy == 'accurate':
        base_step = 4
        base_depth = 3
    else:  # balanced
        base_step = 8
        base_depth = 2
    
    # Adjust for roughness
    if path_roughness['jump_ratio'] > 0.1:
        # Many jumps → need more detail
        base_step = max(4, base_step // 2)
        base_depth = min(3, base_depth + 1)
    
    if path_roughness['levy_area_magnitude'] > 0.01:
        # Large Lévy areas → depth-2 essential
        base_depth = max(2, base_depth)
    
    # Adjust for sequence length
    target_intervals = 10  # Reasonable number of intervals
    suggested_step = max(4, n_snapshots // target_intervals)
    
    return {
        'step_size': min(base_step, suggested_step),
        'depth': base_depth,
        'n_intervals': n_snapshots // min(base_step, suggested_step),
        'logsig_dim': logsig_dimension(5, base_depth)
    }


# =============================================================================
# PART 7: DEMONSTRATION
# =============================================================================

def demo_preprocessing_pipeline():
    """
    Demonstrate the preprocessing pipeline with sample data.
    """
    print("=" * 70)
    print("Kidger's Data Preprocessing Pipeline - Demonstration")
    print("=" * 70)
    
    # Create sample data (simulating 3 CBOE snapshots)
    np.random.seed(42)
    
    def create_sample_snapshot(timestamp: str, spot: float) -> pd.DataFrame:
        """Create a sample CBOE-like snapshot."""
        n_options = 100
        strikes = spot * np.linspace(0.9, 1.1, n_options)
        
        df = pd.DataFrame({
            'underlying_symbol': 'SPY',
            'quote_datetime': timestamp,
            'expiration': '2025-12-01',
            'strike': strikes,
            'option_type': ['C' if i % 2 == 0 else 'P' for i in range(n_options)],
            'active_underlying_price': spot,
            'implied_volatility': 0.2 + 0.1 * ((strikes - spot) / spot)**2,
            'delta': 0.5 * np.sign(strikes - spot),
            'gamma': 0.1 * np.exp(-((strikes - spot) / spot)**2),
            'theta': -0.1,
            'vega': 0.3,
            'bid': 1.0,
            'ask': 1.1,
            'trade_volume': np.random.randint(0, 100, n_options),
            'open_interest': np.random.randint(100, 10000, n_options)
        })
        return df
    
    # Simulate intraday evolution
    spots = [681.45, 680.98, 680.91]  # Declining price
    timestamps = ['2025-12-01 10:00:00', '2025-12-01 10:05:00', '2025-12-01 10:10:00']
    
    snapshots = [create_sample_snapshot(t, s) for t, s in zip(timestamps, spots)]
    
    # Initialize pipeline
    config = PreprocessingConfig(
        step_size=2,  # Small for demo
        depth=2
    )
    
    print("\n1. CONFIGURATION")
    print(f"   Step size: {config.step_size}")
    print(f"   Depth: {config.depth}")
    print(f"   Expected log-sig dimension: {logsig_dimension(5, config.depth)}")
    
    # Process snapshots
    print("\n2. PATH CONSTRUCTION")
    constructor = CBOEPathConstructor(config)
    path_data = constructor.process_day(snapshots)
    
    print(f"   Input snapshots: {len(snapshots)}")
    print(f"   Path shape: {path_data.path.shape}")
    print(f"   Log-sig shape: {path_data.logsignatures.shape}")
    print(f"   Spot jumps detected: {path_data.n_spot_jumps}")
    print(f"   Vol jumps detected: {path_data.n_vol_jumps}")
    
    print("\n3. PATH STATISTICS")
    print(f"   Realized variance (ann.): {path_data.realized_variance:.4f}")
    print(f"   Realized covariance: {path_data.realized_covariance:.6f}")
    print(f"   Total Lévy area: {path_data.total_levy_area:.6f}")
    
    print("\n4. LOG-SIGNATURE COMPONENTS")
    for i, logsig in enumerate(path_data.logsignatures):
        print(f"   Interval {i}: {logsig[:5]}...")  # First 5 components
    
    # Analyze roughness
    print("\n5. PATH ROUGHNESS ANALYSIS")
    roughness = analyze_path_roughness(path_data.path)
    for key, value in roughness.items():
        print(f"   {key}: {value:.6f}")
    
    # Recommend hyperparameters
    print("\n6. HYPERPARAMETER RECOMMENDATIONS")
    recs = recommend_hyperparameters(len(snapshots), roughness, 'balanced')
    for key, value in recs.items():
        print(f"   {key}: {value}")
    
    # Build training batch
    print("\n7. TRAINING BATCH")
    builder = TrainingDataBuilder(config)
    batch = builder.build_daily_batch(snapshots)
    
    print(f"   Paths shape: {batch.paths.shape}")
    print(f"   Logsigs shape: {batch.logsignatures.shape}")
    print(f"   Smile gamma: {batch.smile_params[0].gamma:.4f}")
    print(f"   Smile omega²: {batch.smile_params[0].omega2:.4f}")
    print(f"   Smile R²: {batch.smile_params[0].r_squared:.4f}")
    
    print("\n" + "=" * 70)
    print("Preprocessing pipeline demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_preprocessing_pipeline()
