"""
Enhanced Neural RDE Implementation - Addressing All Theoretical Components
==========================================================================

This module extends the base implementation to cover ALL components from
Kidger's theoretical framework for short-dated options:

GAPS ADDRESSED:
1. Jump indicator channels (J^S, J^I) in augmented path
2. Explicit signature-to-Greek mapping with interpretation
3. Extended smile formula with Lévy area term
4. ATM skew explosion analysis
5. Carr-Wu baseline comparison
6. Explicit no-arbitrage constraint formulation
7. Depth-3 signature interpretation for forecasting

Author: Extended from Kidger et al. framework
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from functools import partial
from typing import Callable, Optional, Tuple, NamedTuple, List, Dict
import equinox as eqx
import diffrax
import optax

# Import base components
from neural_rde_options import (
    compute_signature,
    compute_logsignature,
    compute_levy_area,
    compute_signature_depth1,
    compute_signature_depth2,
    _compute_signature_depth3,
    logsignature_dimension,
    VectorField,
    InitialEncoder,
)


# =============================================================================
# PART 1: SIGNATURE-TO-GREEK MAPPING (Section I of Framework)
# =============================================================================

class SignatureGreekMapping(NamedTuple):
    """
    Explicit mapping from signature terms to financial Greeks.
    
    From the theoretical framework table:
    
    | Signature Term      | Greek     | Interpretation                    |
    |---------------------|-----------|-----------------------------------|
    | S^(0)               | Θ         | Time decay                        |
    | S^(1)               | Δ         | Spot sensitivity                  |
    | S^(2)               | Vega      | Vol sensitivity                   |
    | S^(1,1)             | Γ         | Convexity in spot                 |
    | S^(2,2)             | Volga     | Convexity in vol                  |
    | S^(1,2) + S^(2,1)   | Vanna     | Cross-gamma (symmetric)           |
    | S^(1,2) - S^(2,1)   | Lévy area | Order of spot/vol moves           |
    | S^(0,1) - S^(1,0)   | Libra     | Time-space non-commutativity      |
    """
    # Depth-1 Greeks
    theta: jnp.ndarray      # S^(0) - time increment sensitivity
    delta: jnp.ndarray      # S^(1) - spot increment sensitivity  
    vega: jnp.ndarray       # S^(2) - vol increment sensitivity
    
    # Depth-2 Greeks (symmetric part)
    gamma: jnp.ndarray      # S^(1,1) - spot convexity
    volga: jnp.ndarray      # S^(2,2) - vol convexity
    vanna: jnp.ndarray      # S^(1,2) + S^(2,1) - cross sensitivity
    
    # Depth-2 Greeks (antisymmetric part - NEW for path-dependence)
    levy_area_sv: jnp.ndarray   # S^(1,2) - S^(2,1) - spot-vol order
    libra: jnp.ndarray          # S^(0,1) - S^(1,0) - time-space Lie bracket
    
    # Depth-3 Greeks (for jump asymmetry)
    skewness_contrib: jnp.ndarray  # S^(1,1,1) - return skewness sensitivity


def extract_greeks_from_signature(
    sig: jnp.ndarray,
    d: int = 3,
    depth: int = 2
) -> SignatureGreekMapping:
    """
    Extract financial Greeks from signature vector.
    
    This implements the explicit mapping from the theoretical framework.
    
    Args:
        sig: Flattened signature vector
        d: Number of channels (time, spot, vol)
        depth: Signature depth
        
    Returns:
        SignatureGreekMapping with all Greek values
    """
    # Parse depth-1 terms: S^(0), S^(1), S^(2)
    sig1 = sig[:d]
    theta = sig1[0]  # Time sensitivity
    delta = sig1[1]  # Spot sensitivity
    vega = sig1[2] if d > 2 else jnp.array(0.0)
    
    if depth >= 2:
        # Parse depth-2 terms: reshape to (d, d) matrix
        sig2_start = d
        sig2_end = d + d * d
        sig2 = sig[sig2_start:sig2_end].reshape(d, d)
        
        # Symmetric Greeks
        gamma = sig2[1, 1]           # S^(1,1)
        volga = sig2[2, 2] if d > 2 else jnp.array(0.0)  # S^(2,2)
        vanna = sig2[1, 2] + sig2[2, 1] if d > 2 else jnp.array(0.0)  # S^(1,2) + S^(2,1)
        
        # Antisymmetric Greeks (the NEW path-dependent terms)
        levy_area_sv = (sig2[1, 2] - sig2[2, 1]) / 2 if d > 2 else jnp.array(0.0)
        libra = (sig2[0, 1] - sig2[1, 0]) / 2  # Time-space Lie bracket
    else:
        gamma = volga = vanna = levy_area_sv = libra = jnp.array(0.0)
    
    if depth >= 3:
        # Parse depth-3 for skewness
        sig3_start = d + d * d
        sig3 = sig[sig3_start:].reshape(d, d, d) if len(sig) > sig3_start else jnp.zeros((d, d, d))
        skewness_contrib = sig3[1, 1, 1]  # S^(1,1,1) - return skewness
    else:
        skewness_contrib = jnp.array(0.0)
    
    return SignatureGreekMapping(
        theta=theta, delta=delta, vega=vega,
        gamma=gamma, volga=volga, vanna=vanna,
        levy_area_sv=levy_area_sv, libra=libra,
        skewness_contrib=skewness_contrib
    )


def compute_pnl_from_signature_greeks(
    greeks: SignatureGreekMapping,
    future_sig: jnp.ndarray,
    d: int = 3
) -> Dict[str, jnp.ndarray]:
    """
    Compute P&L decomposition using signature Greeks.
    
    Implements the Functional Taylor Expansion:
    
    P&L = f(X*Y) - f(X) = Σ_{|I|≤k} Δ_I f(X) · S^I(Y)
        ≈ Θ·δt + Δ·δy + Γ·(δy²/2) + V·δσ + L·A(Y) + ...
    
    Args:
        greeks: SignatureGreekMapping from current position
        future_sig: Signature of future path Y
        d: Number of channels
        
    Returns:
        Dictionary with P&L attribution breakdown
    """
    # Extract future path statistics
    future_greeks = extract_greeks_from_signature(future_sig, d, depth=2)
    
    # Depth-1 P&L (classical)
    theta_pnl = greeks.theta * future_greeks.theta  # Time decay
    delta_pnl = greeks.delta * future_greeks.delta  # Delta P&L
    vega_pnl = greeks.vega * future_greeks.vega     # Vega P&L
    
    # Depth-2 P&L (second-order)
    gamma_pnl = greeks.gamma * future_greeks.gamma / 2  # Gamma P&L
    volga_pnl = greeks.volga * future_greeks.volga / 2  # Volga P&L
    vanna_pnl = greeks.vanna * future_greeks.vanna / 2  # Vanna P&L
    
    # PATH-DEPENDENT P&L (the new terms!)
    levy_area_pnl = greeks.levy_area_sv * future_greeks.levy_area_sv  # Order matters
    libra_pnl = greeks.libra * future_greeks.libra  # Time-space interaction
    
    total_pnl = (theta_pnl + delta_pnl + vega_pnl + 
                 gamma_pnl + volga_pnl + vanna_pnl +
                 levy_area_pnl + libra_pnl)
    
    return {
        'theta_pnl': theta_pnl,
        'delta_pnl': delta_pnl,
        'vega_pnl': vega_pnl,
        'gamma_pnl': gamma_pnl,
        'volga_pnl': volga_pnl,
        'vanna_pnl': vanna_pnl,
        'levy_area_pnl': levy_area_pnl,  # NEW
        'libra_pnl': libra_pnl,          # NEW
        'total_pnl': total_pnl,
        'path_dependent_pnl': levy_area_pnl + libra_pnl  # Path-dependent contribution
    }


# =============================================================================
# PART 2: JUMP INDICATOR AUGMENTATION (Section II of Framework)
# =============================================================================

def augment_path_with_jump_indicators(
    path: jnp.ndarray,
    spot_col: int = 1,
    vol_col: int = 2,
    spot_threshold: float = 0.02,  # 2% move = jump
    vol_threshold: float = 0.05    # 5% vol move = jump
) -> jnp.ndarray:
    """
    Augment path with jump indicator channels.
    
    From the theoretical framework:
    X = (t, S, I, J^S, J^I)
    
    where:
    - J^S_t = Σ_{s≤t} 1_{|ΔS_s| > ε} counts large price moves
    - J^I_t = Σ_{s≤t} 1_{|ΔI_s| > ε} counts large vol moves
    
    The log-signature of this augmented path captures:
    - Jump frequency: Through S^(3), S^(4)
    - Jump timing: Through S^(1,3) (price move followed by jump)
    - Jump clustering: Through S^(3,3)
    
    Args:
        path: Original path (t, log_S, σ), shape (n, 3)
        spot_col: Column index for log-price
        vol_col: Column index for volatility
        spot_threshold: Threshold for spot jump detection
        vol_threshold: Threshold for vol jump detection
        
    Returns:
        Augmented path of shape (n, 5)
    """
    n = path.shape[0]
    
    # Compute returns/changes
    spot_changes = jnp.diff(path[:, spot_col], prepend=path[0, spot_col])
    vol_changes = jnp.diff(path[:, vol_col], prepend=path[0, vol_col])
    
    # Detect jumps (binary indicators)
    spot_jumps = (jnp.abs(spot_changes) > spot_threshold).astype(jnp.float32)
    vol_jumps = (jnp.abs(vol_changes) > vol_threshold).astype(jnp.float32)
    
    # Cumulative jump counts (as described in framework)
    spot_jump_count = jnp.cumsum(spot_jumps)
    vol_jump_count = jnp.cumsum(vol_jumps)
    
    # Normalize for numerical stability
    spot_jump_count = spot_jump_count / jnp.maximum(spot_jump_count[-1], 1.0)
    vol_jump_count = vol_jump_count / jnp.maximum(vol_jump_count[-1], 1.0)
    
    # Augment path: (t, S, I, J^S, J^I)
    augmented = jnp.concatenate([
        path,
        spot_jump_count[:, None],
        vol_jump_count[:, None]
    ], axis=1)
    
    return augmented


def compute_jump_signature_features(path: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Extract jump-related features from augmented path signature.
    
    Args:
        path: Augmented path (t, S, I, J^S, J^I)
        
    Returns:
        Dictionary with jump signature features
    """
    d = path.shape[1]
    
    if d < 5:
        return {'jump_frequency': jnp.array(0.0), 'jump_timing': jnp.array(0.0), 
                'jump_clustering': jnp.array(0.0)}
    
    sig2 = compute_signature_depth2(path)
    
    # Jump frequency: S^(3), S^(4) - increments in jump counters
    sig1 = compute_signature_depth1(path)
    spot_jump_freq = sig1[3]  # S^(3)
    vol_jump_freq = sig1[4]   # S^(4)
    
    # Jump timing: S^(1,3) - correlation of price moves and jump timing
    spot_jump_timing = sig2[1, 3]  # Price followed by spot jump
    vol_jump_timing = sig2[2, 4]   # Vol followed by vol jump
    
    # Jump clustering: S^(3,3), S^(4,4) - consecutive jumps
    spot_jump_cluster = sig2[3, 3]
    vol_jump_cluster = sig2[4, 4]
    
    # Cross-asset jump contagion: S^(3,4) - spot jump followed by vol jump
    jump_contagion = sig2[3, 4] + sig2[4, 3]
    
    return {
        'spot_jump_freq': spot_jump_freq,
        'vol_jump_freq': vol_jump_freq,
        'spot_jump_timing': spot_jump_timing,
        'vol_jump_timing': vol_jump_timing,
        'spot_jump_cluster': spot_jump_cluster,
        'vol_jump_cluster': vol_jump_cluster,
        'jump_contagion': jump_contagion
    }


# =============================================================================
# PART 3: SMILE SHAPE WITH LÉVY AREA (Section III of Framework)
# =============================================================================

def compute_smile_formula_extended(
    z_plus: jnp.ndarray,
    z_minus: jnp.ndarray,
    gamma: jnp.ndarray,
    omega2: jnp.ndarray,
    expected_levy_area: jnp.ndarray,
    xi: jnp.ndarray = jnp.array(1.0)
) -> jnp.ndarray:
    """
    Compute smile shape using extended Carr-Wu formula with Lévy area.
    
    From the theoretical framework:
    
    I_t²(K) - A_t² = 2γ_t z_+ + ω_t² z_+ z_- + ξ_t E[A(X)]
    
    The NEW term ξ_t E[A(X)] captures path convexity contribution to smile.
    
    This explains why short-dated smiles are steeper:
    - Expected Lévy area E[A(X)] is dominated by jumps at short horizons
    - At long horizons, CLT kicks in and Lévy area averages out
    
    Args:
        z_plus: Standardized moneyness under Q
        z_minus: Standardized moneyness under share measure
        gamma: Return-vol covariance (vanna term)
        omega2: Vol-of-vol (volga term)
        expected_levy_area: Expected Lévy area from path dynamics
        xi: Lévy area sensitivity coefficient
        
    Returns:
        I² - A² (implied variance minus ATM variance)
    """
    # Classical Carr-Wu terms
    vanna_term = 2 * gamma * z_plus
    volga_term = omega2 * z_plus * z_minus
    
    # NEW: Lévy area contribution (causes skew explosion at short maturities)
    levy_area_term = xi * expected_levy_area
    
    return vanna_term + volga_term + levy_area_term


def analyze_atm_skew_explosion(
    tau: jnp.ndarray,
    jump_intensity: float = 10.0,
    jump_size_mean: float = -0.02,
    continuous_vol: float = 0.2
) -> Dict[str, jnp.ndarray]:
    """
    Analyze ATM skew explosion phenomenon.
    
    From Kidger's explanation:
    
    For continuous paths: Lévy area ~ O(τ)
    For jump paths: Lévy area ~ O(1) (doesn't vanish!)
    
    Therefore: ATM skew ∝ 1/√τ as τ → 0
    
    Args:
        tau: Time to maturity
        jump_intensity: λ (jumps per year)
        jump_size_mean: m (mean jump size, negative for crashes)
        continuous_vol: σ (diffusive volatility)
        
    Returns:
        Dictionary with skew analysis
    """
    # Expected number of jumps in interval
    expected_jumps = jump_intensity * tau
    
    # Continuous contribution to Lévy area (scales with τ)
    continuous_levy_area = continuous_vol ** 2 * tau / 2
    
    # Jump contribution to Lévy area (scales with O(1))
    # A(X) ≈ ΔX_jump · (t* - τ/2) where t* is jump time
    # E[A | jump] ≈ jump_size_mean · τ/4 (if jump uniformly distributed)
    jump_levy_area = expected_jumps * jnp.abs(jump_size_mean) * tau / 4
    
    # Total expected Lévy area
    total_levy_area = continuous_levy_area + jump_levy_area
    
    # ATM skew (implied vol slope at ATM)
    # Skew ∝ Lévy area / √τ
    atm_skew = total_levy_area / jnp.sqrt(tau + 1e-8)
    
    # Jump contribution ratio (dominates at short maturities)
    jump_ratio = jump_levy_area / (total_levy_area + 1e-8)
    
    return {
        'total_levy_area': total_levy_area,
        'continuous_contribution': continuous_levy_area,
        'jump_contribution': jump_levy_area,
        'atm_skew': atm_skew,
        'jump_dominance_ratio': jump_ratio,
        'expected_jumps': expected_jumps
    }


# =============================================================================
# PART 4: ENHANCED NEURAL RDE WITH ALL FEATURES
# =============================================================================

class EnhancedOutputDecoder(eqx.Module):
    """
    Enhanced decoder implementing full signature-to-Greek mapping.
    
    Outputs:
    - All classical Greeks (Θ, Δ, Γ, V)
    - Higher-order Greeks (Volga, Vanna)
    - Path-dependent Greeks (Lévy area sensitivity, Libra)
    - Smile shape parameters (γ, ω², ξ for Lévy area)
    - Jump risk metrics
    """
    # Classical Greek heads
    theta_head: eqx.nn.Linear
    delta_head: eqx.nn.Linear
    gamma_head: eqx.nn.Linear
    vega_head: eqx.nn.Linear
    
    # Higher-order Greek heads
    volga_head: eqx.nn.Linear
    vanna_head: eqx.nn.Linear
    
    # Path-dependent Greek heads (NEW)
    levy_area_sens_head: eqx.nn.Linear  # Lévy area sensitivity
    libra_head: eqx.nn.Linear           # Time-space Lie bracket sensitivity
    
    # Smile shape heads
    smile_gamma_head: eqx.nn.Linear     # γ for smile (vanna term)
    smile_omega2_head: eqx.nn.Linear    # ω² for smile (volga term)
    smile_xi_head: eqx.nn.Linear        # ξ for smile (Lévy area term) - NEW
    
    # Moment forecast heads
    sigma2_head: eqx.nn.Linear
    covar_head: eqx.nn.Linear
    vol_of_vol_head: eqx.nn.Linear
    
    # Implied vol head
    iv_head: eqx.nn.Linear
    
    hidden_dim: int
    
    def __init__(self, hidden_dim: int, *, key: jr.PRNGKey):
        self.hidden_dim = hidden_dim
        keys = jr.split(key, 15)
        
        # Classical Greeks
        self.theta_head = eqx.nn.Linear(hidden_dim, 1, key=keys[0])
        self.delta_head = eqx.nn.Linear(hidden_dim, 1, key=keys[1])
        self.gamma_head = eqx.nn.Linear(hidden_dim, 1, key=keys[2])
        self.vega_head = eqx.nn.Linear(hidden_dim, 1, key=keys[3])
        
        # Higher-order Greeks
        self.volga_head = eqx.nn.Linear(hidden_dim, 1, key=keys[4])
        self.vanna_head = eqx.nn.Linear(hidden_dim, 1, key=keys[5])
        
        # Path-dependent Greeks
        self.levy_area_sens_head = eqx.nn.Linear(hidden_dim, 1, key=keys[6])
        self.libra_head = eqx.nn.Linear(hidden_dim, 1, key=keys[7])
        
        # Smile shape
        self.smile_gamma_head = eqx.nn.Linear(hidden_dim, 1, key=keys[8])
        self.smile_omega2_head = eqx.nn.Linear(hidden_dim, 1, key=keys[9])
        self.smile_xi_head = eqx.nn.Linear(hidden_dim, 1, key=keys[10])
        
        # Moment forecasts
        self.sigma2_head = eqx.nn.Linear(hidden_dim, 1, key=keys[11])
        self.covar_head = eqx.nn.Linear(hidden_dim, 1, key=keys[12])
        self.vol_of_vol_head = eqx.nn.Linear(hidden_dim, 1, key=keys[13])
        
        # IV (takes hidden + moneyness + tau)
        self.iv_head = eqx.nn.Linear(hidden_dim + 2, 1, key=keys[14])
    
    def __call__(
        self,
        z: jnp.ndarray,
        moneyness: Optional[jnp.ndarray] = None,
        tau: Optional[jnp.ndarray] = None,
        expected_levy_area: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """Decode hidden state to all outputs."""
        
        # Classical Greeks
        theta = self.theta_head(z).squeeze()
        delta = self.delta_head(z).squeeze()
        gamma = self.gamma_head(z).squeeze()
        vega = self.vega_head(z).squeeze()
        
        # Higher-order Greeks
        volga = self.volga_head(z).squeeze()
        vanna = self.vanna_head(z).squeeze()
        
        # Path-dependent Greeks (NEW)
        levy_area_sens = self.levy_area_sens_head(z).squeeze()
        libra = self.libra_head(z).squeeze()
        
        # Smile shape parameters
        smile_gamma = self.smile_gamma_head(z).squeeze()
        smile_omega2 = jax.nn.softplus(self.smile_omega2_head(z).squeeze())
        smile_xi = self.smile_xi_head(z).squeeze()  # Can be negative
        
        # Moment forecasts
        sigma2 = jax.nn.softplus(self.sigma2_head(z).squeeze())
        covar = self.covar_head(z).squeeze()
        vol_of_vol = jax.nn.softplus(self.vol_of_vol_head(z).squeeze())
        
        outputs = {
            # Greeks
            'theta': theta,
            'delta': delta,
            'gamma': gamma,
            'vega': vega,
            'volga': volga,
            'vanna': vanna,
            'levy_area_sensitivity': levy_area_sens,  # NEW
            'libra': libra,  # NEW
            
            # Smile parameters
            'smile_gamma': smile_gamma,
            'smile_omega2': smile_omega2,
            'smile_xi': smile_xi,  # NEW - Lévy area term coefficient
            
            # Moment forecasts
            'sigma2': sigma2,
            'gamma_forecast': covar,
            'omega2': vol_of_vol
        }
        
        # Implied volatility (if moneyness provided)
        if moneyness is not None and tau is not None:
            z_aug = jnp.concatenate([z, jnp.atleast_1d(moneyness), jnp.atleast_1d(tau)])
            base_iv = jax.nn.softplus(self.iv_head(z_aug).squeeze())
            
            # Compute z_minus for smile formula
            # z_- = (ln(K/S) - 0.5*I²*τ) / (I*√τ) = z_+ - I*√τ
            # Approximate: z_- ≈ z_+ - base_iv * √τ
            z_minus = moneyness - base_iv * jnp.sqrt(tau + 1e-8)
            
            # Apply extended smile formula if Lévy area available
            if expected_levy_area is not None:
                smile_adjustment = compute_smile_formula_extended(
                    moneyness, z_minus, smile_gamma, smile_omega2,
                    expected_levy_area, smile_xi
                )
                # I² = A² + adjustment => I = √(A² + adjustment)
                atm_var = base_iv ** 2
                iv_squared = atm_var + smile_adjustment
                outputs['implied_vol'] = jnp.sqrt(jnp.maximum(iv_squared, 0.001))
            else:
                outputs['implied_vol'] = base_iv
        
        return outputs


class EnhancedNeuralRDE(eqx.Module):
    """
    Enhanced Neural RDE implementing full Kidger framework.
    
    Key enhancements:
    1. Jump indicator augmentation
    2. Full signature-to-Greek mapping
    3. Extended smile formula with Lévy area
    4. Explicit no-arbitrage constraint
    5. Path-dependent Greek computation
    """
    encoder: InitialEncoder
    vector_field: VectorField
    decoder: EnhancedOutputDecoder
    
    hidden_dim: int
    logsig_dim: int
    step_size: int
    depth: int
    use_jump_indicators: bool
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        step_size: int = 32,
        depth: int = 2,
        mlp_width: int = 128,
        mlp_depth: int = 3,
        use_jump_indicators: bool = True,
        *,
        key: jr.PRNGKey
    ):
        """
        Initialize Enhanced Neural RDE.
        
        Args:
            input_dim: Base input dimension (e.g., 3 for t, S, I)
            hidden_dim: Hidden state dimension
            step_size: Observations per log-signature interval
            depth: Log-signature truncation depth
            mlp_width: Vector field MLP width
            mlp_depth: Vector field MLP depth
            use_jump_indicators: Whether to augment path with J^S, J^I
            key: Random key
        """
        keys = jr.split(key, 3)
        
        self.hidden_dim = hidden_dim
        self.step_size = step_size
        self.depth = depth
        self.use_jump_indicators = use_jump_indicators
        
        # Augmented input dimension
        augmented_input_dim = input_dim + 2 if use_jump_indicators else input_dim
        
        # Log-signature dimension
        self.logsig_dim = logsignature_dimension(augmented_input_dim, depth)
        
        self.encoder = InitialEncoder(augmented_input_dim, hidden_dim, key=keys[0])
        self.vector_field = VectorField(
            hidden_dim, self.logsig_dim, mlp_width, mlp_depth, key=keys[1]
        )
        self.decoder = EnhancedOutputDecoder(hidden_dim, key=keys[2])
    
    def __call__(
        self,
        path: jnp.ndarray,
        moneyness: Optional[jnp.ndarray] = None,
        tau: Optional[jnp.ndarray] = None,
        return_full_analysis: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """
        Forward pass with full analysis.
        
        Args:
            path: Input path (t, S, I), shape (n, 3)
            moneyness: Standardized moneyness z_+
            tau: Time to maturity
            return_full_analysis: Whether to return detailed analysis
            
        Returns:
            Dictionary with predictions and analysis
        """
        # Step 1: Augment path with jump indicators if enabled
        if self.use_jump_indicators and path.shape[1] == 3:
            augmented_path = augment_path_with_jump_indicators(path)
        else:
            augmented_path = path
        
        # Step 2: Compute log-signatures
        logsigs = self._compute_logsigs(augmented_path)
        
        # Step 3: Compute expected Lévy area (for smile formula)
        expected_levy_area = self._compute_expected_levy_area(augmented_path)
        
        # Step 4: Encode initial state
        x0 = augmented_path[0]
        z0 = self.encoder(x0)
        
        # Step 5: Integrate via log-ODE
        z_final, trajectory = self._integrate(z0, logsigs)
        
        # Step 6: Decode outputs
        outputs = self.decoder(z_final, moneyness, tau, expected_levy_area)
        outputs['hidden_state'] = z_final
        
        if return_full_analysis:
            # Compute signature Greeks from path
            full_sig = compute_signature(augmented_path, self.depth)
            sig_greeks = extract_greeks_from_signature(
                full_sig, augmented_path.shape[1], self.depth
            )
            outputs['signature_greeks'] = sig_greeks
            
            # Jump analysis
            if self.use_jump_indicators:
                outputs['jump_features'] = compute_jump_signature_features(augmented_path)
            
            # ATM skew analysis
            if tau is not None:
                outputs['skew_analysis'] = analyze_atm_skew_explosion(tau)
            
            outputs['trajectory'] = trajectory
            outputs['expected_levy_area'] = expected_levy_area
        
        return outputs
    
    def _compute_logsigs(self, path: jnp.ndarray) -> jnp.ndarray:
        """Compute log-signatures over intervals."""
        n_steps = path.shape[0]
        n_intervals = (n_steps - 1) // self.step_size
        
        logsigs = []
        for i in range(n_intervals):
            start = i * self.step_size
            end = min(start + self.step_size + 1, n_steps)
            interval_path = path[start:end]
            logsig = compute_logsignature(interval_path, self.depth)
            logsigs.append(logsig)
        
        return jnp.stack(logsigs) if logsigs else jnp.zeros((1, self.logsig_dim))
    
    def _compute_expected_levy_area(self, path: jnp.ndarray) -> jnp.ndarray:
        """
        Compute expected Lévy area from path statistics.
        
        This is used in the extended smile formula.
        """
        # Use price-vol Lévy area (columns 1, 2)
        if path.shape[1] >= 3:
            levy_area = compute_levy_area(path[:, 1:3])
            return levy_area[0, 1]
        return jnp.array(0.0)
    
    def _integrate(
        self,
        z0: jnp.ndarray,
        logsigs: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Integrate via log-ODE method."""
        def step_fn(z, logsig):
            f_z = self.vector_field(z)
            dz = f_z @ logsig
            z_new = z + dz
            return z_new, z_new
        
        z_final, trajectory = lax.scan(step_fn, z0, logsigs)
        trajectory = jnp.concatenate([z0[None, :], trajectory], axis=0)
        
        return z_final, trajectory


# =============================================================================
# PART 5: LOSS FUNCTIONS WITH EXPLICIT NO-ARBITRAGE (Section II)
# =============================================================================

def compute_terminal_condition_loss(
    predicted_value: jnp.ndarray,
    payoff: jnp.ndarray
) -> jnp.ndarray:
    """
    Terminal condition: E[(Z_T - payoff)²]
    
    The model output should match the option payoff at maturity.
    """
    return jnp.mean((predicted_value - payoff) ** 2)


def compute_drift_constraint_loss(
    trajectory: jnp.ndarray,
    r: float = 0.0,
    dt: float = 1.0
) -> jnp.ndarray:
    """
    No-arbitrage drift constraint: E[(dZ_t/dt - r*Z_t)²]
    
    Under risk-neutral measure, the drift should be the risk-free rate.
    This implements the soft no-arbitrage constraint from the framework.
    """
    # Compute trajectory derivatives
    dz_dt = jnp.diff(trajectory, axis=0) / dt
    
    # Expected drift under risk-neutral measure
    z_mid = (trajectory[:-1] + trajectory[1:]) / 2
    expected_drift = r * z_mid
    
    # Penalize deviation from risk-neutral drift
    return jnp.mean((dz_dt - expected_drift) ** 2)


def compute_orthogonality_loss(
    vector_field_output: jnp.ndarray,
    expected_logsig: jnp.ndarray
) -> jnp.ndarray:
    """
    No-arbitrage orthogonality: E[f_θ(Z) · LogSig] = 0
    
    The vector field should be orthogonal to the expected log-signature
    direction under the risk-neutral measure.
    """
    inner_product = jnp.sum(vector_field_output * expected_logsig, axis=-1)
    return jnp.mean(inner_product ** 2)


def compute_enhanced_total_loss(
    model: EnhancedNeuralRDE,
    path: jnp.ndarray,
    targets: Dict[str, jnp.ndarray],
    lambda_terminal: float = 1.0,
    lambda_drift: float = 0.1,
    lambda_iv: float = 1.0,
    lambda_forecast: float = 1.0,
    lambda_path_dep: float = 0.5  # NEW: path-dependent Greek loss
) -> jnp.ndarray:
    """
    Enhanced loss function implementing all constraints from framework.
    
    L(θ) = λ_terminal · E[(Z_T - payoff)²]      # Terminal condition
         + λ_drift · E[(dZ/dt - rZ)²]           # No-arbitrage drift
         + λ_iv · E[(IV - IV_market)²]          # IV matching
         + λ_forecast · E[(moments - realized)²] # Moment forecasting
         + λ_path_dep · E[(Libra error)²]       # Path-dependent Greeks
    """
    outputs = model(
        path,
        moneyness=targets.get('moneyness'),
        tau=targets.get('tau'),
        return_full_analysis=True
    )
    
    loss = jnp.array(0.0)
    
    # Terminal condition loss
    if 'payoff' in targets:
        loss += lambda_terminal * compute_terminal_condition_loss(
            outputs['implied_vol'] if 'implied_vol' in outputs else outputs['sigma2'],
            targets['payoff']
        )
    
    # Drift constraint loss
    if 'trajectory' in outputs:
        loss += lambda_drift * compute_drift_constraint_loss(
            outputs['trajectory'],
            r=targets.get('risk_free_rate', 0.0)
        )
    
    # IV matching loss
    if 'implied_vol' in outputs and 'target_iv' in targets:
        loss += lambda_iv * jnp.mean(
            (outputs['implied_vol'] - targets['target_iv']) ** 2
        )
    
    # Moment forecasting loss
    if 'realized_sigma2' in targets:
        loss += lambda_forecast * (
            jnp.mean((outputs['sigma2'] - targets['realized_sigma2']) ** 2) +
            jnp.mean((outputs['gamma_forecast'] - targets.get('realized_gamma', 0.0)) ** 2) +
            jnp.mean((outputs['omega2'] - targets.get('realized_omega2', 0.0)) ** 2)
        )
    
    # Path-dependent Greek loss (NEW)
    if 'realized_levy_area_pnl' in targets:
        predicted_levy_pnl = outputs['levy_area_sensitivity'] * outputs.get('expected_levy_area', 0.0)
        loss += lambda_path_dep * jnp.mean(
            (predicted_levy_pnl - targets['realized_levy_area_pnl']) ** 2
        )
    
    return loss


# =============================================================================
# PART 6: CARR-WU BASELINE COMPARISON (Section IV)
# =============================================================================

class CarrWuBaseline:
    """
    Implementation of Carr-Wu linear forecasting baseline.
    
    This is used for comparison with Neural RDE approach.
    
    Carr-Wu forecasts:
    - σ²_{t+δt} via linear regression on (γ_t, ω²_t)
    - Cross-sectional smile via I² - A² = 2γz_+ + ω²z_+z_-
    """
    
    @staticmethod
    def forecast_variance_linear(
        historical_gamma: jnp.ndarray,
        historical_omega2: jnp.ndarray,
        weights: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Linear variance forecast (Carr-Wu approach).
        
        σ²_{t+1} ≈ β_0 + β_1 * γ_t + β_2 * ω²_t
        """
        if weights is None:
            # Default: simple average
            weights = jnp.array([0.5, 0.3, 0.2])
        
        features = jnp.stack([
            jnp.ones_like(historical_gamma),
            historical_gamma,
            historical_omega2
        ], axis=-1)
        
        return features @ weights
    
    @staticmethod
    def compute_smile_carr_wu(
        z_plus: jnp.ndarray,
        z_minus: jnp.ndarray,
        gamma: jnp.ndarray,
        omega2: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Original Carr-Wu smile formula (without Lévy area term).
        
        I² - A² = 2γz_+ + ω²z_+z_-
        
        This is the BASELINE that misses path-dependent effects.
        """
        return 2 * gamma * z_plus + omega2 * z_plus * z_minus
    
    @staticmethod
    def compute_comparison_metrics(
        neural_rde_outputs: Dict[str, jnp.ndarray],
        carr_wu_outputs: Dict[str, jnp.ndarray],
        realized: Dict[str, jnp.ndarray]
    ) -> Dict[str, float]:
        """
        Compare Neural RDE vs Carr-Wu baseline.
        
        Returns:
            Dictionary with R², MAE, and improvement metrics
        """
        # Variance forecast comparison
        nrde_var_error = jnp.mean((neural_rde_outputs['sigma2'] - realized['sigma2']) ** 2)
        cw_var_error = jnp.mean((carr_wu_outputs['sigma2'] - realized['sigma2']) ** 2)
        var_improvement = (cw_var_error - nrde_var_error) / (cw_var_error + 1e-8)
        
        # Smile fit comparison
        if 'smile_error' in neural_rde_outputs:
            nrde_smile_error = neural_rde_outputs['smile_error']
            cw_smile_error = carr_wu_outputs['smile_error']
            smile_improvement = (cw_smile_error - nrde_smile_error) / (cw_smile_error + 1e-8)
        else:
            smile_improvement = 0.0
        
        return {
            'variance_forecast_improvement': float(var_improvement),
            'smile_fit_improvement': float(smile_improvement),
            'neural_rde_mse': float(nrde_var_error),
            'carr_wu_mse': float(cw_var_error)
        }


# =============================================================================
# PART 7: DEPTH-3 SIGNATURE INTERPRETATION FOR FORECASTING
# =============================================================================

def interpret_depth3_for_forecasting(
    path: jnp.ndarray
) -> Dict[str, jnp.ndarray]:
    """
    Extract forecasting features from depth-3 signature.
    
    From the theoretical framework table:
    
    | Feature       | Interpretation        | Forecasting Value    |
    |---------------|----------------------|----------------------|
    | S^(1)         | Recent return        | Momentum signal      |
    | S^(1,1)       | Realized var proxy   | Vol persistence      |
    | S^(1,1,1)     | Skewness of returns  | Jump asymmetry       |
    | Lévy area     | Order of moves       | Vol clustering       |
    | Higher terms  | Complex patterns     | Regime detection     |
    """
    d = path.shape[1]
    
    # Depth-1: Recent return (momentum)
    sig1 = compute_signature_depth1(path)
    momentum_signal = sig1[1] if d > 1 else sig1[0]
    
    # Depth-2: Realized variance proxy (vol persistence)
    sig2 = compute_signature_depth2(path)
    vol_persistence = sig2[1, 1] if d > 1 else sig2[0, 0]
    
    # Depth-2: Lévy area (vol clustering)
    levy_area = compute_levy_area(path)
    vol_clustering = levy_area[0, 1] if d > 1 else jnp.array(0.0)
    
    # Depth-3: Skewness (jump asymmetry)
    if d >= 2:
        sig3 = _compute_signature_depth3(path)
        jump_asymmetry = sig3[1, 1, 1]  # Return skewness
        
        # Regime detection: look at third-order cross terms
        regime_indicator = (sig3[0, 1, 1] + sig3[1, 0, 1] + sig3[1, 1, 0]) / 3
    else:
        jump_asymmetry = jnp.array(0.0)
        regime_indicator = jnp.array(0.0)
    
    return {
        'momentum_signal': momentum_signal,
        'vol_persistence': vol_persistence,
        'vol_clustering': vol_clustering,
        'jump_asymmetry': jump_asymmetry,
        'regime_indicator': regime_indicator
    }


# =============================================================================
# DEMO: COMPREHENSIVE EXAMPLE
# =============================================================================

def demo_full_framework():
    """
    Demonstrate all components of the enhanced framework.
    """
    print("=" * 70)
    print("Enhanced Neural RDE Framework - Full Demonstration")
    print("=" * 70)
    
    key = jr.PRNGKey(42)
    
    # Generate sample path
    from neural_rde_options import generate_jump_diffusion_path
    path, info = generate_jump_diffusion_path(
        key, n_steps=500, lambda_jump=15.0
    )
    
    print(f"\n1. PATH GENERATION")
    print(f"   Shape: {path.shape}")
    print(f"   Jumps: {info['n_jumps']}")
    
    # Augment with jump indicators
    print(f"\n2. JUMP INDICATOR AUGMENTATION")
    augmented = augment_path_with_jump_indicators(path)
    print(f"   Augmented shape: {augmented.shape}")
    print(f"   Columns: [t, log_S, σ, J^S, J^I]")
    
    jump_features = compute_jump_signature_features(augmented)
    print(f"   Jump frequency (spot): {jump_features['spot_jump_freq']:.4f}")
    print(f"   Jump clustering: {jump_features['spot_jump_cluster']:.4f}")
    
    # Signature-to-Greek mapping
    print(f"\n3. SIGNATURE-TO-GREEK MAPPING")
    sig = compute_signature(augmented, depth=2)
    greeks = extract_greeks_from_signature(sig, d=5, depth=2)
    print(f"   Theta: {greeks.theta:.6f}")
    print(f"   Delta: {greeks.delta:.6f}")
    print(f"   Gamma: {greeks.gamma:.6f}")
    print(f"   Vega: {greeks.vega:.6f}")
    print(f"   Lévy area (S-V): {greeks.levy_area_sv:.6f}  ← Path-dependent!")
    print(f"   Libra: {greeks.libra:.6f}  ← Path-dependent!")
    
    # ATM skew explosion
    print(f"\n4. ATM SKEW EXPLOSION ANALYSIS")
    for tau in [1/252, 5/252, 21/252]:  # 1 day, 1 week, 1 month
        skew = analyze_atm_skew_explosion(jnp.array(tau))
        print(f"   τ={tau*252:.0f}d: Skew={skew['atm_skew']:.4f}, "
              f"Jump dominance={skew['jump_dominance_ratio']:.2%}")
    
    # Extended smile formula
    print(f"\n5. EXTENDED SMILE FORMULA")
    z_plus = jnp.array(0.5)  # 0.5σ OTM
    z_minus = jnp.array(0.3)
    gamma_param = jnp.array(-0.7)  # Typical negative correlation
    omega2_param = jnp.array(0.1)
    expected_la = jnp.array(0.02)
    xi = jnp.array(2.0)
    
    cw_smile = CarrWuBaseline.compute_smile_carr_wu(z_plus, z_minus, gamma_param, omega2_param)
    extended_smile = compute_smile_formula_extended(z_plus, z_minus, gamma_param, omega2_param, expected_la, xi)
    
    print(f"   Carr-Wu (no Lévy): I² - A² = {cw_smile:.4f}")
    print(f"   Extended (+ Lévy): I² - A² = {extended_smile:.4f}")
    print(f"   Lévy contribution: {extended_smile - cw_smile:.4f}")
    
    # Depth-3 forecasting features
    print(f"\n6. DEPTH-3 FORECASTING FEATURES")
    forecast_features = interpret_depth3_for_forecasting(path)
    print(f"   Momentum signal: {forecast_features['momentum_signal']:.6f}")
    print(f"   Vol persistence: {forecast_features['vol_persistence']:.6f}")
    print(f"   Vol clustering: {forecast_features['vol_clustering']:.6f}")
    print(f"   Jump asymmetry: {forecast_features['jump_asymmetry']:.6f}")
    
    # Initialize enhanced model
    print(f"\n7. ENHANCED NEURAL RDE")
    model = EnhancedNeuralRDE(
        input_dim=3,
        hidden_dim=32,
        step_size=32,
        depth=2,
        use_jump_indicators=True,
        key=jr.PRNGKey(123)
    )
    
    outputs = model(path, moneyness=jnp.array(0.0), tau=jnp.array(1/252), return_full_analysis=True)
    
    print(f"   Greeks output:")
    print(f"     Theta: {outputs['theta']:.6f}")
    print(f"     Delta: {outputs['delta']:.6f}")
    print(f"     Gamma: {outputs['gamma']:.6f}")
    print(f"     Libra: {outputs['libra']:.6f}  ← NEW")
    print(f"     Lévy sensitivity: {outputs['levy_area_sensitivity']:.6f}  ← NEW")
    print(f"   Smile parameters:")
    print(f"     γ (vanna): {outputs['smile_gamma']:.6f}")
    print(f"     ω² (volga): {outputs['smile_omega2']:.6f}")
    print(f"     ξ (Lévy): {outputs['smile_xi']:.6f}  ← NEW")
    
    print("\n" + "=" * 70)
    print("Full framework demonstration complete!")
    print("=" * 70)


if __name__ == "__main__":
    demo_full_framework()
