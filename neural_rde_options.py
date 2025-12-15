"""
Neural Rough Differential Equations for Short-Dated Options Pricing
====================================================================

A comprehensive implementation of Patrick Kidger's Neural RDE framework
for pricing short-dated options, following the theoretical foundations from:

- Morrill, Salvi, Kidger, Foster, Lyons (2021): "Neural Rough Differential Equations"
- Dupire, Tissot-Daguette: "Signature and the Functional Taylor Expansion"
- Carr, Wu: "Option Profit and Loss Attribution and Pricing"

This implementation uses:
- JAX: Automatic differentiation and JIT compilation
- Equinox: Neural network library for JAX
- Diffrax: Differential equation solvers for JAX

Author: Based on theoretical framework by Patrick Kidger et al.
License: MIT
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import lax
from functools import partial
from typing import Callable, Optional, Tuple, NamedTuple, List
import equinox as eqx
import diffrax
import optax


# =============================================================================
# PART 1: SIGNATURE AND LOG-SIGNATURE COMPUTATION
# =============================================================================

def compute_signature_depth1(path: jnp.ndarray) -> jnp.ndarray:
    """
    Compute depth-1 signature (increments).
    
    Args:
        path: Array of shape (length, channels)
        
    Returns:
        Depth-1 signature: increments, shape (channels,)
    """
    return path[-1] - path[0]


def compute_signature_depth2(path: jnp.ndarray) -> jnp.ndarray:
    """
    Compute depth-2 signature terms including Lévy area.
    
    The depth-2 signature consists of:
    - S^{i,j} = ∫∫_{s<t} dX^i_s dX^j_t for all i,j
    
    The Lévy area is: A^{ij} = (S^{i,j} - S^{j,i}) / 2
    
    Args:
        path: Array of shape (length, channels)
        
    Returns:
        Depth-2 signature terms, shape (channels, channels)
    """
    n_steps, d = path.shape
    
    # Compute increments
    dX = jnp.diff(path, axis=0)  # (n_steps-1, d)
    
    # Cumulative sum for the "past" integral
    X_cumsum = jnp.cumsum(dX, axis=0)  # (n_steps-1, d)
    X_cumsum = jnp.concatenate([jnp.zeros((1, d)), X_cumsum[:-1]], axis=0)
    
    # S^{i,j} = sum_t X_cumsum[t, i] * dX[t, j]
    # This computes ∫_0^T (∫_0^t dX^i_s) dX^j_t
    sig2 = jnp.einsum('ti,tj->ij', X_cumsum, dX)
    
    return sig2


def compute_levy_area(path: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Lévy area (antisymmetric part of depth-2 signature).
    
    A^{ij} = (S^{i,j} - S^{j,i}) / 2
    
    This measures the "signed area" swept by the path projection onto
    the (i,j) plane, capturing the ORDER of movements.
    
    Args:
        path: Array of shape (length, channels)
        
    Returns:
        Lévy area matrix, shape (channels, channels), antisymmetric
    """
    sig2 = compute_signature_depth2(path)
    levy_area = (sig2 - sig2.T) / 2.0
    return levy_area


def compute_signature(path: jnp.ndarray, depth: int = 2) -> jnp.ndarray:
    """
    Compute the truncated signature of a path up to specified depth.
    
    The signature is the fundamental object in rough path theory:
    Sig(X) = (1, S^1, S^2, ..., S^{i,j}, S^{i,j,k}, ...)
    
    Args:
        path: Array of shape (length, channels)
        depth: Maximum depth of signature (1, 2, or 3)
        
    Returns:
        Flattened signature vector
    """
    d = path.shape[1]
    
    # Depth 1: increments
    sig1 = compute_signature_depth1(path)  # (d,)
    
    if depth == 1:
        return sig1
    
    # Depth 2: iterated integrals
    sig2 = compute_signature_depth2(path)  # (d, d)
    sig2_flat = sig2.flatten()  # (d*d,)
    
    if depth == 2:
        return jnp.concatenate([sig1, sig2_flat])
    
    # Depth 3: triple iterated integrals (simplified computation)
    if depth >= 3:
        sig3 = _compute_signature_depth3(path)  # (d, d, d)
        sig3_flat = sig3.flatten()  # (d*d*d,)
        return jnp.concatenate([sig1, sig2_flat, sig3_flat])
    
    return jnp.concatenate([sig1, sig2_flat])


def _compute_signature_depth3(path: jnp.ndarray) -> jnp.ndarray:
    """
    Compute depth-3 signature terms.
    
    S^{i,j,k} = ∫∫∫_{r<s<t} dX^i_r dX^j_s dX^k_t
    """
    n_steps, d = path.shape
    dX = jnp.diff(path, axis=0)
    
    # Build cumulative sums for nested integrals
    sig3 = jnp.zeros((d, d, d))
    
    # Compute via sequential summation
    running_sum_i = jnp.zeros(d)
    running_sum_ij = jnp.zeros((d, d))
    
    def step_fn(carry, dX_t):
        running_sum_i, running_sum_ij, sig3 = carry
        
        # Update depth-3: sig3^{ijk} += running_sum_ij^{ij} * dX_t^k
        sig3 = sig3 + jnp.einsum('ij,k->ijk', running_sum_ij, dX_t)
        
        # Update depth-2 running sum: running_sum_ij^{ij} += running_sum_i^i * dX_t^j
        running_sum_ij = running_sum_ij + jnp.outer(running_sum_i, dX_t)
        
        # Update depth-1 running sum
        running_sum_i = running_sum_i + dX_t
        
        return (running_sum_i, running_sum_ij, sig3), None
    
    (_, _, sig3), _ = lax.scan(step_fn, (running_sum_i, running_sum_ij, sig3), dX)
    
    return sig3


def compute_logsignature(path: jnp.ndarray, depth: int = 2) -> jnp.ndarray:
    """
    Compute the log-signature of a path.
    
    The log-signature is a compressed representation of the signature
    that removes algebraic redundancies. For depth 2:
    
    LogSig = (S^1, S^2, ..., (S^{1,2} - S^{2,1})/2, ...)
    
    The key terms are:
    - Depth 1: increments (same as signature)
    - Depth 2: Lévy areas (antisymmetric parts only)
    
    Args:
        path: Array of shape (length, channels)
        depth: Maximum depth
        
    Returns:
        Log-signature vector of dimension β(d, depth)
    """
    d = path.shape[1]
    
    # Depth 1: increments
    logsig1 = compute_signature_depth1(path)  # (d,)
    
    if depth == 1:
        return logsig1
    
    # Depth 2: Lévy areas (extract upper triangular of antisymmetric matrix)
    levy_area = compute_levy_area(path)  # (d, d)
    
    # Extract unique Lévy area terms (upper triangular, excluding diagonal)
    logsig2_terms = []
    for i in range(d):
        for j in range(i + 1, d):
            logsig2_terms.append(levy_area[i, j])
    
    logsig2 = jnp.array(logsig2_terms) if logsig2_terms else jnp.array([])
    
    if depth == 2:
        return jnp.concatenate([logsig1, logsig2])
    
    # Depth 3: Higher order terms (simplified - use Hall basis elements)
    if depth >= 3:
        sig3 = _compute_signature_depth3(path)
        # Extract Lyndon word terms (simplified: just use antisymmetric combinations)
        logsig3_terms = []
        for i in range(d):
            for j in range(d):
                if i < j:
                    # [e_i, [e_i, e_j]] type terms
                    term = (sig3[i, i, j] + sig3[j, i, i] - 2 * sig3[i, j, i]) / 6.0
                    logsig3_terms.append(term)
                    # [e_j, [e_i, e_j]] type terms  
                    term = -(sig3[i, j, j] + sig3[j, j, i] - 2 * sig3[j, i, j]) / 6.0
                    logsig3_terms.append(term)
        
        logsig3 = jnp.array(logsig3_terms) if logsig3_terms else jnp.array([])
        return jnp.concatenate([logsig1, logsig2, logsig3])
    
    return jnp.concatenate([logsig1, logsig2])


def logsignature_dimension(d: int, depth: int) -> int:
    """
    Compute the dimension of the log-signature.
    
    β(d, M) = Σ_{k=1}^{M} (1/k) Σ_{j|k} μ(k/j) d^j
    
    where μ is the Möbius function.
    
    For practical purposes:
    - depth 1: d
    - depth 2: d + d(d-1)/2 = d(d+1)/2
    - depth 3: d + d(d-1)/2 + d(d-1) = d + d(d-1)/2 + d(d-1)
    """
    if depth == 1:
        return d
    elif depth == 2:
        return d + d * (d - 1) // 2
    elif depth == 3:
        # Approximate for depth 3
        return d + d * (d - 1) // 2 + d * (d - 1)
    else:
        # General formula using Möbius function (simplified)
        total = 0
        for k in range(1, depth + 1):
            total += d ** k // k  # Approximation
        return total


def compute_logsignatures_for_intervals(
    path: jnp.ndarray,
    step_size: int,
    depth: int = 2
) -> jnp.ndarray:
    """
    Compute log-signatures over non-overlapping intervals.
    
    This is the key preprocessing step for Neural RDEs:
    1. Split path into intervals of size `step_size`
    2. Compute log-signature for each interval
    3. Return sequence of log-signatures
    
    Args:
        path: Full path, shape (length, channels)
        step_size: Number of observations per interval
        depth: Log-signature truncation depth
        
    Returns:
        Array of log-signatures, shape (n_intervals, logsig_dim)
    """
    n_steps = path.shape[0]
    n_intervals = (n_steps - 1) // step_size
    
    logsigs = []
    for i in range(n_intervals):
        start = i * step_size
        end = min(start + step_size + 1, n_steps)
        interval_path = path[start:end]
        logsig = compute_logsignature(interval_path, depth)
        logsigs.append(logsig)
    
    return jnp.stack(logsigs)


# Vectorized version for batches
@partial(jax.vmap, in_axes=(0, None, None))
def batch_compute_logsignatures(
    paths: jnp.ndarray,
    step_size: int,
    depth: int
) -> jnp.ndarray:
    """Compute log-signatures for a batch of paths."""
    return compute_logsignatures_for_intervals(paths, step_size, depth)


# =============================================================================
# PART 2: NEURAL RDE ARCHITECTURE
# =============================================================================

class VectorField(eqx.Module):
    """
    Neural network vector field f_θ for the Neural RDE.
    
    Maps hidden state Z ∈ ℝ^hidden_dim to a matrix in ℝ^{hidden_dim × logsig_dim}
    that will be multiplied with the log-signature.
    
    Architecture:
        Z -> MLP -> reshape -> (hidden_dim, logsig_dim)
    """
    layers: List[eqx.nn.Linear]
    activation: Callable
    hidden_dim: int
    logsig_dim: int
    
    def __init__(
        self,
        hidden_dim: int,
        logsig_dim: int,
        width: int = 128,
        depth: int = 3,
        *,
        key: jr.PRNGKey
    ):
        self.hidden_dim = hidden_dim
        self.logsig_dim = logsig_dim
        self.activation = jax.nn.gelu
        
        # Build MLP layers
        keys = jr.split(key, depth)
        layers = []
        
        # Input layer
        layers.append(eqx.nn.Linear(hidden_dim, width, key=keys[0]))
        
        # Hidden layers
        for i in range(1, depth - 1):
            layers.append(eqx.nn.Linear(width, width, key=keys[i]))
        
        # Output layer: maps to hidden_dim * logsig_dim
        layers.append(eqx.nn.Linear(width, hidden_dim * logsig_dim, key=keys[-1]))
        
        self.layers = layers
    
    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        """
        Compute vector field value.
        
        Args:
            z: Hidden state, shape (hidden_dim,)
            
        Returns:
            Matrix of shape (hidden_dim, logsig_dim)
        """
        x = z
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        
        # Reshape to matrix
        return x.reshape(self.hidden_dim, self.logsig_dim)


class InitialEncoder(eqx.Module):
    """
    Encoder for initial hidden state from initial observation.
    
    Maps X_0 ∈ ℝ^input_dim to Z_0 ∈ ℝ^hidden_dim
    """
    layers: List[eqx.nn.Linear]
    activation: Callable
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        width: int = 64,
        *,
        key: jr.PRNGKey
    ):
        keys = jr.split(key, 3)
        self.activation = jax.nn.gelu
        self.layers = [
            eqx.nn.Linear(input_dim, width, key=keys[0]),
            eqx.nn.Linear(width, width, key=keys[1]),
            eqx.nn.Linear(width, hidden_dim, key=keys[2])
        ]
    
    def __call__(self, x0: jnp.ndarray) -> jnp.ndarray:
        x = x0
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        return self.layers[-1](x)


class OutputDecoder(eqx.Module):
    """
    Decoder for extracting predictions from hidden state.
    
    Multiple heads for different outputs:
    - implied_vol: Implied volatility prediction
    - sigma2: Instantaneous variance forecast
    - gamma: Return-vol covariance forecast
    - omega2: Vol-of-vol forecast
    - greeks: Signature Greeks (Δ, Γ, Θ, V, L)
    """
    vol_head: eqx.nn.Linear
    sigma2_head: eqx.nn.Linear
    gamma_head: eqx.nn.Linear
    omega2_head: eqx.nn.Linear
    greeks_head: eqx.nn.Linear
    hidden_dim: int
    
    def __init__(
        self,
        hidden_dim: int,
        n_greeks: int = 5,
        *,
        key: jr.PRNGKey
    ):
        keys = jr.split(key, 5)
        self.hidden_dim = hidden_dim
        
        # Implied vol head takes (hidden_state, moneyness, tau)
        self.vol_head = eqx.nn.Linear(hidden_dim + 2, 1, key=keys[0])
        
        # Moment forecasting heads
        self.sigma2_head = eqx.nn.Linear(hidden_dim, 1, key=keys[1])
        self.gamma_head = eqx.nn.Linear(hidden_dim, 1, key=keys[2])
        self.omega2_head = eqx.nn.Linear(hidden_dim, 1, key=keys[3])
        
        # Greeks head: outputs [Delta, Gamma, Theta, Vega, Libra]
        self.greeks_head = eqx.nn.Linear(hidden_dim, n_greeks, key=keys[4])
    
    def __call__(
        self,
        z: jnp.ndarray,
        moneyness: Optional[jnp.ndarray] = None,
        tau: Optional[jnp.ndarray] = None
    ) -> dict:
        """
        Decode hidden state to predictions.
        
        Args:
            z: Hidden state, shape (hidden_dim,)
            moneyness: Optional standardized moneyness z_+
            tau: Optional time to maturity
            
        Returns:
            Dictionary of predictions
        """
        outputs = {
            'sigma2': jax.nn.softplus(self.sigma2_head(z).squeeze()),
            'gamma': self.gamma_head(z).squeeze(),
            'omega2': jax.nn.softplus(self.omega2_head(z).squeeze()),
            'greeks': self.greeks_head(z)
        }
        
        if moneyness is not None and tau is not None:
            # Augment hidden state with option characteristics
            z_aug = jnp.concatenate([z, jnp.atleast_1d(moneyness), jnp.atleast_1d(tau)])
            outputs['implied_vol'] = jax.nn.softplus(self.vol_head(z_aug).squeeze())
        
        return outputs


class NeuralRDE(eqx.Module):
    """
    Neural Rough Differential Equation for option pricing.
    
    The model solves the log-ODE approximation:
    
        Z_t = Z_0 + ∫_0^t f_θ(Z_s) · (LogSig / Δt) ds
    
    where:
    - Z_t is the hidden state encoding portfolio risk profile
    - f_θ is the learned vector field
    - LogSig is the log-signature over each interval
    
    This implements Kidger's framework for short-dated options:
    1. Compress high-frequency path via log-signatures
    2. Evolve hidden state via Neural RDE
    3. Decode to option prices, Greeks, and forecasts
    """
    encoder: InitialEncoder
    vector_field: VectorField
    decoder: OutputDecoder
    hidden_dim: int
    logsig_dim: int
    step_size: int
    depth: int
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        logsig_dim: int = 8,
        step_size: int = 32,
        depth: int = 2,
        mlp_width: int = 128,
        mlp_depth: int = 3,
        *,
        key: jr.PRNGKey
    ):
        """
        Initialize Neural RDE.
        
        Args:
            input_dim: Dimension of path (e.g., 3 for (t, S, I))
            hidden_dim: Dimension of hidden state
            logsig_dim: Dimension of log-signature
            step_size: Observations per interval for log-signature
            depth: Log-signature truncation depth
            mlp_width: Width of MLP layers
            mlp_depth: Depth of MLP
            key: Random key
        """
        keys = jr.split(key, 3)
        
        self.hidden_dim = hidden_dim
        self.logsig_dim = logsig_dim
        self.step_size = step_size
        self.depth = depth
        
        self.encoder = InitialEncoder(input_dim, hidden_dim, key=keys[0])
        self.vector_field = VectorField(
            hidden_dim, logsig_dim, mlp_width, mlp_depth, key=keys[1]
        )
        self.decoder = OutputDecoder(hidden_dim, key=keys[2])
    
    def __call__(
        self,
        path: jnp.ndarray,
        logsigs: Optional[jnp.ndarray] = None,
        moneyness: Optional[jnp.ndarray] = None,
        tau: Optional[jnp.ndarray] = None,
        return_trajectory: bool = False
    ) -> dict:
        """
        Forward pass through Neural RDE.
        
        Args:
            path: Input path, shape (length, input_dim)
            logsigs: Pre-computed log-signatures (optional)
            moneyness: Standardized moneyness for option pricing
            tau: Time to maturity
            return_trajectory: Whether to return full hidden state trajectory
            
        Returns:
            Dictionary with predictions and optionally trajectory
        """
        # Compute log-signatures if not provided
        if logsigs is None:
            logsigs = compute_logsignatures_for_intervals(
                path, self.step_size, self.depth
            )
        
        # Initialize hidden state from first observation
        x0 = path[0]
        z0 = self.encoder(x0)
        
        # Integrate via log-ODE method
        z_final, trajectory = self._integrate(z0, logsigs, return_trajectory)
        
        # Decode outputs
        outputs = self.decoder(z_final, moneyness, tau)
        
        if return_trajectory:
            outputs['trajectory'] = trajectory
        
        outputs['hidden_state'] = z_final
        
        return outputs
    
    def _integrate(
        self,
        z0: jnp.ndarray,
        logsigs: jnp.ndarray,
        return_trajectory: bool = False
    ) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        """
        Integrate the Neural RDE using log-ODE method.
        
        For each interval i with log-signature L_i:
            dZ/dt = f_θ(Z) · (L_i / Δt)
            
        We use a simple Euler scheme; for production, use Diffrax.
        """
        n_intervals = logsigs.shape[0]
        dt = 1.0  # Normalized interval length
        
        def step_fn(z, logsig):
            # Vector field evaluation
            f_z = self.vector_field(z)  # (hidden_dim, logsig_dim)
            
            # dZ = f_θ(Z) · LogSig
            dz = f_z @ logsig
            
            # Euler update (could use more sophisticated solver)
            z_new = z + dz * dt
            
            return z_new, z_new
        
        z_final, trajectory = lax.scan(step_fn, z0, logsigs)
        
        if return_trajectory:
            # Prepend initial state
            trajectory = jnp.concatenate([z0[None, :], trajectory], axis=0)
            return z_final, trajectory
        
        return z_final, None
    
    def integrate_diffrax(
        self,
        z0: jnp.ndarray,
        logsigs: jnp.ndarray,
        dt0: float = 0.1
    ) -> jnp.ndarray:
        """
        Integrate using Diffrax for better numerical accuracy.
        
        This solves the ODE:
            dZ/dt = f_θ(Z) · LogSig(t) / Δt
            
        where LogSig(t) is piecewise constant over intervals.
        """
        n_intervals = logsigs.shape[0]
        t0, t1 = 0.0, float(n_intervals)
        
        def vector_field_ode(t, z, args):
            logsigs = args
            # Get interval index
            interval_idx = jnp.clip(jnp.floor(t).astype(int), 0, n_intervals - 1)
            logsig = logsigs[interval_idx]
            
            # Compute dZ/dt
            f_z = self.vector_field(z)
            return f_z @ logsig
        
        term = diffrax.ODETerm(vector_field_ode)
        solver = diffrax.Tsit5()  # 5th order Runge-Kutta
        saveat = diffrax.SaveAt(t1=True)
        
        solution = diffrax.diffeqsolve(
            term, solver, t0, t1, dt0, z0,
            args=logsigs,
            saveat=saveat
        )
        
        return solution.ys[0]


# =============================================================================
# PART 3: SIGNATURE-BASED P&L ATTRIBUTION
# =============================================================================

class SignatureGreeks(NamedTuple):
    """
    Greeks derived from signature-based P&L decomposition.
    
    P&L ≈ Θ·δt + Δ·δy + Γ·(δy²/2) + L·A(Y)
    
    where:
    - Θ: Theta (time decay)
    - Δ: Delta (spot sensitivity)
    - Γ: Gamma (convexity)
    - V: Vega (vol sensitivity)
    - L: Libra (Lévy area sensitivity)
    """
    theta: jnp.ndarray
    delta: jnp.ndarray
    gamma: jnp.ndarray
    vega: jnp.ndarray
    libra: jnp.ndarray


def decompose_pnl_signature(
    pnl: jnp.ndarray,
    path: jnp.ndarray,
    greeks: SignatureGreeks
) -> dict:
    """
    Decompose realized P&L using signature Greeks.
    
    This implements the Functional Taylor Expansion:
    
    P&L = f(X*Y) - f(X) 
        ≈ Σ_{|I|≤k} Δ_I f(X) · S^I(Y)
        = Θ·δt + Δ·δy + Γ·(δy²/2) + V·δσ + L·A(Y) + ...
    
    Args:
        pnl: Realized P&L
        path: Future path Y
        greeks: Signature Greeks
        
    Returns:
        Dictionary with P&L attribution
    """
    # Extract path increments
    dt = path[-1, 0] - path[0, 0]  # Time increment
    dy = path[-1, 1] - path[0, 1]  # Price increment
    dsigma = path[-1, 2] - path[0, 2]  # Vol increment
    
    # Compute Lévy area for (price, vol) pair
    levy_area = compute_levy_area(path[:, 1:3])
    levy_area_pv = levy_area[0, 1]  # Price-vol Lévy area
    
    # P&L decomposition
    theta_pnl = greeks.theta * dt
    delta_pnl = greeks.delta * dy
    gamma_pnl = greeks.gamma * (dy ** 2) / 2
    vega_pnl = greeks.vega * dsigma
    libra_pnl = greeks.libra * levy_area_pv
    
    explained_pnl = theta_pnl + delta_pnl + gamma_pnl + vega_pnl + libra_pnl
    residual = pnl - explained_pnl
    
    return {
        'theta_pnl': theta_pnl,
        'delta_pnl': delta_pnl,
        'gamma_pnl': gamma_pnl,
        'vega_pnl': vega_pnl,
        'libra_pnl': libra_pnl,
        'explained_pnl': explained_pnl,
        'residual': residual,
        'levy_area': levy_area_pv
    }


def compute_libra_greek(
    pricing_fn: Callable,
    path: jnp.ndarray,
    h: float = 0.01,
    dt: float = 0.01
) -> jnp.ndarray:
    """
    Compute the Libra Greek via finite differences.
    
    L = Δ_xt - Δ_tx = lim_{h,δt→0} [f((X^h)^{*δt}) - f((X^{*δt})^h)] / (h·δt)
    
    This measures sensitivity to the order of time and space increments,
    which is nonzero only for genuinely path-dependent functionals.
    
    Args:
        pricing_fn: Function that takes path and returns price
        path: Observed path X
        h: Bump size for space
        dt: Bump size for time
        
    Returns:
        Libra Greek estimate
    """
    # (X^h)^{*δt}: bump in space, then extend in time
    path_h_dt = _bump_then_extend(path, h, dt)
    
    # (X^{*δt})^h: extend in time, then bump in space
    path_dt_h = _extend_then_bump(path, h, dt)
    
    # (X^{-h})^{*δt}: negative bump
    path_mh_dt = _bump_then_extend(path, -h, dt)
    
    # (X^{*δt})^{-h}: extend then negative bump
    path_dt_mh = _extend_then_bump(path, -h, dt)
    
    # Finite difference approximation
    libra = (pricing_fn(path_h_dt) - pricing_fn(path_dt_h) 
             - pricing_fn(path_mh_dt) + pricing_fn(path_dt_mh)) / (2 * h * dt)
    
    return libra


def _bump_then_extend(path: jnp.ndarray, h: float, dt: float) -> jnp.ndarray:
    """Bump the path in space, then extend in time."""
    bumped = path.at[:, 1].add(h)  # Bump price channel
    # Extend: add a new point at time t + dt with same price
    new_point = jnp.array([[path[-1, 0] + dt, path[-1, 1] + h, path[-1, 2]]])
    return jnp.concatenate([bumped, new_point], axis=0)


def _extend_then_bump(path: jnp.ndarray, h: float, dt: float) -> jnp.ndarray:
    """Extend the path in time, then bump in space."""
    # Extend first
    new_point = jnp.array([[path[-1, 0] + dt, path[-1, 1], path[-1, 2]]])
    extended = jnp.concatenate([path, new_point], axis=0)
    # Then bump
    return extended.at[:, 1].add(h)


# =============================================================================
# PART 4: LOSS FUNCTIONS AND TRAINING
# =============================================================================

def compute_pnl_loss(
    model: NeuralRDE,
    path: jnp.ndarray,
    logsigs: jnp.ndarray,
    realized_pnl: jnp.ndarray
) -> jnp.ndarray:
    """
    Loss for P&L prediction accuracy.
    
    The model predicts Greeks, which are combined with signature
    terms to predict P&L.
    """
    outputs = model(path, logsigs)
    greeks = outputs['greeks']
    
    # Extract signature terms from path
    sig = compute_signature(path, depth=2)
    d = path.shape[1]
    
    # Parse signature
    dt = sig[0]  # Time increment
    dy = sig[1]  # Price increment
    dsigma = sig[2] if d > 2 else 0.0
    
    # Get Lévy area (depth-2 term between price and vol)
    if d >= 3:
        sig2_flat = sig[d:d + d*d]
        sig2 = sig2_flat.reshape(d, d)
        levy_area = (sig2[1, 2] - sig2[2, 1]) / 2.0
    else:
        levy_area = 0.0
    
    # Predict P&L: Θ·δt + Δ·δy + Γ·(δy²/2) + V·δσ + L·A
    theta, delta, gamma, vega, libra = greeks[0], greeks[1], greeks[2], greeks[3], greeks[4]
    
    predicted_pnl = (
        theta * dt +
        delta * dy +
        gamma * (dy ** 2) / 2 +
        vega * dsigma +
        libra * levy_area
    )
    
    return jnp.mean((predicted_pnl - realized_pnl) ** 2)


def compute_forecast_loss(
    model: NeuralRDE,
    path: jnp.ndarray,
    logsigs: jnp.ndarray,
    realized_sigma2: jnp.ndarray,
    realized_gamma: jnp.ndarray,
    realized_omega2: jnp.ndarray
) -> jnp.ndarray:
    """
    Loss for moment condition forecasting.
    
    The model should predict:
    - σ²: Instantaneous variance rate
    - γ: Return-vol covariance
    - ω²: Vol-of-vol
    """
    outputs = model(path, logsigs)
    
    sigma2_loss = jnp.mean((outputs['sigma2'] - realized_sigma2) ** 2)
    gamma_loss = jnp.mean((outputs['gamma'] - realized_gamma) ** 2)
    omega2_loss = jnp.mean((outputs['omega2'] - realized_omega2) ** 2)
    
    return sigma2_loss + gamma_loss + omega2_loss


def compute_no_arbitrage_loss(
    model: NeuralRDE,
    path: jnp.ndarray,
    logsigs: jnp.ndarray
) -> jnp.ndarray:
    """
    Soft constraint for no-arbitrage condition.
    
    Under risk-neutral measure, the hidden state drift should be zero
    (after accounting for risk-free rate).
    
    This is a regularization term.
    """
    outputs = model(path, logsigs, return_trajectory=True)
    trajectory = outputs['trajectory']  # (n_intervals + 1, hidden_dim)
    
    # Compute trajectory increments
    dz = jnp.diff(trajectory, axis=0)  # (n_intervals, hidden_dim)
    
    # Penalize large drifts (should be zero under Q)
    drift_penalty = jnp.mean(dz ** 2)
    
    return drift_penalty


def compute_implied_vol_loss(
    model: NeuralRDE,
    path: jnp.ndarray,
    logsigs: jnp.ndarray,
    moneyness: jnp.ndarray,
    tau: jnp.ndarray,
    target_iv: jnp.ndarray
) -> jnp.ndarray:
    """
    Loss for implied volatility prediction.
    
    Compare model's IV prediction against market IV.
    """
    outputs = model(path, logsigs, moneyness, tau)
    predicted_iv = outputs['implied_vol']
    
    return jnp.mean((predicted_iv - target_iv) ** 2)


def total_loss(
    model: NeuralRDE,
    batch: dict,
    lambda_pnl: float = 1.0,
    lambda_forecast: float = 1.0,
    lambda_arbitrage: float = 0.1,
    lambda_iv: float = 1.0
) -> jnp.ndarray:
    """
    Combined loss function for training Neural RDE.
    
    L = λ_pnl · L_pnl + λ_forecast · L_forecast + λ_arbitrage · L_arbitrage + λ_iv · L_iv
    """
    path = batch['path']
    logsigs = batch['logsigs']
    
    loss = 0.0
    
    if 'realized_pnl' in batch:
        loss += lambda_pnl * compute_pnl_loss(model, path, logsigs, batch['realized_pnl'])
    
    if 'realized_sigma2' in batch:
        loss += lambda_forecast * compute_forecast_loss(
            model, path, logsigs,
            batch['realized_sigma2'],
            batch['realized_gamma'],
            batch['realized_omega2']
        )
    
    loss += lambda_arbitrage * compute_no_arbitrage_loss(model, path, logsigs)
    
    if 'target_iv' in batch:
        loss += lambda_iv * compute_implied_vol_loss(
            model, path, logsigs,
            batch['moneyness'], batch['tau'], batch['target_iv']
        )
    
    return loss


# =============================================================================
# PART 5: DATA GENERATION (SYNTHETIC)
# =============================================================================

def generate_jump_diffusion_path(
    key: jr.PRNGKey,
    n_steps: int = 1000,
    dt: float = 1/252/390,  # 1 minute
    S0: float = 100.0,
    sigma0: float = 0.2,
    kappa: float = 2.0,      # Vol mean reversion speed
    theta: float = 0.2,      # Vol long-term mean
    xi: float = 0.3,         # Vol-of-vol
    rho: float = -0.7,       # Spot-vol correlation
    lambda_jump: float = 5.0,  # Jump intensity (per year)
    mu_jump: float = -0.02,  # Mean jump size
    sigma_jump: float = 0.03 # Jump size std
) -> Tuple[jnp.ndarray, dict]:
    """
    Generate a path from a Bates-like jump-diffusion model.
    
    dS/S = σ dW^S + (e^J - 1) dN - λE[e^J - 1] dt
    dσ = κ(θ - σ) dt + ξ σ dW^σ
    
    where:
    - dW^S, dW^σ have correlation ρ
    - N is Poisson with intensity λ
    - J ~ N(μ_jump, σ_jump²)
    
    Args:
        key: Random key
        n_steps: Number of time steps
        dt: Time step size
        ... model parameters ...
        
    Returns:
        path: Array of shape (n_steps, 3) with columns (t, log_S, σ)
        info: Dictionary with additional information
    """
    keys = jr.split(key, 4)
    
    # Time grid
    t = jnp.arange(n_steps) * dt
    
    # Generate correlated Brownian increments
    dW = jr.normal(keys[0], (n_steps - 1, 2)) * jnp.sqrt(dt)
    # Apply correlation
    L = jnp.array([[1, 0], [rho, jnp.sqrt(1 - rho**2)]])
    dW = dW @ L.T
    dW_S, dW_sigma = dW[:, 0], dW[:, 1]
    
    # Generate Poisson jumps
    n_jumps = jr.poisson(keys[1], lambda_jump * dt, (n_steps - 1,))
    jump_sizes = jr.normal(keys[2], (n_steps - 1,)) * sigma_jump + mu_jump
    jump_sizes = jump_sizes * (n_jumps > 0)  # Only apply when jump occurs
    
    # Initialize
    log_S = jnp.zeros(n_steps)
    sigma = jnp.zeros(n_steps)
    log_S = log_S.at[0].set(jnp.log(S0))
    sigma = sigma.at[0].set(sigma0)
    
    # Euler-Maruyama simulation
    def step_fn(carry, inputs):
        log_S_t, sigma_t = carry
        dW_S_t, dW_sigma_t, jump_t = inputs
        
        # Volatility update (CIR-like, but on sigma not variance)
        sigma_new = sigma_t + kappa * (theta - sigma_t) * dt + xi * sigma_t * dW_sigma_t
        sigma_new = jnp.maximum(sigma_new, 0.01)  # Floor at 1%
        
        # Log-price update with jump
        drift = -0.5 * sigma_t ** 2 - lambda_jump * (jnp.exp(mu_jump + 0.5 * sigma_jump**2) - 1)
        log_S_new = log_S_t + drift * dt + sigma_t * dW_S_t + jump_t
        
        return (log_S_new, sigma_new), (log_S_new, sigma_new)
    
    inputs = (dW_S, dW_sigma, jump_sizes)
    _, (log_S_path, sigma_path) = lax.scan(
        step_fn, (log_S[0], sigma[0]), 
        (dW_S, dW_sigma, jump_sizes)
    )
    
    log_S = jnp.concatenate([log_S[:1], log_S_path])
    sigma = jnp.concatenate([sigma[:1], sigma_path])
    
    # Construct path: (t, log_S, σ)
    path = jnp.stack([t, log_S, sigma], axis=1)
    
    # Compute realized quantities for targets
    returns = jnp.diff(log_S)
    realized_sigma2 = jnp.mean(returns ** 2) / dt  # Annualized
    realized_gamma = jnp.corrcoef(returns[:-1], jnp.diff(sigma)[:-1])[0, 1] * jnp.std(returns) * jnp.std(jnp.diff(sigma))
    realized_omega2 = jnp.var(sigma) / dt
    
    info = {
        'realized_sigma2': realized_sigma2,
        'realized_gamma': realized_gamma,
        'realized_omega2': realized_omega2,
        'n_jumps': jnp.sum(n_jumps),
        'returns': returns
    }
    
    return path, info


def generate_training_batch(
    key: jr.PRNGKey,
    batch_size: int = 32,
    n_steps: int = 1000,
    step_size: int = 32,
    depth: int = 2,
    **model_params
) -> dict:
    """
    Generate a batch of training data.
    
    Each sample consists of:
    - path: The price/vol path
    - logsigs: Pre-computed log-signatures
    - realized_*: Target moment conditions
    - target_iv: Implied volatility (from BS formula)
    - moneyness, tau: Option characteristics
    """
    keys = jr.split(key, batch_size + 1)
    
    paths = []
    logsigs_list = []
    realized_sigma2s = []
    realized_gammas = []
    realized_omega2s = []
    
    for i in range(batch_size):
        path, info = generate_jump_diffusion_path(keys[i], n_steps, **model_params)
        logsigs = compute_logsignatures_for_intervals(path, step_size, depth)
        
        paths.append(path)
        logsigs_list.append(logsigs)
        realized_sigma2s.append(info['realized_sigma2'])
        realized_gammas.append(info['realized_gamma'])
        realized_omega2s.append(info['realized_omega2'])
    
    # Generate random option characteristics
    moneyness = jr.uniform(keys[-1], (batch_size,), minval=-2.0, maxval=2.0)
    tau = jr.uniform(keys[-1], (batch_size,), minval=1/252, maxval=5/252)  # 1-5 days
    
    # Target IV (simplified: use average realized vol)
    target_iv = jnp.sqrt(jnp.array(realized_sigma2s))
    
    return {
        'path': jnp.stack(paths),
        'logsigs': jnp.stack(logsigs_list),
        'realized_sigma2': jnp.array(realized_sigma2s),
        'realized_gamma': jnp.array(realized_gammas),
        'realized_omega2': jnp.array(realized_omega2s),
        'moneyness': moneyness,
        'tau': tau,
        'target_iv': target_iv
    }


# =============================================================================
# PART 6: TRAINING LOOP
# =============================================================================

def create_train_step(
    model: NeuralRDE,
    optimizer: optax.GradientTransformation,
    loss_weights: dict
):
    """Create a JIT-compiled training step function."""
    
    @eqx.filter_jit
    def train_step(model, opt_state, batch):
        def loss_fn(model):
            # Process each sample in batch
            total_loss = 0.0
            batch_size = batch['path'].shape[0]
            
            for i in range(batch_size):
                sample = {k: v[i] for k, v in batch.items()}
                total_loss += total_loss_single(model, sample, **loss_weights)
            
            return total_loss / batch_size
        
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model)
        updates, opt_state = optimizer.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        
        return model, opt_state, loss
    
    return train_step


def total_loss_single(
    model: NeuralRDE,
    sample: dict,
    lambda_pnl: float = 1.0,
    lambda_forecast: float = 1.0,
    lambda_arbitrage: float = 0.1,
    lambda_iv: float = 1.0
) -> jnp.ndarray:
    """Compute loss for a single sample."""
    path = sample['path']
    logsigs = sample['logsigs']
    
    outputs = model(path, logsigs, sample.get('moneyness'), sample.get('tau'))
    
    loss = 0.0
    
    # Forecast loss
    if 'realized_sigma2' in sample:
        loss += lambda_forecast * (
            (outputs['sigma2'] - sample['realized_sigma2']) ** 2 +
            (outputs['gamma'] - sample['realized_gamma']) ** 2 +
            (outputs['omega2'] - sample['realized_omega2']) ** 2
        )
    
    # IV loss
    if 'target_iv' in sample and 'implied_vol' in outputs:
        loss += lambda_iv * (outputs['implied_vol'] - sample['target_iv']) ** 2
    
    return loss


def train_model(
    model: NeuralRDE,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    key: jr.PRNGKey = jr.PRNGKey(0),
    verbose: bool = True
) -> Tuple[NeuralRDE, List[float]]:
    """
    Train the Neural RDE model.
    
    Args:
        model: Initialized NeuralRDE
        n_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        key: Random key
        verbose: Print progress
        
    Returns:
        Trained model and loss history
    """
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    loss_weights = {
        'lambda_pnl': 1.0,
        'lambda_forecast': 1.0,
        'lambda_arbitrage': 0.1,
        'lambda_iv': 1.0
    }
    
    train_step = create_train_step(model, optimizer, loss_weights)
    losses = []
    
    for epoch in range(n_epochs):
        key, subkey = jr.split(key)
        batch = generate_training_batch(
            subkey, 
            batch_size=batch_size,
            step_size=model.step_size,
            depth=model.depth
        )
        
        model, opt_state, loss = train_step(model, opt_state, batch)
        losses.append(float(loss))
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss:.6f}")
    
    return model, losses


# =============================================================================
# PART 7: INFERENCE AND PRICING
# =============================================================================

def price_option(
    model: NeuralRDE,
    path: jnp.ndarray,
    strike: float,
    spot: float,
    tau: float,
    implied_vol_atm: float
) -> dict:
    """
    Price a short-dated option using the trained Neural RDE.
    
    Args:
        model: Trained NeuralRDE
        path: Recent price/vol history, shape (n_steps, 3)
        strike: Option strike
        spot: Current spot price
        tau: Time to maturity (years)
        implied_vol_atm: ATM implied volatility (for normalization)
        
    Returns:
        Dictionary with pricing outputs
    """
    # Compute standardized moneyness
    # z_+ = (ln(K/S) + 0.5 * I² * τ) / (I * √τ)
    moneyness = (jnp.log(strike / spot) + 0.5 * implied_vol_atm**2 * tau) / (implied_vol_atm * jnp.sqrt(tau))
    
    # Compute log-signatures
    logsigs = compute_logsignatures_for_intervals(path, model.step_size, model.depth)
    
    # Forward pass
    outputs = model(path, logsigs, moneyness, tau)
    
    # Extract results
    implied_vol = outputs['implied_vol']
    greeks_raw = outputs['greeks']
    
    # Parse Greeks
    greeks = SignatureGreeks(
        theta=greeks_raw[0],
        delta=greeks_raw[1],
        gamma=greeks_raw[2],
        vega=greeks_raw[3],
        libra=greeks_raw[4]
    )
    
    # Compute option price via Black-Scholes with model's IV
    price = black_scholes_call(spot, strike, tau, implied_vol, 0.0)
    
    return {
        'price': price,
        'implied_vol': implied_vol,
        'greeks': greeks,
        'sigma2_forecast': outputs['sigma2'],
        'gamma_forecast': outputs['gamma'],
        'omega2_forecast': outputs['omega2'],
        'hidden_state': outputs['hidden_state']
    }


def black_scholes_call(S, K, T, sigma, r):
    """Black-Scholes call option price."""
    d1 = (jnp.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    
    call_price = S * jax.scipy.stats.norm.cdf(d1) - K * jnp.exp(-r * T) * jax.scipy.stats.norm.cdf(d2)
    return call_price


def compute_smile_surface(
    model: NeuralRDE,
    path: jnp.ndarray,
    spot: float,
    strikes: jnp.ndarray,
    taus: jnp.ndarray,
    implied_vol_atm: float = 0.2
) -> jnp.ndarray:
    """
    Compute implied volatility surface using the Neural RDE.
    
    Args:
        model: Trained NeuralRDE
        path: Recent price/vol history
        spot: Current spot price
        strikes: Array of strikes
        taus: Array of maturities
        implied_vol_atm: ATM implied vol for normalization
        
    Returns:
        IV surface of shape (len(strikes), len(taus))
    """
    logsigs = compute_logsignatures_for_intervals(path, model.step_size, model.depth)
    
    iv_surface = jnp.zeros((len(strikes), len(taus)))
    
    for i, K in enumerate(strikes):
        for j, tau in enumerate(taus):
            moneyness = (jnp.log(K / spot) + 0.5 * implied_vol_atm**2 * tau) / (implied_vol_atm * jnp.sqrt(tau))
            outputs = model(path, logsigs, moneyness, tau)
            iv_surface = iv_surface.at[i, j].set(outputs['implied_vol'])
    
    return iv_surface


# =============================================================================
# PART 8: MAIN EXAMPLE
# =============================================================================

def main():
    """
    Main example demonstrating the Neural RDE framework for short-dated options.
    """
    print("=" * 70)
    print("Neural RDE for Short-Dated Options Pricing")
    print("=" * 70)
    
    # Set random seed
    key = jr.PRNGKey(42)
    
    # Model configuration
    input_dim = 3      # (time, log_price, implied_vol)
    hidden_dim = 64    # Hidden state dimension
    step_size = 32     # Observations per log-signature interval
    depth = 2          # Log-signature depth
    
    # Compute log-signature dimension
    logsig_dim = logsignature_dimension(input_dim, depth)
    print(f"\nModel Configuration:")
    print(f"  Input dimension: {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Step size: {step_size}")
    print(f"  Log-signature depth: {depth}")
    print(f"  Log-signature dimension: {logsig_dim}")
    
    # Initialize model
    key, model_key = jr.split(key)
    model = NeuralRDE(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        logsig_dim=logsig_dim,
        step_size=step_size,
        depth=depth,
        key=model_key
    )
    
    print(f"\nModel initialized successfully!")
    
    # Generate sample path
    print("\n" + "-" * 50)
    print("Generating sample jump-diffusion path...")
    key, path_key = jr.split(key)
    path, info = generate_jump_diffusion_path(
        path_key,
        n_steps=1000,      # ~2.5 days of minute data
        lambda_jump=10.0,  # Higher jump intensity for short-dated
        mu_jump=-0.01,
        sigma_jump=0.02
    )
    
    print(f"  Path shape: {path.shape}")
    print(f"  Number of jumps: {info['n_jumps']}")
    print(f"  Realized σ²: {info['realized_sigma2']:.4f}")
    print(f"  Realized γ: {info['realized_gamma']:.6f}")
    print(f"  Realized ω²: {info['realized_omega2']:.6f}")
    
    # Compute log-signatures
    print("\n" + "-" * 50)
    print("Computing log-signatures...")
    logsigs = compute_logsignatures_for_intervals(path, step_size, depth)
    print(f"  Number of intervals: {logsigs.shape[0]}")
    print(f"  Log-signature dimension: {logsigs.shape[1]}")
    
    # Forward pass (untrained)
    print("\n" + "-" * 50)
    print("Forward pass (untrained model)...")
    outputs = model(path, logsigs, moneyness=0.0, tau=1/252)
    print(f"  Predicted σ²: {outputs['sigma2']:.4f}")
    print(f"  Predicted γ: {outputs['gamma']:.6f}")
    print(f"  Predicted ω²: {outputs['omega2']:.4f}")
    print(f"  Predicted IV: {outputs['implied_vol']:.4f}")
    print(f"  Greeks: {outputs['greeks']}")
    
    # Training
    print("\n" + "-" * 50)
    print("Training Neural RDE...")
    key, train_key = jr.split(key)
    
    trained_model, losses = train_model(
        model,
        n_epochs=50,
        batch_size=16,
        learning_rate=1e-3,
        key=train_key,
        verbose=True
    )
    
    print(f"\nTraining complete!")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Loss reduction: {(losses[0] - losses[-1]) / losses[0] * 100:.1f}%")
    
    # Forward pass (trained)
    print("\n" + "-" * 50)
    print("Forward pass (trained model)...")
    outputs = trained_model(path, logsigs, moneyness=0.0, tau=1/252)
    print(f"  Predicted σ²: {outputs['sigma2']:.4f} (true: {info['realized_sigma2']:.4f})")
    print(f"  Predicted γ: {outputs['gamma']:.6f} (true: {info['realized_gamma']:.6f})")
    print(f"  Predicted ω²: {outputs['omega2']:.4f} (true: {info['realized_omega2']:.6f})")
    print(f"  Predicted IV: {outputs['implied_vol']:.4f}")
    
    # Price a short-dated option
    print("\n" + "-" * 50)
    print("Pricing a short-dated option...")
    spot = jnp.exp(path[-1, 1])
    pricing_result = price_option(
        trained_model,
        path,
        strike=spot * 1.01,  # 1% OTM call
        spot=spot,
        tau=1/252,           # 1-day option
        implied_vol_atm=0.2
    )
    
    print(f"  Spot: {spot:.2f}")
    print(f"  Strike: {spot * 1.01:.2f}")
    print(f"  Maturity: 1 day")
    print(f"  Implied Vol: {pricing_result['implied_vol']:.4f}")
    print(f"  Price: {pricing_result['price']:.4f}")
    print(f"  Greeks:")
    print(f"    Theta: {pricing_result['greeks'].theta:.6f}")
    print(f"    Delta: {pricing_result['greeks'].delta:.6f}")
    print(f"    Gamma: {pricing_result['greeks'].gamma:.6f}")
    print(f"    Vega: {pricing_result['greeks'].vega:.6f}")
    print(f"    Libra: {pricing_result['greeks'].libra:.6f}")
    
    # Compute Lévy area for insight
    print("\n" + "-" * 50)
    print("Path Statistics (Signature Analysis)...")
    levy_area_full = compute_levy_area(path[:, 1:])  # Price-vol Lévy area
    print(f"  Lévy area (price-vol): {levy_area_full[0, 1]:.6f}")
    print(f"  This measures the 'order' of price-vol movements")
    print(f"  Positive: price moves before vol increases")
    print(f"  Negative: vol moves before price")
    
    print("\n" + "=" * 70)
    print("Neural RDE demonstration complete!")
    print("=" * 70)
    
    return trained_model, losses


if __name__ == "__main__":
    trained_model, losses = main()
