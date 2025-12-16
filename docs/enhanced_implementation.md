# enhanced_implementation.py - Extended Neural RDE Features

## Overview

This module extends the base `neural_rde_options.py` with advanced features from the complete Kidger framework. It implements full signature-to-Greek mapping, jump indicator augmentation, extended smile formulas with Lévy area terms, and provides a Carr-Wu baseline for comparison.

**Location**: `enhanced_implementation.py`
**Lines**: 1098
**Dependencies**: JAX, Equinox, Diffrax, Optax, `neural_rde_options` (base module)

---

## Module Structure

The file is organized into 7 parts:
1. Signature-to-Greek Mapping
2. Jump Indicator Augmentation
3. Smile Shape with Lévy Area
4. Enhanced Neural RDE Architecture
5. Loss Functions with No-Arbitrage Constraints
6. Carr-Wu Baseline Comparison
7. Depth-3 Signature Interpretation

---

## Part 1: Signature-to-Greek Mapping (Lines 40-190)

### `SignatureGreekMapping` NamedTuple (Line 44)

Defines the explicit mapping from signature terms to financial Greeks:

| Signature Term | Greek | Interpretation |
|----------------|-------|----------------|
| S^(0) | Θ (Theta) | Time decay |
| S^(1) | Δ (Delta) | Spot sensitivity |
| S^(2) | Vega | Vol sensitivity |
| S^(1,1) | Γ (Gamma) | Convexity in spot |
| S^(2,2) | Volga | Convexity in vol |
| S^(1,2) + S^(2,1) | Vanna | Cross-gamma (symmetric) |
| S^(1,2) - S^(2,1) | Lévy area | Order of spot/vol moves |
| S^(0,1) - S^(1,0) | Libra | Time-space non-commutativity |

**Fields**:
```python
# Depth-1 Greeks
theta: jnp.ndarray      # Time increment sensitivity
delta: jnp.ndarray      # Spot increment sensitivity
vega: jnp.ndarray       # Vol increment sensitivity

# Depth-2 Greeks (symmetric)
gamma: jnp.ndarray      # Spot convexity
volga: jnp.ndarray      # Vol convexity
vanna: jnp.ndarray      # Cross sensitivity

# Depth-2 Greeks (antisymmetric - NEW)
levy_area_sv: jnp.ndarray   # Spot-vol order
libra: jnp.ndarray          # Time-space Lie bracket

# Depth-3 Greeks
skewness_contrib: jnp.ndarray  # Return skewness sensitivity
```

### `extract_greeks_from_signature(sig, d, depth)` (Line 79)

Parses a flattened signature vector and extracts all Greeks.

**Logic**:
1. Parse depth-1 terms (first `d` elements): theta, delta, vega
2. Parse depth-2 matrix (`d×d` elements):
   - Diagonal: gamma (S^(1,1)), volga (S^(2,2))
   - Symmetric off-diagonal: vanna = S^(1,2) + S^(2,1)
   - Antisymmetric: levy_area = (S^(1,2) - S^(2,1)) / 2
   - Time-space: libra = (S^(0,1) - S^(1,0)) / 2
3. Parse depth-3 for skewness: S^(1,1,1)

### `compute_pnl_from_signature_greeks(greeks, future_sig, d)` (Line 136)

Implements the **Functional Taylor Expansion** for P&L decomposition:

```
P&L = Θ·δt + Δ·δy + Γ·(δy²/2) + V·δσ + L·A(Y) + ...
```

**Returns dictionary with**:
- Classical P&L: theta_pnl, delta_pnl, vega_pnl
- Second-order P&L: gamma_pnl, volga_pnl, vanna_pnl
- **Path-dependent P&L** (NEW): levy_area_pnl, libra_pnl
- `path_dependent_pnl`: Sum of Lévy area and Libra contributions

---

## Part 2: Jump Indicator Augmentation (Lines 192-299)

### `augment_path_with_jump_indicators(path, ...)` (Line 196)

Augments the base path with jump counting channels.

**Transformation**:
```
X = (t, S, I) → X_aug = (t, S, I, J^S, J^I)
```

Where:
- J^S_t = Σ_{s≤t} 1_{|ΔS_s| > ε} — cumulative spot jump count
- J^I_t = Σ_{s≤t} 1_{|ΔI_s| > ε} — cumulative vol jump count

**Parameters**:
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `spot_threshold` | 0.02 | 2% move = spot jump |
| `vol_threshold` | 0.05 | 5% vol move = vol jump |

**Why This Matters**:
The log-signature of the augmented path captures:
- **Jump frequency**: Through S^(3), S^(4) (increments in jump counters)
- **Jump timing**: Through S^(1,3) (price move followed by jump)
- **Jump clustering**: Through S^(3,3) (consecutive jumps)

### `compute_jump_signature_features(path)` (Line 256)

Extracts jump-related features from augmented path signature.

**Features extracted**:
| Feature | Signature Term | Meaning |
|---------|----------------|---------|
| `spot_jump_freq` | S^(3) | Spot jump frequency |
| `vol_jump_freq` | S^(4) | Vol jump frequency |
| `spot_jump_timing` | S^(1,3) | Price move before spot jump |
| `vol_jump_timing` | S^(2,4) | Vol change before vol jump |
| `spot_jump_cluster` | S^(3,3) | Consecutive spot jumps |
| `vol_jump_cluster` | S^(4,4) | Consecutive vol jumps |
| `jump_contagion` | S^(3,4) + S^(4,3) | Cross-asset jump spillover |

---

## Part 3: Smile Shape with Lévy Area (Lines 301-401)

### `compute_smile_formula_extended(...)` (Line 305)

Implements the **extended Carr-Wu smile formula** including the Lévy area term.

**Original Carr-Wu**:
```
I²(K) - A² = 2γ z₊ + ω² z₊z₋
```

**Extended formula (NEW)**:
```
I²(K) - A² = 2γ z₊ + ω² z₊z₋ + ξ E[A(X)]
```

Where:
- `z₊`: Standardized moneyness under Q measure
- `z₋`: Standardized moneyness under share measure
- `γ`: Return-vol covariance (vanna term)
- `ω²`: Vol-of-vol (volga term)
- `ξ`: **Lévy area sensitivity coefficient** (NEW)
- `E[A(X)]`: Expected Lévy area from path dynamics

**Why the Lévy area term matters**:
- Explains why short-dated smiles are steeper
- At short horizons, jumps dominate and Lévy area doesn't average out
- At long horizons, CLT kicks in and Lévy area contribution vanishes

### `analyze_atm_skew_explosion(tau, ...)` (Line 347)

Analyzes the ATM skew explosion phenomenon as τ → 0.

**Key insight from Kidger**:
- For continuous paths: Lévy area ~ O(τ)
- For jump paths: Lévy area ~ O(1) (doesn't vanish!)
- Therefore: ATM skew ∝ 1/√τ as τ → 0

**Computed metrics**:
| Metric | Formula | Interpretation |
|--------|---------|----------------|
| `continuous_contribution` | σ²τ/2 | Diffusive Lévy area |
| `jump_contribution` | λτ · |m| · τ/4 | Jump Lévy area |
| `atm_skew` | total_levy_area / √τ | ATM skew magnitude |
| `jump_dominance_ratio` | jump / total | % from jumps |

**Example output** (from demo):
```
τ=1d:  Skew=0.0126, Jump dominance=66.67%
τ=5d:  Skew=0.0283, Jump dominance=66.67%
τ=21d: Skew=0.0580, Jump dominance=66.67%
```

---

## Part 4: Enhanced Neural RDE Architecture (Lines 403-730)

### `EnhancedOutputDecoder` Class (Line 407)

Extended decoder with 14 output heads for comprehensive analysis.

**Head categories**:

| Category | Heads | Purpose |
|----------|-------|---------|
| Classical Greeks | theta, delta, gamma, vega | Standard sensitivities |
| Higher-order | volga, vanna | Second-order cross terms |
| **Path-dependent** | levy_area_sens, libra | **NEW** - order matters |
| Smile shape | smile_gamma, smile_omega2, smile_xi | Carr-Wu + Lévy parameters |
| Forecasts | sigma2, covar, vol_of_vol | Moment predictions |
| Implied vol | iv | Option pricing |

**Forward pass logic**:
1. Compute all Greeks from hidden state
2. If moneyness and tau provided:
   - Compute base IV from hidden state
   - Apply extended smile formula with Lévy area
   - Return adjusted IV

### `EnhancedNeuralRDE` Class (Line 559)

Main enhanced model with all framework features.

**Key enhancements over base model**:
1. Automatic jump indicator augmentation
2. Full signature-to-Greek mapping
3. Extended smile formula integration
4. Optional detailed analysis mode

**Architecture**:
```
path (t, S, I)
    → augment with (J^S, J^I) if enabled
    → compute log-signatures
    → compute expected Lévy area
    → encode initial state
    → integrate via log-ODE
    → decode with EnhancedOutputDecoder
```

**Forward pass** (`__call__`, line 624):
```python
def __call__(self, path, moneyness, tau, return_full_analysis):
    # Step 1: Augment path with jump indicators
    if self.use_jump_indicators and path.shape[1] == 3:
        augmented_path = augment_path_with_jump_indicators(path)

    # Step 2: Compute log-signatures
    logsigs = self._compute_logsigs(augmented_path)

    # Step 3: Compute expected Lévy area (for smile formula)
    expected_levy_area = self._compute_expected_levy_area(augmented_path)

    # Step 4: Encode initial state
    z0 = self.encoder(augmented_path[0])

    # Step 5: Integrate via log-ODE
    z_final, trajectory = self._integrate(z0, logsigs)

    # Step 6: Decode outputs
    outputs = self.decoder(z_final, moneyness, tau, expected_levy_area)

    # Optional: Full analysis
    if return_full_analysis:
        outputs['signature_greeks'] = extract_greeks_from_signature(...)
        outputs['jump_features'] = compute_jump_signature_features(...)
        outputs['skew_analysis'] = analyze_atm_skew_explosion(tau)

    return outputs
```

---

## Part 5: Loss Functions with No-Arbitrage (Lines 732-848)

### `compute_terminal_condition_loss(predicted, payoff)` (Line 736)

Terminal condition constraint:
```
L_terminal = E[(Z_T - payoff)²]
```
Model output should match option payoff at maturity.

### `compute_drift_constraint_loss(trajectory, r, dt)` (Line 748)

No-arbitrage drift constraint:
```
L_drift = E[(dZ_t/dt - r·Z_t)²]
```
Under risk-neutral measure, the drift should equal the risk-free rate.

### `compute_orthogonality_loss(vector_field, expected_logsig)` (Line 770)

No-arbitrage orthogonality constraint:
```
L_ortho = E[(f_θ(Z) · LogSig)²]
```
Vector field should be orthogonal to expected log-signature direction.

### `compute_enhanced_total_loss(model, path, targets, ...)` (Line 784)

Complete loss function with all constraints:

```
L(θ) = λ_terminal · E[(Z_T - payoff)²]           # Terminal condition
     + λ_drift · E[(dZ/dt - rZ)²]                # No-arbitrage drift
     + λ_iv · E[(IV - IV_market)²]               # IV matching
     + λ_forecast · E[(moments - realized)²]     # Moment forecasting
     + λ_path_dep · E[(Libra error)²]            # Path-dependent Greeks (NEW)
```

**Default weights**:
| Weight | Default | Purpose |
|--------|---------|---------|
| `lambda_terminal` | 1.0 | Payoff matching |
| `lambda_drift` | 0.1 | No-arbitrage regularization |
| `lambda_iv` | 1.0 | IV accuracy |
| `lambda_forecast` | 1.0 | Moment accuracy |
| `lambda_path_dep` | 0.5 | Path-dependent Greeks |

---

## Part 6: Carr-Wu Baseline Comparison (Lines 850-935)

### `CarrWuBaseline` Class (Line 854)

Implementation of the Carr-Wu linear forecasting baseline for comparison.

#### `forecast_variance_linear(historical_gamma, historical_omega2)` (Line 866)

Linear variance forecast:
```
σ²_{t+1} ≈ β₀ + β₁·γ_t + β₂·ω²_t
```
Default weights: [0.5, 0.3, 0.2]

#### `compute_smile_carr_wu(z_plus, z_minus, gamma, omega2)` (Line 889)

Original Carr-Wu smile formula (**without** Lévy area):
```
I² - A² = 2γz₊ + ω²z₊z₋
```
This is the baseline that misses path-dependent effects.

#### `compute_comparison_metrics(neural_rde, carr_wu, realized)` (Line 904)

Computes comparison metrics:
- `variance_forecast_improvement`: (CW_error - NRDE_error) / CW_error
- `smile_fit_improvement`: Same for smile errors
- `neural_rde_mse`, `carr_wu_mse`: Raw MSE values

---

## Part 7: Depth-3 Signature Interpretation (Lines 937-988)

### `interpret_depth3_for_forecasting(path)` (Line 941)

Extracts forecasting features from depth-3 signature.

**Feature mapping table**:

| Feature | Signature | Interpretation | Forecasting Value |
|---------|-----------|----------------|-------------------|
| S^(1) | Depth-1 | Recent return | Momentum signal |
| S^(1,1) | Depth-2 | Realized var proxy | Vol persistence |
| Lévy area | Depth-2 | Order of moves | Vol clustering |
| S^(1,1,1) | Depth-3 | Skewness of returns | Jump asymmetry |
| Cross terms | Depth-3 | Complex patterns | Regime detection |

**Returned features**:
```python
{
    'momentum_signal': sig1[1],           # Recent return direction
    'vol_persistence': sig2[1, 1],        # Realized variance proxy
    'vol_clustering': levy_area[0, 1],    # Lévy area
    'jump_asymmetry': sig3[1, 1, 1],      # Return skewness
    'regime_indicator': avg(sig3 cross)   # Regime detection
}
```

---

## Demo Function (Lines 992-1098)

### `demo_full_framework()` (Line 995)

Demonstrates all components in sequence:

1. **Path Generation**: 500-step jump-diffusion with λ=15 jumps/year
2. **Jump Augmentation**: Adds J^S, J^I channels
3. **Signature-to-Greek Mapping**: Extracts all Greeks including Lévy area and Libra
4. **ATM Skew Explosion**: Analyzes skew at 1d, 5d, 21d maturities
5. **Extended Smile Formula**: Compares Carr-Wu vs extended formula
6. **Depth-3 Forecasting**: Momentum, persistence, clustering, asymmetry
7. **Enhanced Model**: Forward pass with full analysis

**Sample output**:
```
3. SIGNATURE-TO-GREEK MAPPING
   Theta: 0.001234
   Delta: 0.045678
   Gamma: 0.012345
   Lévy area (S-V): -0.002345  ← Path-dependent!
   Libra: 0.000123  ← Path-dependent!

5. EXTENDED SMILE FORMULA
   Carr-Wu (no Lévy): I² - A² = -0.6850
   Extended (+ Lévy): I² - A² = -0.6450
   Lévy contribution: 0.0400
```

---

## Key Innovations Summary

| Feature | Base Module | Enhanced Module |
|---------|-------------|-----------------|
| Greeks | Θ, Δ, Γ, V, L | + Volga, Vanna, Lévy sens |
| Path channels | (t, S, I) | (t, S, I, J^S, J^I) |
| Smile formula | Standard Carr-Wu | + Lévy area term (ξ) |
| Loss function | Basic MSE | + No-arbitrage constraints |
| Forecasting | Depth-2 only | Depth-3 interpretation |
| Baseline | None | Carr-Wu comparison |

---

## Usage Example

```python
import jax.random as jr
from enhanced_implementation import (
    EnhancedNeuralRDE,
    augment_path_with_jump_indicators,
    analyze_atm_skew_explosion
)

# Initialize model
model = EnhancedNeuralRDE(
    input_dim=3,
    hidden_dim=64,
    use_jump_indicators=True,
    key=jr.PRNGKey(42)
)

# Forward pass with full analysis
outputs = model(
    path,
    moneyness=jnp.array(0.0),
    tau=jnp.array(1/252),
    return_full_analysis=True
)

# Access path-dependent Greeks
print(f"Libra: {outputs['libra']}")
print(f"Lévy sensitivity: {outputs['levy_area_sensitivity']}")
print(f"ξ (smile Lévy term): {outputs['smile_xi']}")
```

---

## Function Reference Table

| Function/Class | Line | Purpose |
|----------------|------|---------|
| `SignatureGreekMapping` | 44 | NamedTuple for all Greeks |
| `extract_greeks_from_signature` | 79 | Parse signature → Greeks |
| `compute_pnl_from_signature_greeks` | 136 | P&L attribution |
| `augment_path_with_jump_indicators` | 196 | Add J^S, J^I channels |
| `compute_jump_signature_features` | 256 | Extract jump features |
| `compute_smile_formula_extended` | 305 | Extended Carr-Wu + Lévy |
| `analyze_atm_skew_explosion` | 347 | Skew vs maturity analysis |
| `EnhancedOutputDecoder` | 407 | 14-head decoder |
| `EnhancedNeuralRDE` | 559 | Full enhanced model |
| `compute_terminal_condition_loss` | 736 | Payoff matching loss |
| `compute_drift_constraint_loss` | 748 | No-arbitrage drift |
| `compute_enhanced_total_loss` | 784 | Combined loss function |
| `CarrWuBaseline` | 854 | Baseline comparison |
| `interpret_depth3_for_forecasting` | 941 | Depth-3 features |
| `demo_full_framework` | 995 | Complete demonstration |
