# neural_rde_options.py - Core Mathematics Module

## Overview

This is the foundational module implementing Patrick Kidger's Neural Rough Differential Equation (RDE) framework for pricing short-dated options. It provides the mathematical core for signature computation, neural network architecture, training, and inference.

**Location**: `neural_rde_options.py`
**Lines**: 1496
**Dependencies**: JAX, Equinox, Diffrax, Optax

---

## Theoretical Foundation

The module implements concepts from three key papers:
1. **Morrill, Salvi, Kidger, Foster, Lyons (2021)**: "Neural Rough Differential Equations"
2. **Dupire, Tissot-Daguette**: "Signature and the Functional Taylor Expansion"
3. **Carr, Wu**: "Option Profit and Loss Attribution and Pricing"

### Core Idea
Traditional Black-Scholes fails for short-dated options (< 1 week) because:
- Jump risk dominates
- Path-dependence matters (order of movements affects price)
- Rough volatility behavior at short horizons

The Neural RDE solves this by:
1. Compressing high-frequency paths via **log-signatures**
2. Learning path-to-price mappings via **controlled differential equations**
3. Extracting novel Greeks including **Libra** (Lévy area sensitivity)

---

## Module Structure

### Part 1: Signature and Log-Signature Computation (Lines 34-310)

#### `compute_signature_depth1(path)` (Line 38)
Computes depth-1 signature (simple increments).
```
S^i = X^i_T - X^i_0
```
- **Input**: path of shape `(length, channels)`
- **Output**: increments of shape `(channels,)`

#### `compute_signature_depth2(path)` (Line 51)
Computes depth-2 signature terms (iterated integrals).
```
S^{i,j} = ∫∫_{s<t} dX^i_s dX^j_t
```
- Uses cumulative sum for efficient computation
- **Output**: matrix of shape `(channels, channels)`

#### `compute_levy_area(path)` (Line 82)
Extracts the **Lévy area** - the antisymmetric part of depth-2 signature.
```
A^{ij} = (S^{i,j} - S^{j,i}) / 2
```
**Key insight**: The Lévy area measures the "signed area" swept by the path projection onto the (i,j) plane. It captures the **ORDER** of movements - crucial for short-dated options where whether price moves before volatility matters.

#### `compute_signature(path, depth)` (Line 102)
Computes truncated signature up to specified depth (1, 2, or 3).
- Concatenates depth-1, depth-2, and optionally depth-3 terms
- **Output**: flattened signature vector

#### `_compute_signature_depth3(path)` (Line 140)
Computes depth-3 signature (triple iterated integrals).
```
S^{i,j,k} = ∫∫∫_{r<s<t} dX^i_r dX^j_s dX^k_t
```
- Uses `lax.scan` for efficient sequential computation
- Maintains running sums for nested integrals

#### `compute_logsignature(path, depth)` (Line 175)
Computes the **log-signature** - a compressed representation removing algebraic redundancies.
- Depth 1: increments (same as signature)
- Depth 2: Lévy areas only (antisymmetric parts)
- Depth 3: Hall basis elements (Lyndon words)

**Why log-signature?**: Reduces dimensionality while preserving all path information needed for CDEs. For 5 channels at depth 2: signature has 5 + 25 = 30 terms, log-signature has 5 + 10 = 15 terms.

#### `logsignature_dimension(d, depth)` (Line 238)
Returns the dimension of log-signature space using Möbius function formula:
```
β(d, M) = Σ_{k=1}^{M} (1/k) Σ_{j|k} μ(k/j) d^j
```
| Depth | Dimension Formula |
|-------|-------------------|
| 1 | d |
| 2 | d + d(d-1)/2 |
| 3 | d + d(d-1)/2 + d(d-1) |

#### `compute_logsignatures_for_intervals(path, step_size, depth)` (Line 266)
**Key preprocessing step**: Splits path into non-overlapping intervals and computes log-signature for each.
- **Input**: Full path of shape `(length, channels)`
- **Output**: Sequence of log-signatures, shape `(n_intervals, logsig_dim)`

This is how 1000s of observations get compressed to ~30 log-signature vectors.

#### `batch_compute_logsignatures(paths, step_size, depth)` (Line 302)
Vectorized version using `jax.vmap` for batch processing.

---

### Part 2: Neural RDE Architecture (Lines 312-666)

#### `VectorField` Class (Line 316)
Neural network that maps hidden state to a matrix for CDE integration.

**Architecture**:
```
Z ∈ ℝ^hidden_dim → MLP → reshape → ℝ^{hidden_dim × logsig_dim}
```

**Parameters**:
- `hidden_dim`: Dimension of hidden state
- `logsig_dim`: Dimension of log-signature
- `width`: MLP layer width (default 128)
- `depth`: Number of MLP layers (default 3)
- `activation`: GELU

**Forward pass** (`__call__`):
```python
x = z
for layer in layers[:-1]:
    x = gelu(layer(x))
x = layers[-1](x)
return x.reshape(hidden_dim, logsig_dim)
```

#### `InitialEncoder` Class (Line 379)
Encodes initial observation to initial hidden state.

**Architecture**:
```
X_0 ∈ ℝ^input_dim → MLP(3 layers) → Z_0 ∈ ℝ^hidden_dim
```

#### `OutputDecoder` Class (Line 411)
Multi-head decoder extracting predictions from hidden state.

**Heads**:
| Head | Input | Output | Activation |
|------|-------|--------|------------|
| `vol_head` | hidden_state + moneyness + tau | implied_vol | softplus |
| `sigma2_head` | hidden_state | σ² (variance) | softplus |
| `gamma_head` | hidden_state | γ (return-vol cov) | linear |
| `omega2_head` | hidden_state | ω² (vol-of-vol) | softplus |
| `greeks_head` | hidden_state | [Θ, Δ, Γ, V, L] | linear |

#### `NeuralRDE` Class (Line 482)
**The main model class** implementing the Neural RDE.

**The Log-ODE Method**:
```
Z_t = Z_0 + ∫_0^t f_θ(Z_s) · (LogSig / Δt) ds
```

Where:
- `Z_t`: Hidden state encoding portfolio risk profile
- `f_θ`: Learned vector field (VectorField class)
- `LogSig`: Log-signature over each interval

**Components**:
- `encoder`: InitialEncoder
- `vector_field`: VectorField
- `decoder`: OutputDecoder

**Forward pass** (`__call__`, line 546):
1. Compute log-signatures if not provided
2. Initialize hidden state from first observation: `z0 = encoder(x0)`
3. Integrate via log-ODE method
4. Decode outputs

**Integration** (`_integrate`, line 590):
Uses `lax.scan` with Euler scheme:
```python
def step_fn(z, logsig):
    f_z = vector_field(z)      # (hidden_dim, logsig_dim)
    dz = f_z @ logsig          # Matrix-vector product
    z_new = z + dz * dt        # Euler update
    return z_new, z_new
```

**Diffrax Integration** (`integrate_diffrax`, line 628):
Higher-accuracy integration using Tsit5 (5th order Runge-Kutta) solver.

---

### Part 3: Signature-Based P&L Attribution (Lines 668-802)

#### `SignatureGreeks` NamedTuple (Line 672)
Defines the five Greeks derived from signature decomposition:

| Greek | Symbol | Meaning | Signature Term |
|-------|--------|---------|----------------|
| Theta | Θ | Time decay | S^(0) |
| Delta | Δ | Spot sensitivity | S^(1) |
| Gamma | Γ | Spot convexity | S^(1,1) |
| Vega | V | Vol sensitivity | S^(2) |
| **Libra** | L | **Lévy area sensitivity** | A^(1,2) |

**Libra is novel** - it measures sensitivity to the Lévy area (order of movements).

#### `decompose_pnl_signature(pnl, path, greeks)` (Line 692)
Implements the **Functional Taylor Expansion**:
```
P&L = f(X*Y) - f(X)
    ≈ Θ·δt + Δ·δy + Γ·(δy²/2) + V·δσ + L·A(Y) + residual
```

Returns attribution dictionary with each component's contribution.

#### `compute_libra_greek(pricing_fn, path, h, dt)` (Line 745)
Computes Libra via **finite differences**:
```
L = lim_{h,δt→0} [f((X^h)^{*δt}) - f((X^{*δt})^h)] / (h·δt)
```

This measures the difference between:
- Bumping price first, then extending time
- Extending time first, then bumping price

Non-zero only for genuinely path-dependent functionals.

---

### Part 4: Loss Functions and Training (Lines 804-961)

#### `compute_pnl_loss(model, path, logsigs, realized_pnl)` (Line 808)
Loss for P&L prediction accuracy. Combines Greeks with signature terms:
```
predicted_pnl = Θ·δt + Δ·δy + Γ·(δy²/2) + V·δσ + L·A
loss = MSE(predicted_pnl, realized_pnl)
```

#### `compute_forecast_loss(model, path, logsigs, realized_*)` (Line 854)
Loss for moment condition forecasting:
```
loss = MSE(σ²_pred, σ²_true) + MSE(γ_pred, γ_true) + MSE(ω²_pred, ω²_true)
```

#### `compute_no_arbitrage_loss(model, path, logsigs)` (Line 879)
**Soft constraint for no-arbitrage**: Under risk-neutral measure, hidden state drift should be zero.
```
loss = mean(dZ²)
```

#### `compute_implied_vol_loss(model, ...)` (Line 904)
MSE between predicted and market implied volatilities.

#### `total_loss(model, batch, lambda_*)` (Line 923)
Combined loss with configurable weights:
```
L = λ_pnl·L_pnl + λ_forecast·L_forecast + λ_arbitrage·L_arbitrage + λ_iv·L_iv
```

---

### Part 5: Data Generation - Synthetic (Lines 963-1122)

#### `generate_jump_diffusion_path(key, ...)` (Line 967)
Generates paths from a **Bates-like jump-diffusion model**:
```
dS/S = σ dW^S + (e^J - 1) dN - λE[e^J - 1] dt
dσ = κ(θ - σ) dt + ξ σ dW^σ
```

**Parameters**:
| Parameter | Default | Meaning |
|-----------|---------|---------|
| `n_steps` | 1000 | Number of time steps |
| `dt` | 1/252/390 | Time step (~1 minute) |
| `S0` | 100.0 | Initial spot |
| `sigma0` | 0.2 | Initial volatility |
| `kappa` | 2.0 | Vol mean reversion speed |
| `theta` | 0.2 | Vol long-term mean |
| `xi` | 0.3 | Vol-of-vol |
| `rho` | -0.7 | Spot-vol correlation |
| `lambda_jump` | 5.0 | Jump intensity (per year) |
| `mu_jump` | -0.02 | Mean jump size |
| `sigma_jump` | 0.03 | Jump size std |

**Implementation**:
1. Generate correlated Brownian increments with Cholesky decomposition
2. Generate Poisson jumps
3. Euler-Maruyama simulation via `lax.scan`
4. Return path `(t, log_S, σ)` and realized quantities

#### `generate_training_batch(key, batch_size, ...)` (Line 1069)
Generates complete training batches with:
- Paths
- Pre-computed log-signatures
- Realized moment conditions
- Random option characteristics (moneyness, tau)
- Target implied volatilities

---

### Part 6: Training Loop (Lines 1124-1239)

#### `create_train_step(model, optimizer, loss_weights)` (Line 1128)
Creates JIT-compiled training step using `eqx.filter_jit`.

**Training step flow**:
1. Compute loss over batch
2. Compute gradients via `eqx.filter_value_and_grad`
3. Update parameters via optimizer
4. Return updated model, optimizer state, loss

#### `total_loss_single(model, sample, lambda_*)` (Line 1157)
Computes loss for a single sample (used in batch loop).

#### `train_model(model, n_epochs, batch_size, learning_rate, key)` (Line 1188)
Main training function:
1. Initialize Adam optimizer
2. For each epoch:
   - Generate training batch
   - Execute train step
   - Record loss
3. Return trained model and loss history

---

### Part 7: Inference and Pricing (Lines 1241-1346)

#### `price_option(model, path, strike, spot, tau, implied_vol_atm)` (Line 1245)
Prices a short-dated option using trained Neural RDE.

**Steps**:
1. Compute standardized moneyness:
   ```
   z_+ = (ln(K/S) + 0.5·I²·τ) / (I·√τ)
   ```
2. Compute log-signatures
3. Forward pass through model
4. Extract implied vol and Greeks
5. Compute price via Black-Scholes with model's IV

**Returns**: price, implied_vol, SignatureGreeks, forecasts, hidden_state

#### `black_scholes_call(S, K, T, sigma, r)` (Line 1304)
Standard Black-Scholes call formula using JAX's `scipy.stats.norm.cdf`.

#### `compute_smile_surface(model, path, spot, strikes, taus, ...)` (Line 1313)
Computes full implied volatility surface across strikes and maturities.

---

### Part 8: Main Example (Lines 1348-1496)

Demonstrates the complete workflow:

1. **Configuration**: Set input_dim=3, hidden_dim=64, step_size=32, depth=2
2. **Initialize model**: Create NeuralRDE with computed logsig_dim
3. **Generate sample path**: Jump-diffusion with 1000 steps
4. **Compute log-signatures**: ~31 intervals of dimension 6
5. **Forward pass (untrained)**: Show initial predictions
6. **Training**: 50 epochs, batch_size=16
7. **Forward pass (trained)**: Compare to true values
8. **Price option**: 1-day 1% OTM call
9. **Lévy area analysis**: Interpret path-dependence

---

## Key Mathematical Insights

### Why Log-Signatures Work

The signature of a path uniquely characterizes the path up to reparametrization. The log-signature is the "optimal" compression for CDEs:

```
dZ = f(Z) dX  →  Z_T ≈ exp(f(Z_0) · LogSig(X))
```

### The Libra Greek

Traditional Greeks (Δ, Γ, Θ, V) assume price is a function of state variables. Libra captures **path-dependence**:

- If Libra ≈ 0: Option is essentially path-independent
- If Libra ≠ 0: Order of price/vol movements matters

For short-dated options, Libra becomes significant because there's less time for path effects to average out.

### Lévy Area Interpretation

```
A^{price,vol} > 0: Price moves before vol increases (vol chasing price)
A^{price,vol} < 0: Vol moves before price (anticipatory vol)
```

This asymmetry affects option P&L in ways Black-Scholes cannot capture.

---

## Usage Example

```python
import jax.random as jr
from neural_rde_options import NeuralRDE, generate_jump_diffusion_path, price_option

# Initialize
key = jr.PRNGKey(42)
model = NeuralRDE(input_dim=3, hidden_dim=64, logsig_dim=6, key=key)

# Generate path
path, info = generate_jump_diffusion_path(key, n_steps=1000)

# Price option
result = price_option(model, path, strike=101, spot=100, tau=1/252, implied_vol_atm=0.2)
print(f"Price: {result['price']:.4f}")
print(f"Libra: {result['greeks'].libra:.6f}")
```

---

## Function Reference Table

| Function | Line | Purpose |
|----------|------|---------|
| `compute_signature_depth1` | 38 | Depth-1 signature (increments) |
| `compute_signature_depth2` | 51 | Depth-2 signature (iterated integrals) |
| `compute_levy_area` | 82 | Antisymmetric part of depth-2 |
| `compute_signature` | 102 | Full truncated signature |
| `compute_logsignature` | 175 | Compressed log-signature |
| `logsignature_dimension` | 238 | Dimension formula |
| `compute_logsignatures_for_intervals` | 266 | Interval-wise computation |
| `VectorField` | 316 | Neural network for CDE |
| `InitialEncoder` | 379 | Initial state encoder |
| `OutputDecoder` | 411 | Multi-head decoder |
| `NeuralRDE` | 482 | Main model class |
| `SignatureGreeks` | 672 | Greeks named tuple |
| `decompose_pnl_signature` | 692 | P&L attribution |
| `compute_libra_greek` | 745 | Libra via finite differences |
| `compute_pnl_loss` | 808 | P&L loss function |
| `compute_forecast_loss` | 854 | Forecast loss function |
| `compute_no_arbitrage_loss` | 879 | No-arbitrage regularizer |
| `total_loss` | 923 | Combined loss |
| `generate_jump_diffusion_path` | 967 | Synthetic data generation |
| `generate_training_batch` | 1069 | Batch generation |
| `create_train_step` | 1128 | JIT-compiled train step |
| `train_model` | 1188 | Training loop |
| `price_option` | 1245 | Option pricing |
| `black_scholes_call` | 1304 | BS formula |
| `compute_smile_surface` | 1313 | IV surface computation |
| `main` | 1352 | Example demonstration |
