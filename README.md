# Neural Rough Differential Equations for Short-Dated Options Pricing

A JAX-based implementation of Patrick Kidger's Neural RDE framework for pricing short-dated options, addressing the challenges identified by Carr-Wu in their P&L attribution framework.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   CBOE CSV Files                                                            │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  cboe_preprocessing.py                                              │   │
│   │  • CBOEPathConstructor: builds X = (t, log_S, σ, J^S, J^I)         │   │
│   │  • TrainingDataBuilder: creates batches with log-signatures         │   │
│   │  • SmileExtractor: extracts (γ, ω², ξ) from cross-section          │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  kidger_cboe_integration.py (THE BRIDGE)                            │   │
│   │  • Connects preprocessing → Neural RDE model                        │   │
│   │  • create_training_loop(): sets up model + optimizer                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   ┌──────────────────────────┐    ┌──────────────────────────┐             │
│   │  neural_rde_options.py   │───▶│ enhanced_implementation.py│             │
│   │  • NeuralRDE (base)      │    │ • EnhancedNeuralRDE      │             │
│   │  • Signatures            │    │ • Jump indicators        │             │
│   │  • Training loops        │    │ • Extended smile formula │             │
│   │  • Base Greeks           │    │ • CarrWuBaseline         │             │
│   └──────────────────────────┘    └──────────────────────────┘             │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │  carr_wu_kidger_trading.py                                          │   │
│   │  • CarrWuPortfolioConstructor: vol/skew/smile trades                │   │
│   │  • KidgerTimingEngine: log-signature → timing signals               │   │
│   │  • create_trading_pipeline(): end-to-end CBOE → decisions           │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   Trading Decisions: LONG/SHORT vol, skew, smile trades                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 File Organization

### Core Implementation

| File | Type | Description |
|------|------|-------------|
| **`neural_rde_options.py`** | Core | Base Neural RDE implementation: signatures, log-signatures, Lévy areas, `NeuralRDE` class, training loops, pricing functions |
| **`enhanced_implementation.py`** | Core | Extended features: signature-to-Greek mapping, jump indicators, extended smile formula, `EnhancedNeuralRDE`, no-arbitrage constraints |

### CBOE Data Pipeline

| File | Type | Description |
|------|------|-------------|
| **`cboe_preprocessing.py`** | Implementation | **The workhorse** — `CBOEPathConstructor`, `SmileExtractor`, `TrainingDataBuilder`, path construction X=(t, log_S, σ, J^S, J^I), smile parameter extraction |
| **`kidger_cboe_integration.py`** | Bridge | **Connects preprocessing → model**: `create_training_loop()`, `prepare_neural_rde_inputs()`, interpretation utilities |
| **`CBOE_INTEGRATION.md`** | Docs | Quick reference for CBOE pipeline |

### Forecasting & Trading

| File | Type | Description |
|------|------|-------------|
| **`carr_wu_kidger_trading.py`** | Implementation | **Complete trading framework** — Carr-Wu portfolio construction (vol/skew/smile trades), `KidgerTimingEngine` for signal extraction, `create_trading_pipeline()` for end-to-end usage |
| **`kidger_forecasting.py`** | Docs | Philosophy document: WHY log-signatures forecast, depth-by-depth interpretation tables |

### Utilities & Documentation

| File | Type | Description |
|------|------|-------------|
| **`visualization.py`** | Utility | Plotting: paths, signatures, smile surfaces, Greeks, training curves |
| **`example_quickstart.py`** | Example | Minimal working example |
| **`GAP_ANALYSIS.md`** | Docs | Theoretical gap analysis and how each gap was addressed |
| **`README.md`** | Docs | This file |

---

## 🚀 Quick Start

```python
# Check all components are available
from neural_rde_options import check_components
check_components()

# End-to-end trading pipeline
from neural_rde_options import create_trading_pipeline

pipeline = create_trading_pipeline()
decisions = pipeline.process_and_trade(cboe_df, forward=680.0, tau=1/252)

for dec in decisions:
    print(f"{dec.trade_type.value}: {dec.action} (size={dec.position_size:.2%})")
```

---

## 🔍 Where to Find Specific Components

### Signature Computation
```
neural_rde_options.py:
  - compute_signature()
  - compute_logsignature()
  - compute_levy_area()
  - logsignature_dimension()
```

### Greek Extraction
```
enhanced_implementation.py:
  - SignatureGreekMapping (NamedTuple)
  - extract_greeks_from_signature()
  - EnhancedOutputDecoder (multi-head decoder)
```

### CBOE Path Construction
```
cboe_preprocessing.py:
  - CBOEPreprocessor.process_snapshot() → X = (t, log_S, σ, J^S, J^I)
  - CBOEPreprocessor.extract_atm_iv()
  - CBOEPreprocessor.detect_jumps()
```

### Smile Parameter Extraction
```
cboe_preprocessing.py:
  - SmileExtractor.fit_carr_wu_smile() → (γ, ω², ξ, A², R²)
  - SmileExtractor.extract_smile_parameters()
```

### Forecasting Signals
```
carr_wu_kidger_trading.py:
  - KidgerTimingEngine.extract_vol_signal()   → S^(log_S, log_S) vs I²
  - KidgerTimingEngine.extract_skew_signal()  → A^(log_S, σ) vs b
  - KidgerTimingEngine.extract_smile_signal() → S^(σ, σ) vs c

enhanced_implementation.py:
  - interpret_depth3_for_forecasting() → momentum, vol_persistence, jump_asymmetry
```

### Carr-Wu Baseline
```
enhanced_implementation.py:
  - CarrWuBaseline.forecast_variance_linear()
  - CarrWuBaseline.compute_smile_carr_wu()
  - CarrWuBaseline.compute_comparison_metrics()
```

### Trading Portfolios
```
carr_wu_kidger_trading.py:
  - CarrWuPortfolioConstructor.construct_vol_trade()   → ATM straddle
  - CarrWuPortfolioConstructor.construct_skew_trade()  → Risk-reversal
  - CarrWuPortfolioConstructor.construct_smile_trade() → Butterfly
  - CarrWuKidgerStrategy.generate_decisions()
```

### Training
```
neural_rde_options.py:
  - train_model()
  - compute_loss()

enhanced_implementation.py:
  - compute_enhanced_total_loss() (with no-arbitrage constraints)
```

---

## 🚀 Recommended Reading Order

1. **Start here:** `example_quickstart.py` — minimal working example
2. **Understand theory:** `kidger_forecasting.py` — why this works
3. **For your CBOE data:** `cboe_preprocessing.py` + `CBOE_INTEGRATION.md`
4. **For trading:** `carr_wu_kidger_trading.py` — the Carr-Wu/Kidger dialogue and trading framework
5. **Deep dive:** `neural_rde_options.py` + `enhanced_implementation.py`

---

## Overview

Short-dated options pose unique challenges for traditional pricing frameworks:
1. **Jump sensitivity**: Short-term options are highly sensitive to price and implied volatility jumps
2. **ATM skew explosion**: The at-the-money skew explodes as τ → 0
3. **Poor forecastability**: Variance/covariance forecasts degrade at short horizons

This implementation uses **Neural Rough Differential Equations** to address these challenges by:
- Compressing high-frequency paths via **log-signatures**
- Learning the relationship between path statistics and option prices
- Providing **signature-based Greeks** including the **Libra** (Lévy area sensitivity)

## Theoretical Foundation

### The Log-ODE Method

The core insight is that a Controlled Differential Equation (CDE):

```
dZ_t = f_θ(Z_t) dX_t
```

can be approximated by an Ordinary Differential Equation (ODE):

```
dẐ/dt = f̂_θ(Ẑ) · LogSig(X) / Δt
```

where `LogSig(X)` is the log-signature of the path over an interval, capturing:
- **Depth 1**: Increments (Δ, V - delta, vega)
- **Depth 2**: Lévy areas (order of movements, path convexity)
- **Depth 3+**: Higher-order path interactions

### Signature-Based P&L Attribution

The Functional Taylor Expansion (Dupire-Tissot-Daguette) decomposes P&L as:

```
P&L ≈ Θ·δt + Δ·δy + Γ·(δy²/2) + V·δσ + L·A(Y)
```

where:
- **Θ, Δ, Γ, V**: Classical Greeks
- **L**: Libra (sensitivity to Lévy area)
- **A(Y)**: Lévy area of future path

For short-dated options, the Libra term becomes significant because jumps create large Lévy areas that don't vanish as τ → 0.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd neural_rde_options

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### GPU Support (Optional)

For GPU acceleration, install JAX with CUDA support:

```bash
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Quick Start

```python
import jax.random as jr
from neural_rde_options import (
    NeuralRDE,
    generate_jump_diffusion_path,
    compute_logsignatures_for_intervals,
    logsignature_dimension,
    price_option,
    train_model
)

# Initialize model
key = jr.PRNGKey(42)
input_dim = 3  # (time, log_price, implied_vol)
logsig_dim = logsignature_dimension(input_dim, depth=2)

model = NeuralRDE(
    input_dim=input_dim,
    hidden_dim=64,
    logsig_dim=logsig_dim,
    step_size=32,
    depth=2,
    key=key
)

# Generate sample path
path, info = generate_jump_diffusion_path(
    key,
    n_steps=1000,
    lambda_jump=10.0  # High jump intensity for short-dated
)

# Train model
trained_model, losses = train_model(
    model,
    n_epochs=100,
    batch_size=32,
    key=key
)

# Price a short-dated option
spot = jnp.exp(path[-1, 1])
result = price_option(
    trained_model,
    path,
    strike=spot * 1.01,
    spot=spot,
    tau=1/252,  # 1-day option
    implied_vol_atm=0.2
)

print(f"Implied Vol: {result['implied_vol']:.4f}")
print(f"Libra Greek: {result['greeks'].libra:.6f}")
```

## Architecture

### Model Components

1. **InitialEncoder**: Maps initial observation to hidden state
2. **VectorField**: Neural network f_θ: ℝ^hidden → ℝ^{hidden × logsig_dim}
3. **OutputDecoder**: Multiple heads for different outputs:
   - Implied volatility
   - Moment forecasts (σ², γ, ω²)
   - Signature Greeks (Θ, Δ, Γ, V, L)

### Key Functions

| Function | Description |
|----------|-------------|
| `compute_logsignature(path, depth)` | Compute log-signature of a path |
| `compute_levy_area(path)` | Extract Lévy area (depth-2 antisymmetric part) |
| `compute_logsignatures_for_intervals(path, step_size, depth)` | Preprocess path into log-signature sequence |
| `NeuralRDE(...)` | Main model class |
| `train_model(model, ...)` | Training loop |
| `price_option(model, path, ...)` | Price a short-dated option |

## Key Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `step_size` | Observations per log-signature interval | 16-128 |
| `depth` | Log-signature truncation depth | 2-3 |
| `hidden_dim` | Hidden state dimension | 32-128 |
| `mlp_width` | Vector field MLP width | 64-256 |

**Trade-offs**:
- Larger `step_size` → Faster training, less detail
- Higher `depth` → More path information, more parameters
- Larger `hidden_dim` → More expressive, harder to train

## Signature Greeks

The model outputs five Greeks based on the signature decomposition:

| Greek | Symbol | Interpretation |
|-------|--------|----------------|
| Theta | Θ | Time decay |
| Delta | Δ | Spot sensitivity |
| Gamma | Γ | Convexity in spot |
| Vega | V | Vol sensitivity |
| **Libra** | L | **Lévy area sensitivity** |

The **Libra** is new - it measures sensitivity to the *order* of price and volatility movements:
- Positive Libra: Value increases when price moves before vol
- Negative Libra: Value increases when vol moves before price

This is crucial for short-dated options where the path shape matters, not just endpoints.

## Mathematical Details

### Log-Signature Dimension

The dimension of the depth-M log-signature is:

```
β(d, M) = Σ_{k=1}^{M} (1/k) Σ_{j|k} μ(k/j) d^j
```

For practical use:
- depth=1: d
- depth=2: d + d(d-1)/2
- depth=3: d + d(d-1)/2 + d(d-1)

### Lévy Area

For a 2D path (X¹, X²), the Lévy area is:

```
A = (1/2)(S^{1,2} - S^{2,1}) = (1/2) ∫∫_{s<t} (dX¹_s dX²_t - dX²_s dX¹_t)
```

This is the signed area between the path and the chord connecting its endpoints.

### Connection to Carr-Wu

The Carr-Wu no-arbitrage relation:

```
0 = B_t + B_I I_t μ_t + (1/2)B_{SS} S²σ² + (1/2)B_{II} I²ω² + B_{IS} IS γ
```

is a special case of the signature decomposition at depth 2 with specific functional forms for the Greeks.

## Performance

Based on the original NRDE paper (Morrill et al., 2021):

| Metric | NCDE (step=1) | NRDE (depth=2, step=32) |
|--------|---------------|-------------------------|
| Accuracy | 62.4% | **83.8%** |
| Training Time | 22.0 hrs | **2.4 hrs** |
| Memory | 176 MB | 180 MB |

*Results on EigenWorms dataset (length 17,984)*

## References

1. Morrill, Salvi, Kidger, Foster, Lyons (2021). "Neural Rough Differential Equations"
2. **Al-Jaaf, Carr (2023). "Vol, Skew, and Smile Trading", Journal of Derivatives** — The three trades
3. Carr, Wu (2020). "Option Profit and Loss Attribution and Pricing", Journal of Finance
4. Carr, Lee, Lorig. "Robust Replication of Volatility and Hybrid Derivatives on Jump Diffusions"
5. Dupire, Tissot-Daguette. "Signature and the Functional Taylor Expansion"
6. Friz, Hairer. "A Course on Rough Paths"

---

## Key Equations Quick Reference

### Carr-Wu Three Trades (from Al-Jaaf & Carr 2023)

| Trade | Portfolio | Mean Gain Rate | Sharpe (1M) |
|-------|-----------|----------------|-------------|
| **Vol** | ATM straddle, η = 2/$Γ | G = σ² - I² | 0.42 |
| **Skew** | Risk-reversal + vega hedge | G = γ - b | **1.38** |
| **Smile** | Butterfly + vega hedge | G = ω² - c | 0.89 |

### Kidger Timing Signals

| Trade | Log-Signature Component | Signal |
|-------|------------------------|--------|
| Vol | S^(log_S, log_S) | S^(1,1)/(I²τ) - 1 |
| Skew | A^(log_S, σ) = S^(1,2) - S^(2,1) | A/√τ - b |
| Smile | S^(σ, σ) | S^(2,2)/(cτ) - 1 |

### Extended Smile Formula

```
Standard Carr-Wu:  I² - A² = 2γz₊ + ω²z₊z₋
Kidger Extension:  I² - A² = 2γz₊ + ω²z₊z₋ + ξE[A(X)]
                                              ↑
                                    Lévy area term (large for τ < 1 month)
```

## Citation

If you use this code, please cite:

```bibtex
@article{morrill2021neural,
  title={Neural Rough Differential Equations},
  author={Morrill, James and Salvi, Cristopher and Kidger, Patrick and Foster, James and Lyons, Terry},
  journal={arXiv preprint arXiv:2009.08295},
  year={2021}
}
```

## License

MIT License

## Contributing

Contributions welcome! Please open an issue or submit a pull request.

## Acknowledgments

This implementation is based on the theoretical framework developed by:
- Patrick Kidger (Neural CDEs/RDEs)
- Terry Lyons (Rough path theory)
- Bruno Dupire (Functional Itô calculus)
- Peter Carr and Liuren Wu (Options P&L attribution)
