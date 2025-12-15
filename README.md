# Neural Rough Differential Equations for Short-Dated Options Pricing

A JAX-based implementation of Patrick Kidger's Neural RDE framework for pricing short-dated options, addressing the challenges identified by Carr-Wu in their P&L attribution framework.

## Overview

Short-dated options pose unique challenges for traditional pricing frameworks:
1. **Jump sensitivity**: Short-term options are highly sensitive to price and implied volatility jumps
2. **ATM skew explosion**: The at-the-money skew explodes as τ → 0
3. **Poor forecastability**: Variance/covariance forecasts degrade at short horizons

This implementation uses **Neural Rough Differential Equations** to address these challenges by:
- Compressing high-frequency paths via **log-signatures**
- Learning the relationship between path statistics and option prices
- Providing **signature-based Greeks** including the **Libra** (Lévy area sensitivity)

---

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

---

## Usage

This is a research library/framework with multiple execution modes depending on your use case.

### Running the Demos

Each module is executable as a standalone script demonstrating its functionality:

```bash
# Core Neural RDE - trains model, generates paths, prices options
python neural_rde_options.py

# Enhanced implementation - jump indicators, Libra Greek, extended smile formula
python enhanced_implementation.py

# CBOE data preprocessing pipeline
python cboe_preprocessing.py

# Trading framework - Carr-Wu portfolios + Kidger timing signals
python carr_wu_kidger_trading.py

# Visualization - generates plots for paths, signatures, Greeks, IV surfaces
python visualization.py

# Forecasting philosophy - theoretical exposition
python kidger_forecasting.py
```

**Recommended starting point:**
```bash
python neural_rde_options.py
```

This runs a complete end-to-end demo: generates synthetic jump-diffusion data, trains the model, and prices a short-dated option with all signature Greeks.

### Using as a Library

```python
import jax.numpy as jnp
import jax.random as jr
from neural_rde_options import (
    NeuralRDE,
    generate_jump_diffusion_path,
    logsignature_dimension,
    train_model,
    price_option
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

### Production Workflow with CBOE Data

```python
import jax.random as jr
from neural_rde_options import NeuralRDE, logsignature_dimension, train_model, price_option
from cboe_preprocessing import CBOEPathConstructor, PreprocessingConfig
from carr_wu_kidger_trading import (
    CarrWuPortfolioConstructor, KidgerTimingEngine, CarrWuKidgerStrategy
)

# Step 1: Configure preprocessing
config = PreprocessingConfig(step_size=8, depth=2)

# Step 2: Process CBOE snapshots into paths
constructor = CBOEPathConstructor(config)
path_data = constructor.process_day(snapshots)  # List of DataFrames

# Step 3: Initialize Neural RDE
key = jr.PRNGKey(42)
logsig_dim = logsignature_dimension(5, depth=2)  # 5-channel augmented path

model = NeuralRDE(
    input_dim=5,
    hidden_dim=64,
    logsig_dim=logsig_dim,
    step_size=8,
    depth=2,
    key=key
)

# Step 4: Train on historical data
trained_model, losses = train_model(model, n_epochs=100, key=key)

# Step 5: Price options
result = price_option(
    trained_model, path_data.path,
    strike=682, spot=680, tau=1/252, implied_vol_atm=0.18
)

# Step 6: Generate trading decisions
portfolio_constructor = CarrWuPortfolioConstructor()
timing_engine = KidgerTimingEngine()
strategy = CarrWuKidgerStrategy(portfolio_constructor, timing_engine)

decisions = strategy.generate_decisions(
    forward=680, atm_strike=680, put_strike=650, call_strike=710,
    atm_iv=0.18, put_iv=0.22, call_iv=0.16, tau=21/252,
    logsig=path_data.logsignatures[-1]
)
```

### Real-Time Streaming Mode

For live trading with streaming CBOE data:

```python
from cboe_preprocessing import OnlinePathProcessor, PreprocessingConfig

config = PreprocessingConfig(step_size=8, depth=2)
processor = OnlinePathProcessor(config)

# On each new 5-minute snapshot:
for snapshot_df in live_data_stream:
    current_path, current_logsigs = processor.update(snapshot_df)

    # Run inference when you have enough data
    if len(current_logsigs) > 0:
        outputs = trained_model(
            current_path, current_logsigs,
            moneyness=0.0, tau=1/252
        )
        # Make trading decisions...
```

---

## File Organization

### Core Implementation

| File | Description |
|------|-------------|
| `neural_rde_options.py` | Base Neural RDE: signatures, log-signatures, Lévy areas, `NeuralRDE` class, training, pricing |
| `enhanced_implementation.py` | Extended features: signature-to-Greek mapping, jump indicators, extended smile formula, `EnhancedNeuralRDE` |

### Data Pipeline

| File | Description |
|------|-------------|
| `cboe_preprocessing.py` | CBOE data processing: `CBOEPathConstructor`, path construction X=(t, log_S, σ, J^S, J^I), smile extraction |
| `kidger_cboe_integration.py` | Tutorial: load snapshots → build paths → compute log-signatures → train |

### Trading Framework

| File | Description |
|------|-------------|
| `carr_wu_kidger_trading.py` | Complete trading framework: Carr-Wu portfolios (vol/skew/smile), `KidgerTimingEngine`, combined strategy |
| `kidger_forecasting.py` | Philosophy: why log-signatures forecast, depth interpretation, Carr-Wu comparison |

### Utilities

| File | Description |
|------|-------------|
| `visualization.py` | Plotting: paths, signatures, smile surfaces, Greeks, training curves |
| `example_quickstart.py` | Minimal working example |

---

## Architecture

### Model Components

1. **InitialEncoder**: Maps initial observation to hidden state
2. **VectorField**: Neural network f_θ: R^hidden → R^{hidden × logsig_dim}
3. **OutputDecoder**: Multiple heads for:
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

---

## Theoretical Foundation

### The Log-ODE Method

The core insight is that a Controlled Differential Equation (CDE):

```
dZ_t = f_θ(Z_t) dX_t
```

can be approximated by an Ordinary Differential Equation (ODE):

```
dZ/dt = f_θ(Z) · LogSig(X) / Δt
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

### Signature Greeks

| Greek | Symbol | Interpretation |
|-------|--------|----------------|
| Theta | Θ | Time decay |
| Delta | Δ | Spot sensitivity |
| Gamma | Γ | Convexity in spot |
| Vega | V | Vol sensitivity |
| **Libra** | L | **Lévy area sensitivity** |

The **Libra** measures sensitivity to the *order* of price and volatility movements:
- Positive Libra: Value increases when price moves before vol
- Negative Libra: Value increases when vol moves before price

---

## Key Hyperparameters

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `step_size` | Observations per log-signature interval | 4-32 |
| `depth` | Log-signature truncation depth | 2-3 |
| `hidden_dim` | Hidden state dimension | 32-128 |
| `mlp_width` | Vector field MLP width | 64-256 |

**Trade-offs**:
- Larger `step_size` → Faster training, less detail
- Higher `depth` → More path information, more parameters
- Larger `hidden_dim` → More expressive, harder to train

---

## Key Equations

### Carr-Wu Three Trades (Al-Jaaf & Carr 2023)

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

---

## Performance

Based on the original NRDE paper (Morrill et al., 2021):

| Metric | NCDE (step=1) | NRDE (depth=2, step=32) |
|--------|---------------|-------------------------|
| Accuracy | 62.4% | **83.8%** |
| Training Time | 22.0 hrs | **2.4 hrs** |
| Memory | 176 MB | 180 MB |

*Results on EigenWorms dataset (length 17,984)*

---

## References

1. Morrill, Salvi, Kidger, Foster, Lyons (2021). "Neural Rough Differential Equations"
2. Al-Jaaf, Carr (2023). "Vol, Skew, and Smile Trading", Journal of Derivatives
3. Carr, Wu (2020). "Option Profit and Loss Attribution and Pricing", Journal of Finance
4. Dupire, Tissot-Daguette. "Signature and the Functional Taylor Expansion"
5. Friz, Hairer. "A Course on Rough Paths"

---

## Citation

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
