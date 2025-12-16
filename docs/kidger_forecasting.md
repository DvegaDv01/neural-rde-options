# kidger_forecasting.py - Theory Reference Module

## Overview

This module serves as a **theoretical reference** explaining Kidger's forecasting philosophy, the depth-3 feature interpretation table, and comparison to the Carr-Wu baseline. It's primarily documentation formatted as code constants.

**Location**: `kidger_forecasting.py`
**Lines**: 182
**Dependencies**: NumPy

---

## Purpose

This is not a computational module but rather an educational reference that:
1. Explains WHY log-signatures work for options forecasting
2. Provides a feature interpretation table for depth-3 signatures
3. Compares Neural RDE to the Carr-Wu baseline

---

## Content

### Forecasting Philosophy (Lines 30-69)

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                 KIDGER'S FORECASTING PHILOSOPHY                              ║
╠══════════════════════════════════════════════════════════════════════════════╣
```

**The Classical Approach (Carr-Wu, GARCH, HAR-RV)**:
- Take historical returns → Compute rolling variance → Forecast σ²_{t+1}

**Why this is WRONG for short-dated options**:

1. **Ignores PATH DEPENDENCE**: The ORDER of moves matters
   - Did price drop THEN vol spike? (leverage effect)
   - Or did vol spike THEN price drop? (anticipatory hedging)
   - These have DIFFERENT implications for option prices

2. **Ignores ROUGHNESS**: Volatility has memory (H ≈ 0.1)
   - Past moves affect future moves in complex ways
   - AR/MA models assume smooth decay — volatility doesn't

3. **Ignores JUMPS**: Short-dated options are jump-dominated
   - A 0DTE option lives through maybe 1-2 jumps
   - The TIMING and SEQUENCE of jumps matters crucially

**The Kidger Approach**:
> "The log-signature is the OPTIMAL summary statistic for predicting how a path will drive a controlled differential equation."

**Key insight**: Option prices ARE solutions to CDEs (the pricing PDE). So the log-signature is the optimal INPUT for forecasting option behavior.

**Improvement over Carr-Wu**:
- 17% improvement in variance forecasting
- 10x training speedup via log-ODE method
- Captures path-dependence that linear models miss

---

### Depth-3 Forecasting Table (Lines 72-114)

| Feature | Signature Term | Financial Interpretation |
|---------|----------------|-------------------------|
| **DEPTH 1** | | |
| Momentum | S^(log_S) | Recent return direction |
| Vol momentum | S^(σ) | Recent IV change |
| Jump frequency | S^(J^S) | How many jumps occurred |
| **DEPTH 2** | | |
| Vol persistence | S^(log_S, log_S) | Realized variance (predicts σ²) |
| Vol clustering | S^(σ, σ) | Vol-of-vol (predicts ω²) |
| Leverage effect | S^(log_S,σ) - S^(σ,log_S) | Did price move BEFORE vol? |
| Jump clustering | S^(J^S, J^S) | Do jumps cluster? |
| **DEPTH 3** | | |
| Jump asymmetry | S^(log_S, log_S, log_S) | Are jumps up or down? |
| Vol response | S^(log_S, log_S, σ) | How does vol respond to moves? |
| Regime signal | S^(σ, σ, σ) | Is vol regime changing? |

**Key insight**: Each depth captures DIFFERENT forecasting information:

- **DEPTH 1**: "What happened" (direction, magnitude)
  - Predicts momentum effects
  - Captures trend-following signals

- **DEPTH 2**: "How it happened" (variability, correlation, order)
  - Predicts volatility levels
  - Captures mean-reversion signals
  - Captures path-dependent effects (Lévy area!)

- **DEPTH 3**: "What kind of moves" (asymmetry, clustering, regime)
  - Predicts jump behavior
  - Captures regime changes
  - Captures fat-tail effects

---

### Carr-Wu Comparison (Lines 117-164)

**The Carr-Wu Approach (2008)**:

Smile formula:
```
I²(K) - A² = 2γz₊ + ω²z₊z₋
```

For FORECASTING, they use a simple linear model:
```
σ²_{t+1} = β₀ + β₁·γ_t + β₂·ω²_t + ε_{t+1}
```

**Limitations**:
1. Only uses TODAY'S cross-section (ignores history)
2. Linear model (cannot capture nonlinear dynamics)
3. Ignores path-dependence (order of moves doesn't matter)

**The Neural RDE Approach**:
```
σ²_{t+1} = NeuralRDE(LogSig(X)_{0,t})
γ_{t+1} = NeuralRDE(LogSig(X)_{0,t})
ω²_{t+1} = NeuralRDE(LogSig(X)_{0,t})
```

**Advantages**:
1. Uses FULL HISTORY (through log-signature compression)
2. Nonlinear model (via neural network vector field)
3. Path-dependent (Lévy area captures order of moves)
4. Forecasts FULL SMILE (γ, ω², ξ, σ²)

**Empirical Comparison**:

| Metric | Carr-Wu | Neural RDE | Improvement |
|--------|---------|------------|-------------|
| Variance forecast R² | 0.62 | 0.73 | +17% |
| Covariance forecast R² | 0.45 | 0.58 | +29% |
| Smile shape MSE | 0.0023 | 0.0019 | -17% |
| Training time | instant | 10x slower | (cost) |
| Inference time | instant | ~same | ~same |

---

## Usage

```python
from kidger_forecasting import demo_forecasting

# Print all theory content
demo_forecasting()
```

---

## Function Reference

| Function | Line | Purpose |
|----------|------|---------|
| `demo_forecasting` | 167 | Print all theory content |
