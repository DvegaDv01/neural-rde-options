# Codebase Overview: Neural RDE Options Pricing Framework

This is a **production-ready implementation** of Patrick Kidger's Neural RDE framework for short-dated options pricing, with CBOE data integration and Carr-Wu trading signals.

## Directory Structure

```
neural_rde_options/
├── Core Mathematics (JAX/Equinox)
│   ├── neural_rde_options.py        # Signatures, Neural RDE model
│   └── enhanced_implementation.py    # Jump indicators, Carr-Wu baseline
│
├── CBOE Data Pipeline (NumPy/Pandas)
│   ├── cboe_preprocessing.py        # Data loading, path construction
│   ├── kidger_cboe_integration.py   # Bridge: preprocessing → model
│   ├── kidger_preprocessing.py      # Log-signature utilities
│   └── s3_data_source.py            # AWS S3 integration
│
├── Trading Application
│   ├── carr_wu_kidger_trading.py    # Portfolio construction, signals
│   ├── kidger_forecasting.py        # Theory reference
│   └── diagnose_signals.py          # Signal debugging
│
├── Utilities
│   ├── visualization.py             # Plotting functions
│   ├── main.py                      # CLI entry point
│   └── __init__.py                  # Package interface
│
└── Configuration
    ├── requirements.txt
    └── README.md
```

## Key Files by Layer

| Layer | File | Purpose |
|-------|------|---------|
| **Core** | `neural_rde_options.py` | Log-signatures, Lévy areas, NeuralRDE model, training |
| **Core** | `enhanced_implementation.py` | Extended model with jump augmentation, Carr-Wu baseline |
| **Data** | `cboe_preprocessing.py` | Path construction, smile extraction (workhorse) |
| **Data** | `kidger_cboe_integration.py` | Bridges preprocessing to model training |
| **Trading** | `carr_wu_kidger_trading.py` | Vol/Skew/Smile trades, timing signals |
| **Entry** | `main.py` | CLI with modes: demo, train, analyze, trade |

## Core Concepts

- **Log-signature**: Compresses path history into ~15 features via iterated integrals
- **Lévy area**: Captures movement ordering (crucial for short-dated options)
- **Libra Greek**: New sensitivity measure to Lévy area
- **Three Carr-Wu Trades**: Vol (straddle), Skew (risk-reversal), Smile (butterfly)

## Data Flow

```
CBOE CSV → CBOEPathConstructor → Path (t, log_S, σ, J^S, J^I)
    → Log-signature computation → Neural RDE training
    → Smile parameters + Greeks → Trading signals
```

## Dependencies

- **JAX ecosystem**: jax, equinox, diffrax, optax
- **Data**: numpy, pandas
- **Cloud**: boto3 (S3)
- **Visualization**: matplotlib, tqdm

## Architecture Patterns

### Layered Architecture
- **Layer 1 (Core)**: JAX/Equinox for mathematical operations
- **Layer 2 (Data)**: NumPy/Pandas for data pipelines
- **Layer 3 (Application)**: Trading logic and decision-making

### Bridge Pattern
- `kidger_cboe_integration.py` bridges preprocessing and model
- Handles data format conversions and integration concerns

### Factory Pattern
- `create_training_loop()`: Creates training infrastructure
- `create_trading_pipeline()`: Assembles complete trading system

### Strategy Pattern
- Multiple trading strategies (Vol, Skew, Smile) as separate trades
- `CarrWuKidgerStrategy` orchestrates all three

### Graceful Degradation
- JAX components optional (fallback to NumPy)
- S3 integration optional (work with local data)
- Each module can function independently with warnings

## Key Hyperparameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| `step_size` | 8 | 4-128 | Snapshots per interval; trade-off speed vs detail |
| `depth` | 2 | 1-3 | Log-signature depth; essential for Lévy areas |
| `hidden_dim` | 64 | 32-128 | Neural network width |
| `mlp_width` | 128 | 64-256 | Vector field MLP capacity |
| `spot_jump_threshold` | 0.5% | - | Jump detection sensitivity |
| `vol_jump_threshold` | 2% | - | IV jump detection sensitivity |

## Greeks and P&L Attribution

### Signature Greeks
- **Θ (Theta)**: Time decay — S^(0)
- **Δ (Delta)**: Spot sensitivity — S^(1)
- **Γ (Gamma)**: Spot convexity — S^(1,1)
- **V (Vega)**: Vol sensitivity — S^(2)
- **L (Libra)**: Lévy area sensitivity — **NEW** for short-dated

### Functional Taylor Expansion
```
P&L ≈ Θ·δt + Δ·δy + Γ·(δy²/2) + V·δσ + L·A(Y)
```

Where A(Y) is the Lévy area of future path — crucial term for path-dependence.

## Problem Being Solved

- **Short-dated options** (weeks to days): Classical Black-Scholes fails
- **Jump sensitivity**: Price/vol jumps dominate risk
- **Path-dependence**: Order of movements matters
- **Rough volatility**: Non-Brownian behavior at short horizons

## Solution Provided

- **Neural RDE model**: Learns path-to-price mapping via signature compression
- **End-to-end pipeline**: CBOE data → preprocessing → model training → trading signals
- **Three complementary trades**: Vol, skew, smile targeting different market dislocations
- **Enhanced Greeks**: Includes Libra for Lévy area sensitivity

## Key Innovation

- **Log-signature as optimal input**: Captures all path information needed for pricing
- **Theoretical grounding**: Based on rough path theory and functional calculus
- **Practical integration**: Ready-to-use with real CBOE data
