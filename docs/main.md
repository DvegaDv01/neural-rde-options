# main.py - Unified CLI Entry Point

## Overview

This is the **unified entry point** for the complete Neural RDE pipeline, providing command-line access to demo, training, analysis, and trading signal generation modes.

**Location**: `main.py`
**Lines**: 1018
**Dependencies**: All modules (with graceful fallbacks)

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 1: Core Math (JAX)                                          │
│  neural_rde_options.py, enhanced_implementation.py                 │
│  • NeuralRDE / EnhancedNeuralRDE models                            │
│  • Signature computation                                           │
│  • Training loops                                                  │
└─────────────────────────────────────────────────────────────────────┘
                                ↑
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 2: Data Pipeline (NumPy/Pandas)                             │
│  cboe_preprocessing.py, kidger_cboe_integration.py                 │
│  • CBOE data loading                                               │
│  • Path construction: X = (t, log_S, σ, J^S, J^I)                  │
│  • Log-signature computation                                       │
│  • Smile parameter extraction                                      │
└─────────────────────────────────────────────────────────────────────┘
                                ↓
┌─────────────────────────────────────────────────────────────────────┐
│  LAYER 3: Trading Application                                      │
│  carr_wu_kidger_trading.py                                         │
│  • Carr-Wu portfolio construction (vol/skew/smile)                 │
│  • Kidger timing signals from log-signatures                       │
│  • Trading decisions                                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Command-Line Usage

```bash
# Demo mode (synthetic data, shows full pipeline)
python main.py --mode demo

# Train on synthetic data
python main.py --mode train --epochs 100

# Train on CBOE data
python main.py --mode train --data /path/to/cboe --epochs 100 --save model.eqx

# Analyze CBOE data without training
python main.py --mode analyze --data /path/to/cboe

# Generate trading signals
python main.py --mode trade --data /path/to/cboe --model model.eqx
```

---

## Command-Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `demo` | Operating mode: demo, train, analyze, trade |
| `--data` | None | Path to CBOE data directory |
| `--model` | None | Path to trained model (for trade or continue training) |
| `--save` | None | Path to save trained model |
| `--epochs` | 100 | Number of training epochs |
| `--batch-size` | 32 | Training batch size |
| `--lr` | 1e-3 | Learning rate |
| `--step-size` | 8 | Log-signature step size |
| `--depth` | 2 | Log-signature truncation depth |
| `--no-enhanced` | False | Use basic NeuralRDE instead of enhanced |

---

## Configuration

### `PipelineConfig` Dataclass (Line 158)

Central configuration for the complete pipeline.

**Preprocessing settings**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `step_size` | 8 | Snapshots per log-signature interval |
| `depth` | 2 | Log-signature truncation depth |
| `spot_jump_threshold` | 0.005 | 0.5% for spot jumps |
| `vol_jump_threshold` | 0.02 | 2% IV for vol jumps |

**Model settings**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `hidden_dim` | 64 | Hidden state dimension |
| `mlp_width` | 128 | MLP layer width |
| `mlp_depth` | 3 | Number of MLP layers |
| `use_enhanced` | True | Use EnhancedNeuralRDE |

**Training settings**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_epochs` | 100 | Training epochs |
| `batch_size` | 32 | Batch size |
| `learning_rate` | 1e-3 | Learning rate |

**Trading settings**:
| Parameter | Default | Description |
|-----------|---------|-------------|
| `vol_threshold` | 0.10 | Vol signal threshold |
| `skew_threshold` | 0.005 | Skew signal threshold |
| `smile_threshold` | 0.15 | Smile signal threshold |
| `risk_budget` | 0.10 | Risk budget fraction |
| `min_confidence` | 0.6 | Minimum confidence |

---

## Operating Modes

### Mode: Demo (Lines 286-512)

**Purpose**: Demonstrates all three layers without requiring real CBOE data.

**Steps**:
1. **Layer 1**: Generate synthetic jump-diffusion path, compute log-signatures, train model
2. **Layer 2**: Create synthetic CBOE-like data, preprocess for Neural RDE
3. **Layer 3**: Create trading pipeline, generate sample decisions

**Output**:
```
NEURAL RDE OPTIONS PRICING - DEMONSTRATION
==================================================================

LAYER 1: Core Math (Signature Computation + Model)
------------------------------------------------------------------
1.1 Generating synthetic jump-diffusion path...
    Path shape: (500, 3)
    Jumps detected: 5
    Realized vol (ann.): 18.45%

1.2 Computing log-signatures...
    Log-signature shape: (62, 6)
    Intervals: 62
    Features per interval: 6

1.3 Creating Neural RDE model...
    Model: NeuralRDE
    Hidden dim: 64
    Log-sig dim: 6

1.4 Training model (quick demo - 20 epochs)...
    Initial loss: 0.234567
    Final loss: 0.012345
    Improvement: 94.7%
```

---

### Mode: Train (Lines 519-704)

**Purpose**: Train a Neural RDE model on CBOE data or synthetic data.

**With CBOE data**:
1. Load and preprocess data
2. Prepare inputs (log-signatures, targets)
3. Create model
4. Train with custom CBOE training loop
5. Save model

**With synthetic data**:
Uses built-in `train_model()` function.

**Training loop** (Line 641):
- Uses Optax Adam optimizer
- Samples batch indices
- Computes loss (P&L, forecast, IV, arbitrage)
- Updates parameters

---

### Mode: Analyze (Lines 711-788)

**Purpose**: Analyze CBOE data without training.

**Statistics provided**:
1. Data quality checks (columns, IV range, 0DTE presence)
2. Smile parameter statistics (γ, ω², ATM IV)
3. Jump statistics (spot jumps, vol jumps per day)
4. Path roughness analysis
5. Hyperparameter recommendations

---

### Mode: Trade (Lines 795-892)

**Purpose**: Generate trading signals from CBOE data.

**Steps**:
1. Load model (optional)
2. Load CBOE data
3. Create trading pipeline
4. For each day:
   - Process snapshots
   - Generate timing signals
   - Output trading decisions

**Output**:
```
2025-12-01 (spot=$680.45, IV=18.5%):
  vol   : HOLD  (size=0.00%)
  skew  : LONG  (size=2.50%)
  smile : HOLD  (size=0.00%)
```

---

## Model Persistence

### `save_model(model, path, config)` (Line 204)

Saves model to disk using Equinox serialization.

**Saved data**:
- Hyperparameters (for reconstruction)
- Model weights (serialized leaves)
- Configuration (optional)

### `load_model(path)` (Line 246)

Loads model from disk.

**Returns**: Tuple of (model, config)

---

## Dependency Handling

The module gracefully handles missing dependencies:

```python
try:
    import jax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    warnings.warn("JAX not available — training disabled")
```

**Component availability check** (printed at startup):
```
Checking dependencies...
  JAX:           ✓
  Core:          ✓
  Enhanced:      ✓
  Preprocessing: ✓
  Integration:   ✓
  Trading:       ✓
```

---

## Function Reference

| Function | Line | Purpose |
|----------|------|---------|
| `PipelineConfig` | 158 | Configuration dataclass |
| `save_model` | 204 | Save model to disk |
| `load_model` | 246 | Load model from disk |
| `run_demo` | 286 | Demo mode |
| `run_train` | 519 | Training mode |
| `_train_with_cboe_data` | 641 | CBOE training loop |
| `run_analyze` | 711 | Analysis mode |
| `run_trade` | 795 | Trading mode |
| `main` | 899 | CLI entry point |
