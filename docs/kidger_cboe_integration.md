# kidger_cboe_integration.py - Bridge Module

## Overview

This module serves as the **bridge** between the CBOE preprocessing pipeline and the Neural RDE model. It provides the complete workflow for converting CBOE options data into training-ready formats, along with interpretable output mappings.

**Location**: `kidger_cboe_integration.py`
**Lines**: 942
**Dependencies**: NumPy, Pandas, cboe_preprocessing, neural_rde_options, enhanced_implementation

---

## Kidger's Philosophy

The module opens with a key philosophical statement:

> "The beauty of the Neural RDE framework is that it transforms a seemingly intractable problem — pricing short-dated options with rough volatility — into a well-posed machine learning problem with a clean mathematical structure."

The framework provides:
1. **UNIVERSALITY**: Log-signatures optimally summarize path behavior
2. **EFFICIENCY**: log-ODE method compresses thousands of observations
3. **ACCURACY**: Captures jumps and path-dependence that Black-Scholes misses
4. **INTERPRETABILITY**: Signature terms map to financial Greeks

---

## Module Structure

### Part 1: Data Loading and Validation (Lines 180-310)

#### `CBOEDataset` Dataclass (Line 184)

Container for a complete CBOE dataset ready for Neural RDE training.

**Fields**:
```python
days: List[str]                          # Trading days
snapshots: Dict[str, List[pd.DataFrame]] # Date → snapshots
path_data: Dict[str, PathData]           # Date → processed paths
batches: Dict[str, TrainingBatch]        # Date → training batches
config: PreprocessingConfig              # Configuration used
```

#### `load_cboe_data(data_dir, file_pattern)` (Line 202)

Loads CBOE options data from a directory.

**Expected file format**: `UnderlyingOptionsIntervals_300sec_calcs_oi_{DATE}_{HHMM}.csv`

**Logic**:
1. Glob files matching pattern
2. Parse date from filename (second-to-last underscore segment)
3. Group by date
4. Sort snapshots within each day by timestamp

#### `validate_cboe_snapshot(df)` (Line 249)

Validates data quality of a CBOE snapshot.

**Checks performed**:
| Check | Description |
|-------|-------------|
| `has_required_columns` | All necessary columns present |
| `has_data` | DataFrame not empty |
| `has_calls` | Contains call options |
| `has_puts` | Contains put options |
| `iv_reasonable` | IV between 1% and 1000% |
| `spot_positive` | Spot price > 0 |
| `has_0dte` | Contains 0DTE options |
| `has_short_dated` | Contains ≤5 DTE options (>50) |

#### `print_data_summary(snapshots_by_date)` (Line 283)

Prints formatted summary of loaded data.

---

### Part 2: Complete Preprocessing Pipeline (Lines 311-398)

#### `preprocess_cboe_for_neural_rde(snapshots_by_date, config)` (Line 315)

**Main entry point** for converting CBOE data to Neural RDE training data.

**Steps**:
1. Analyze data and print configuration
2. For each trading day:
   - Process day via `CBOEPathConstructor`
   - Build training batch via `TrainingDataBuilder`
   - Store path_data and batches
3. Analyze path roughness across all days
4. Generate hyperparameter recommendations
5. Return `CBOEDataset`

**Console output example**:
```
PREPROCESSING CBOE DATA FOR NEURAL RDE

Configuration:
  Step size: 8 (→ 9 intervals/day)
  Depth: 2
  Log-sig dimension: 15
  Jump thresholds: spot=0.5%, vol=2%

Processing days:
  2025-12-01: 78 snaps → 9 intervals, γ=-0.450, ω²=0.120, jumps: S=2, I=1
```

---

### Part 3: Neural RDE Integration (Lines 400-544)

#### `prepare_neural_rde_inputs(dataset)` (Line 404)

Converts `CBOEDataset` into arrays ready for Neural RDE training.

**Returns dictionary with**:
```python
{
    'logsignatures': np.ndarray,    # (n_days, n_intervals, logsig_dim)
    'paths': np.ndarray,            # (n_days, n_snapshots, 5)
    'smile_gamma': np.ndarray,      # (n_days,)
    'smile_omega2': np.ndarray,     # (n_days,)
    'realized_variance': np.ndarray,# (n_days,)
    'target_iv': np.ndarray,        # (n_days, n_strikes)
    'moneyness': np.ndarray,        # (n_strikes,)
    'tau': np.ndarray               # (n_days,)
}
```

**Note**: Paths are padded to uniform length using edge values.

#### `create_training_loop(dataset, model_config, use_enhanced, random_seed)` (Line 464)

Creates complete training setup for Neural RDE.

**Default model configuration**:
```python
{
    'hidden_dim': 64,
    'step_size': 8,
    'depth': 2,
    'mlp_width': 128,
    'mlp_depth': 3,
}
```

**Returns**:
```python
{
    'model': NeuralRDE or EnhancedNeuralRDE,
    'inputs': dict from prepare_neural_rde_inputs,
    'config': model_config dict,
    'dataset': CBOEDataset,
    'loss_fn': appropriate loss function,
    'use_enhanced': bool
}
```

**Model selection**:
- If `use_enhanced=True` and `HAS_ENHANCED`: Use `EnhancedNeuralRDE` with jump indicators
- Otherwise: Use base `NeuralRDE`

---

### Part 4: Interpretable Outputs (Lines 546-690)

#### `KidgerInterpretation` Dataclass (Line 550)

Human-readable interpretation of Neural RDE outputs.

**Fields by category**:

| Category | Field | Signature Term | Meaning |
|----------|-------|----------------|---------|
| Classical | `delta` | S^(1) | Spot sensitivity |
| Classical | `gamma` | S^(1,1) | Convexity |
| Classical | `vega` | S^(2) | Vol sensitivity |
| Classical | `theta` | S^(0) | Time decay |
| Higher-order | `vanna` | S^(1,2)+S^(2,1) | Spot-vol cross |
| Higher-order | `volga` | S^(2,2) | Vol convexity |
| **Path-dependent** | `levy_area_spot_vol` | S^(1,2)-S^(2,1) | **Order sensitivity** |
| **Path-dependent** | `libra` | S^(0,1)-S^(1,0) | **Time-space bracket** |
| Smile | `smile_gamma` | - | Return-vol correlation |
| Smile | `smile_omega2` | - | Vol-of-vol |
| Smile | `smile_xi` | - | **Lévy area term** |
| Forecast | `predicted_variance` | - | Implied variance |
| Forecast | `realized_variance` | - | Actual variance |
| Forecast | `forecast_error` | - | Prediction error |

#### `interpret_logsignature(logsig, path, smile_params)` (Line 585)

Converts log-signature and smile parameters to interpretable quantities.

**Signature term mapping for 5-channel path** (t, log_S, σ, J^S, J^I):

**Depth-1 (indices 0-4)**:
| Index | Term | Greek |
|-------|------|-------|
| 0 | Δt | Θ (theta) |
| 1 | Δlog_S | Δ (delta) |
| 2 | Δσ | V (vega) |
| 3 | ΔJ^S | Jump frequency |
| 4 | ΔJ^I | Vol jump frequency |

**Depth-2 Lévy areas (indices 5-14)**:
| Index | Term | Meaning |
|-------|------|---------|
| 5 | A^(0,1) | Libra (time-spot) |
| 6 | A^(0,2) | Time-vol |
| 7 | A^(1,2) | **Spot-vol Lévy area** |
| ... | ... | ... |

#### `print_interpretation(interp)` (Line 658)

Prints human-readable interpretation with sections:
1. Classical Greeks (from Depth-1 signature)
2. Higher-order Greeks (from Depth-2 signature)
3. Path-dependent quantities (Kidger's contribution)
4. Smile parameters (Carr-Wu extension)
5. Variance forecast

**Includes interpretation of Lévy area sign**:
- Positive: "spot moved BEFORE vol"
- Negative: "vol moved BEFORE spot"

---

### Part 5: Workflow Demonstration (Lines 692-847)

#### `demo_cboe_integration(data_dir)` (Line 696)

Complete demonstration of CBOE → Neural RDE workflow.

**Steps**:
1. Load CBOE data (or create sample)
2. Configure preprocessing (step_size=8, depth=2)
3. Run preprocessing pipeline
4. Prepare Neural RDE inputs
5. Interpret first day's output
6. Show training loop schema

**Training loop schema** (printed):
```python
for epoch in range(n_epochs):
    for batch in dataset:
        predictions = model(batch.logsignatures)
        loss = (
            λ_iv * MSE(predictions.iv, batch.target_iv) +
            λ_smile * MSE(predictions.gamma, batch.smile_gamma) +
            λ_smile * MSE(predictions.omega2, batch.smile_omega2) +
            λ_forecast * MSE(predictions.variance, batch.realized_variance)
        )
        grads = grad(loss)(params)
        params = optimizer.update(params, grads)
```

**Expected improvements**:
- 17% improvement in variance forecasting over Carr-Wu
- Better short-dated option pricing via Lévy area term
- Interpretable Greeks via signature decomposition

#### `_create_sample_cboe_data()` (Line 790)

Creates synthetic CBOE-like data for demonstration when no real data provided.

---

### Part 6: Convenience Functions (Lines 849-930)

#### `process_your_cboe_files(file_paths)` (Line 853)

Process specific CBOE files directly.

**Usage**:
```python
files = [
    '/path/to/sample_1000.csv',
    '/path/to/sample_1005.csv',
    '/path/to/sample_1010.csv'
]
dataset = process_your_cboe_files(files)
```

#### `quick_analysis(file_paths)` (Line 883)

Quick analysis of CBOE data files showing:
- Data quality checks
- Path characteristics
- Smile parameters
- Lévy area magnitudes

---

## Usage Example

```python
from kidger_cboe_integration import (
    load_cboe_data,
    preprocess_cboe_for_neural_rde,
    create_training_loop,
    demo_cboe_integration
)

# Load data
snapshots = load_cboe_data("/path/to/cboe/data")

# Preprocess
dataset = preprocess_cboe_for_neural_rde(snapshots)

# Create training setup
training = create_training_loop(dataset, use_enhanced=True)
model = training['model']
inputs = training['inputs']

# Or run full demo
demo_cboe_integration("/path/to/data")
```

---

## Function Reference Table

| Function/Class | Line | Purpose |
|----------------|------|---------|
| `CBOEDataset` | 184 | Complete dataset container |
| `load_cboe_data` | 202 | Load from directory |
| `validate_cboe_snapshot` | 249 | Data quality validation |
| `print_data_summary` | 283 | Print summary |
| `preprocess_cboe_for_neural_rde` | 315 | **Main preprocessing entry point** |
| `prepare_neural_rde_inputs` | 404 | Convert to arrays |
| `create_training_loop` | 464 | Create training setup |
| `KidgerInterpretation` | 550 | Interpretable outputs |
| `interpret_logsignature` | 585 | Convert to interpretable form |
| `print_interpretation` | 658 | Print interpretation |
| `demo_cboe_integration` | 696 | Full workflow demo |
| `process_your_cboe_files` | 853 | Process specific files |
| `quick_analysis` | 883 | Quick data analysis |
