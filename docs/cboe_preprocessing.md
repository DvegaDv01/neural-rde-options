# cboe_preprocessing.py - CBOE Data Pipeline

## Overview

This is the **workhorse** module for processing CBOE options data into the format required by Neural RDE models. It handles path construction, signature computation, smile parameter extraction, and training batch assembly.

**Location**: `cboe_preprocessing.py`
**Lines**: 1133
**Dependencies**: NumPy, Pandas, optional JAX

---

## Key Concepts

### The Path Representation

The Neural RDE framework requires paths of the form:

```
X = (t, log_S, σ, J^S, J^I)
```

Where:
- **t**: Normalized time [0, 1] within trading day (9:30 AM = 0.0, 4:00 PM = 1.0)
- **log_S**: Natural log of spot price
- **σ**: ATM implied volatility (instantaneous vol proxy)
- **J^S**: Cumulative spot jump count (normalized)
- **J^I**: Cumulative vol jump count (normalized)

### Log-Signature Compression

For CBOE 5-minute data with 78 snapshots/day:
- `step_size=8` → ~9 log-signature vectors per day (10x compression)
- `step_size=4` → ~19 log-signature vectors per day
- `depth=2` → captures Lévy areas (essential for jumps)
- `depth=3` → captures skewness (jump asymmetry)

---

## Module Structure

### Part 1: Configuration and Data Structures (Lines 23-170)

#### `PreprocessingConfig` Dataclass (Line 27)

Central configuration for the entire preprocessing pipeline.

**Column mappings** (for CBOE data):
| Field | Default Value | Purpose |
|-------|---------------|---------|
| `spot_column` | `'active_underlying_price'` | Underlying price |
| `time_column` | `'quote_datetime'` | Timestamp |
| `expiration_column` | `'expiration'` | Option expiration |
| `strike_column` | `'strike'` | Strike price |
| `option_type_column` | `'option_type'` | 'C' or 'P' |
| `iv_column` | `'implied_volatility'` | IV |
| `delta_column` | `'delta'` | Delta |
| `gamma_column` | `'gamma'` | Gamma |
| `theta_column` | `'theta'` | Theta |
| `vega_column` | `'vega'` | Vega |
| `bid_column` | `'bid'` | Bid price |
| `ask_column` | `'ask'` | Ask price |
| `volume_column` | `'trade_volume'` | Volume |
| `oi_column` | `'open_interest'` | Open interest |

**Hyperparameters**:
| Parameter | Default | Kidger's Guidance |
|-----------|---------|-------------------|
| `step_size` | 8 | "Increasing step_size leads to faster (but less informative) training" |
| `depth` | 2 | "Increasing depth leads to slower (but more informative) training" |
| `spot_jump_threshold` | 0.005 | 0.5% ≈ 2σ for 5-min data |
| `vol_jump_threshold` | 0.02 | 2% absolute IV change |
| `atm_moneyness_range` | 0.02 | ±2% around spot for ATM |
| `smile_moneyness_range` | 1.0 | |z₊| ≤ 1 for smile fitting |
| `min_options_for_fit` | 5 | Minimum for regression |

#### `PathData` Dataclass (Line 85)

Container for processed path data ready for Neural RDE.

**Fields**:
```python
timestamps: np.ndarray          # Original timestamps
path: np.ndarray                # Shape: (n_snapshots, 5)
logsignatures: np.ndarray       # Shape: (n_intervals, logsig_dim)

# Metadata
n_snapshots: int
n_intervals: int
logsig_dim: int

# Jump statistics
n_spot_jumps: int
n_vol_jumps: int

# Path statistics
realized_variance: float
realized_covariance: float
total_levy_area: float
```

#### `SmileParameters` Dataclass (Line 116)

Cross-sectional smile parameters from Carr-Wu regression.

**Fields**:
```python
gamma: float      # Vanna coefficient (typically negative)
omega2: float     # Volga coefficient (positive)
xi: float         # Lévy area coefficient (NEW)
atm_iv: float     # ATM implied volatility
r_squared: float  # Regression fit quality
n_options: int    # Number of options used

# Extended diagnostics
skew_25d: Optional[float]      # 25-delta skew
butterfly_25d: Optional[float] # 25-delta butterfly
```

**Carr-Wu Formula**:
```
I² - A² = 2γz₊ + ω²z₊z₋ + ξE[A(X)]
```

#### `TrainingBatch` Dataclass (Line 140)

Complete training batch structure.

**Contents**:
- Path data (paths, logsignatures)
- Cross-sectional targets (smile_params, realized moments)
- IV prediction targets (moneyness, tau, target_iv)
- Market Greeks for validation

---

### Part 2: Signature Computation (Lines 172-368)

#### `logsig_dimension(d, depth)` (Line 176)

Computes log-signature dimension using Möbius function formula.

| Depth | Formula | For d=5 |
|-------|---------|---------|
| 1 | d | 5 |
| 2 | d + d(d-1)/2 | 15 |
| 3 | d + d(d-1)/2 + d(d-1) | 35 |

#### `compute_signature_depth1(path)` (Line 206)

Simple increments: `S^i = X^i_T - X^i_0`

#### `compute_signature_depth2(path)` (Line 211)

Iterated integrals: `S^{i,j} = ∫∫_{s<t} dX^i_s dX^j_t`

Uses efficient einsum computation:
```python
dX = np.diff(path, axis=0)
X_cumsum = np.cumsum(dX, axis=0)
sig2 = np.einsum('ti,tj->ij', X_cumsum, dX)
```

#### `compute_levy_area(path)` (Line 226)

Antisymmetric part of depth-2 signature:
```
A^{ij} = (S^{i,j} - S^{j,i}) / 2
```

**Key insight**: "The log-signature of a path with jumps looks very different from a continuous path: the depth-2 terms (Lévy areas) are much larger because jumps create sudden changes in direction."

#### `_compute_signature_depth3(path)` (Line 238)

Triple iterated integrals via sequential accumulation:
```python
for t in range(len(dX)):
    sig3 += np.einsum('ij,k->ijk', running_sum_ij, dX_t)
    running_sum_ij += np.outer(running_sum_i, dX_t)
    running_sum_i += dX_t
```

#### `compute_logsignature(path, depth)` (Line 271)

Computes full log-signature with components:
- **Depth 1**: Increments (classical first-order sensitivities)
- **Depth 2**: Lévy areas (order of movements - essential for jumps)
- **Depth 3**: Skewness/Lyndon basis terms (jump asymmetry)

Returns concatenated vector.

#### `compute_logsignatures_for_intervals(path, step_size, depth)` (Line 329)

**KEY preprocessing step**: Splits path into intervals and computes log-signature for each.

**Kidger quote**: "The log-signature preprocessing is done ONCE and reused. This amortizes the cost of processing the high-frequency path. For 78 snapshots/day with step_size=8, we get ~9 log-signature vectors — a 10x compression with IMPROVED accuracy."

---

### Part 3: Path Construction (Lines 370-726)

#### `CBOEPathConstructor` Class (Line 374)

Main class for constructing paths from CBOE snapshots.

**State maintained**:
```python
_prev_log_s: float          # Previous log price
_prev_sigma: float          # Previous ATM IV
_cumulative_spot_jumps: int # Running spot jump count
_cumulative_vol_jumps: int  # Running vol jump count
```

##### `reset()` / `reset_state()` (Line 401)
Resets state for new trading day.

##### `extract_atm_iv(df, spot)` (Line 411)

Extracts ATM implied volatility from options snapshot.

**Logic**:
1. Filter to short-dated (0-5 DTE) calls
2. Find options within `atm_moneyness_range` of spot
3. Weight by proximity to ATM and open interest
4. Return weighted average IV

##### `detect_jumps(log_s, sigma)` (Line 454)

Detects jumps in spot and volatility.

**Jump criteria**:
- Spot jump: |Δlog_S| > `spot_jump_threshold` (default 0.5%)
- Vol jump: |Δσ| > `vol_jump_threshold` (default 2%)

Updates cumulative jump counters.

##### `normalize_time(timestamp)` (Line 486)

Normalizes timestamp to [0, 1] within trading day:
```
9:30 AM → 0.0
4:00 PM → 1.0
```

##### `process_snapshot(df, compute_smile)` (Line 502)

Processes single CBOE snapshot into path point.

**Steps**:
1. Extract timestamp and spot price
2. Normalize time to [0, 1]
3. Compute log price
4. Extract ATM IV
5. Detect jumps
6. Compute normalized cumulative jump counts
7. Optionally extract smile parameters

**Returns**: `path_point` array [t, log_S, σ, J^S, J^I] and optional `SmileParameters`

##### `extract_smile_parameters(df, spot, atm_iv)` (Line 550)

Fits Carr-Wu smile formula to cross-section.

**Regression**:
```
I² - A² = 2γz₊ + ω²z₊z₋
```

**Implementation**:
1. Filter to short-dated options (0-5 DTE)
2. Compute standardized moneyness z₊, z₋
3. Filter to fitting range (|z₊| ≤ 1)
4. Compute I² - A² for each option
5. Weighted least squares with:
   - Weight by bid-ask tightness
   - Weight by open interest
6. Extract γ (can be negative), ω² (forced positive)
7. Compute R² for fit quality
8. Optionally compute 25-delta skew and butterfly

##### `process_day(snapshots, compute_smile)` (Line 666)

Processes full day of snapshots into `PathData`.

**Steps**:
1. Reset state
2. Process each snapshot → path points
3. Stack into path array
4. Compute log-signatures over intervals
5. Compute path statistics:
   - Realized variance (annualized)
   - Realized covariance
   - Total Lévy area
6. Return `PathData` object

---

### Part 4: Batch Construction (Lines 728-841)

#### `TrainingDataBuilder` Class (Line 732)

Builds training batches for Neural RDE.

**Kidger quote**: "The computational efficiency comes from the log-ODE method: instead of processing 2,340 minute bars individually, we compress them into ~70 log-signature vectors. This gives 10x speedup with 17% accuracy improvement."

##### `load_snapshots_from_files(file_paths)` (Line 746)

Loads snapshots from CSV files in sorted order.

##### `build_daily_batch(snapshots, target_strikes, target_tau)` (Line 757)

Builds single-day training batch.

**Steps**:
1. Process day via `CBOEPathConstructor`
2. Use last snapshot for cross-sectional targets
3. Compute moneyness for target strikes
4. Look up target IVs from options chain
5. Extract smile parameters
6. Optionally extract market Greeks for validation
7. Assemble `TrainingBatch`

---

### Part 5: Online/Streaming Processing (Lines 843-924)

#### `OnlinePathProcessor` Class (Line 847)

Real-time processor for streaming inference.

**Kidger quote**: "Log-signatures can be computed in an online fashion, making the model suitable for real-time problems."

**State maintained**:
```python
_current_interval_path: List   # Points in current interval
_logsig_sequence: List         # Completed log-signatures
_full_path: List               # All path points
_snapshots_in_interval: int    # Counter
```

##### `reset()` (Line 870)

Resets all state for new trading day.

##### `update(df)` (Line 878)

Processes new snapshot and returns current state.

**Logic**:
1. Process snapshot → path point
2. Add to current interval
3. If interval complete (≥ step_size):
   - Compute log-signature
   - Add to sequence
   - Reset interval (with overlap of 1)
4. Return current path and log-signature sequence

##### `get_partial_logsig()` (Line 913)

Returns log-signature for incomplete current interval (useful for real-time inference before interval completes).

---

### Part 6: Utility Functions (Lines 926-1025)

#### `analyze_path_roughness(path)` (Line 930)

Analyzes path roughness characteristics.

**Metrics returned**:
| Metric | Meaning |
|--------|---------|
| `realized_variance` | Annualized realized variance |
| `realized_skewness` | Skewness of returns |
| `levy_area_magnitude` | |A^{price,vol}| |
| `roughness_ratio` | Hurst-like exponent (< 0.5 = rough) |
| `jump_count` | Number of 2σ moves |
| `jump_ratio` | Fraction of jumps |

**Kidger quote**: "Short-dated options have ROUGH dynamics — dominated by jumps and rapid regime changes. The signature captures this roughness through higher-order terms."

#### `recommend_hyperparameters(n_snapshots, roughness, target_accuracy)` (Line 981)

Recommends step_size and depth based on data characteristics.

**Base recommendations**:
| Target | step_size | depth |
|--------|-----------|-------|
| 'fast' | 16 | 2 |
| 'balanced' | 8 | 2 |
| 'accurate' | 4 | 3 |

**Adjustments**:
- High jump ratio (> 10%) → halve step_size, increase depth
- Large Lévy areas (> 0.01) → ensure depth ≥ 2
- Adjust step_size for target ~10 intervals

---

### Part 7: Demonstration (Lines 1027-1133)

#### `demo_preprocessing_pipeline()` (Line 1031)

Complete demonstration of the preprocessing pipeline.

**Steps demonstrated**:
1. Create sample CBOE-like snapshots
2. Configure preprocessing
3. Process day → PathData
4. Show path statistics
5. Display log-signature components
6. Analyze path roughness
7. Get hyperparameter recommendations
8. Build training batch

---

## Data Flow Diagram

```
CBOE CSV Files (5-min snapshots)
        │
        ▼
┌───────────────────────────────────────┐
│     CBOEPathConstructor               │
│  ┌─────────────────────────────────┐  │
│  │ extract_atm_iv()                │  │
│  │ detect_jumps()                  │  │
│  │ normalize_time()                │  │
│  │ process_snapshot()              │  │
│  └─────────────────────────────────┘  │
│                 │                     │
│                 ▼                     │
│  Path: (t, log_S, σ, J^S, J^I)       │
│  + SmileParameters                    │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│  compute_logsignatures_for_intervals  │
│  ┌─────────────────────────────────┐  │
│  │ Split into step_size intervals  │  │
│  │ compute_logsignature() each     │  │
│  │ → depth-1: increments           │  │
│  │ → depth-2: Lévy areas           │  │
│  │ → depth-3: skewness terms       │  │
│  └─────────────────────────────────┘  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│     PathData                          │
│  - path: (n_snapshots, 5)             │
│  - logsignatures: (n_intervals, dim)  │
│  - realized_variance, covariance      │
│  - total_levy_area                    │
│  - jump counts                        │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│     TrainingDataBuilder               │
│  - build_daily_batch()                │
│  - Add IV targets                     │
│  - Add smile parameters               │
│  - Add market Greeks                  │
└───────────────────────────────────────┘
        │
        ▼
┌───────────────────────────────────────┐
│     TrainingBatch                     │
│  Ready for Neural RDE training        │
└───────────────────────────────────────┘
```

---

## Usage Example

```python
from cboe_preprocessing import (
    PreprocessingConfig,
    CBOEPathConstructor,
    TrainingDataBuilder,
    analyze_path_roughness
)

# Configure
config = PreprocessingConfig(
    step_size=8,
    depth=2,
    spot_jump_threshold=0.005
)

# Process day
constructor = CBOEPathConstructor(config)
path_data = constructor.process_day(snapshots)

print(f"Path shape: {path_data.path.shape}")
print(f"Log-sig shape: {path_data.logsignatures.shape}")
print(f"Lévy area: {path_data.total_levy_area}")

# Build training batch
builder = TrainingDataBuilder(config)
batch = builder.build_daily_batch(snapshots)

# Analyze roughness
roughness = analyze_path_roughness(path_data.path)
print(f"Jump ratio: {roughness['jump_ratio']:.2%}")
```

---

## Function Reference Table

| Function/Class | Line | Purpose |
|----------------|------|---------|
| `PreprocessingConfig` | 27 | Central configuration |
| `PathData` | 85 | Processed path container |
| `SmileParameters` | 116 | Carr-Wu smile parameters |
| `TrainingBatch` | 140 | Training batch structure |
| `logsig_dimension` | 176 | Log-signature dimension |
| `compute_signature_depth1` | 206 | Depth-1 signature |
| `compute_signature_depth2` | 211 | Depth-2 signature |
| `compute_levy_area` | 226 | Antisymmetric part |
| `_compute_signature_depth3` | 238 | Depth-3 signature |
| `compute_logsignature` | 271 | Full log-signature |
| `compute_logsignatures_for_intervals` | 329 | Interval-wise computation |
| `CBOEPathConstructor` | 374 | Path construction class |
| `CBOEPathConstructor.extract_atm_iv` | 411 | ATM IV extraction |
| `CBOEPathConstructor.detect_jumps` | 454 | Jump detection |
| `CBOEPathConstructor.process_snapshot` | 502 | Single snapshot processing |
| `CBOEPathConstructor.extract_smile_parameters` | 550 | Smile fitting |
| `CBOEPathConstructor.process_day` | 666 | Full day processing |
| `TrainingDataBuilder` | 732 | Batch construction |
| `OnlinePathProcessor` | 847 | Streaming processor |
| `analyze_path_roughness` | 930 | Roughness analysis |
| `recommend_hyperparameters` | 981 | Hyperparameter suggestions |
| `demo_preprocessing_pipeline` | 1031 | Demonstration |
