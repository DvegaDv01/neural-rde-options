# kidger_preprocessing.py - Log-Signature Utilities

## Overview

This module provides a focused, educational implementation of log-signature computation with extensive inline commentary from Kidger's papers. It serves as both a utility library and a reference for understanding the theoretical foundations.

**Location**: `kidger_preprocessing.py`
**Lines**: 694
**Dependencies**: NumPy, Pandas

---

## Core Insight

The module opens with Kidger's key insight:

> "The log-signature may be treated as a carefully-selected binning method, to reduce the amount of data considered whilst retaining the information most important for solving a CDE."

**This is not arbitrary feature engineering** — log-signatures are the mathematically optimal compression for CDE-driven dynamics.

For options: the CDE is the pricing equation, the control is (t, S, I), and the log-signature tells us exactly how (t, S, I) will drive option value.

---

## Module Structure

### Part I: Data Structures (Lines 26-61)

#### `Interval` Dataclass (Line 26)

Immutable container for a time interval with its log-signature.

**Fields**:
```python
start: float      # Start time
end: float        # End time
logsig: np.ndarray # Log-signature vector
```

**Property**: `duration` returns `end - start`

**Design note**: Immutable by design — once computed, a log-signature doesn't change. This enables the "preprocess once, use many times" paradigm.

#### `signature_dimension(d, depth)` (Line 43)

Computes dimension of depth-N log-signature for d-dimensional paths.

| Depth | Formula | Meaning |
|-------|---------|---------|
| 1 | d | Just increments |
| 2 | d + d(d-1)/2 | Increments + Lévy areas |
| 3 | d + d(d-1)/2 + d²(d-1)/3 | + Higher-order terms |

**Key quote**: "The Lévy area is the KEY term for options — it captures the ordering of movements."

---

### Part II: Signature Computation (Lines 63-153)

#### Theoretical Background

The signature is defined by iterated integrals:
```
S^{i,j}(X) = ∫∫_{s<t} dX^i_s dX^j_t
```

Chen's identity enables incremental computation:
```
S(X * Y) = S(X) ⊗ S(Y)
```

#### `compute_increment(path)` (Line 81)

Depth-1 signature: simple increment.

```
S^i = X^i_end - X^i_start
```

This is what classical Greeks see: ΔS, Δt, ΔI.

#### `compute_levy_area(path)` (Line 92)

Depth-2 signature: the **Lévy area** (antisymmetric part).

```
A^{ij} = (S^{i,j} - S^{j,i}) / 2
```

**Financial interpretation**:
- When price drops THEN vol rises: **negative** area
- When vol rises THEN price drops: **positive** area

**Same total moves, different P&L**. The Lévy area sees the difference; classical Greeks (Δ, Γ, V) do not.

**Implementation**:
```python
dX = np.diff(path, axis=0)
cumsum = np.cumsum(dX, axis=0)  # Running integral
sig2 = np.einsum('ti,tj->ij', cumsum, dX)  # Iterated integral
levy = (sig2 - sig2.T) / 2  # Antisymmetric part
```

Returns upper triangle (independent components).

#### `compute_logsignature(path, depth)` (Line 132)

Computes the full log-signature.

**Kidger quote**: "The log-signature transform is obtained by computing the signature and throwing out redundant terms, to obtain some minimal collection."

At depth 2: `[increment, Lévy areas]`

**Key property**: The log-signature determines the solution of any CDE driven by the path.

---

### Part III: Path Construction (Lines 155-234)

#### `extract_path_point(snapshot, ...)` (Line 173)

Extracts a single point (t, log S, σ) from an options snapshot.

**Components**:
| Component | Extraction Method |
|-----------|-------------------|
| t | Normalized to [0, 1] over trading day (9:30 AM = 0, 4:00 PM = 1) |
| log_S | `np.log(spot)` for delta/gamma consistency |
| σ | Weighted average of short-dated ATM call IVs |

**ATM IV extraction**:
1. Filter to 0-7 DTE calls
2. Compute distance from spot: `|strike/spot - 1|`
3. Take 5 closest options
4. Weighted average (closer = more weight)

#### `construct_path(snapshots)` (Line 225)

Constructs the control path from a sequence of snapshots.

Returns array of shape `(n_snapshots, 3)` with columns `[t, log_s, sigma]`.

---

### Part IV: Log-ODE Preprocessing (Lines 236-327)

#### Theoretical Motivation

**Kidger quote**: "When training a model in practice, the log-signatures need only be computed once and thus the computation can be performed as part of data preprocessing."

The transformation:
```
Long sequence of raw observations → Short sequence of log-signatures
```

**Hyperparameters**:
- `step_size`: Observations per interval
- `depth`: Log-signature truncation depth

#### `preprocess_to_logsignatures(path, step_size, depth)` (Line 261)

The key preprocessing function.

**Kidger quote**: "The sequence of log-signatures is now of length m, which was chosen to be much smaller than n. As such, it is much more slowly varying over the interval [t_0, t_n] than the original data."

This slower variation is the source of speedups: the neural ODE solver can take larger steps.

**Implementation**:
```python
intervals = []
i = 0
while i + step_size < n:
    segment = path[i : i + step_size + 1]
    logsig = compute_logsignature(segment, depth)
    intervals.append(Interval(
        start=path[i, 0],
        end=path[i + step_size, 0],
        logsig=logsig
    ))
    i += step_size
# Handle remainder...
```

#### `intervals_to_array(intervals)` (Line 319)

Converts list of Intervals to array of shape `(n_intervals, logsig_dim)`.

---

### Part V: Cross-Sectional Targets (Lines 329-432)

#### Theoretical Background

Carr-Wu show that the implied volatility smile encodes moment conditions:

```
I²(K) - A² = 2γz₊ + ω²z₊z₋
```

Where:
- A: ATM implied vol
- z₊, z₋: Standardized moneyness measures
- γ: Return-vol covariance (drives skew)
- ω²: Vol-of-vol (drives curvature)

These are **targets** for the Neural RDE to predict from path log-signatures.

#### `SmileTargets` Dataclass (Line 348)

```python
atm_iv: float    # ATM implied volatility
gamma: float     # Return-vol covariance (skew)
omega2: float    # Vol-of-vol (curvature)
r_squared: float # Fit quality
```

#### `fit_smile(snapshot, ...)` (Line 359)

Extracts Carr-Wu smile parameters from cross-section.

**Steps**:
1. Focus on short-dated (≤5 DTE) options
2. Compute ATM IV
3. Compute standardized moneyness z₊, z₋
4. Filter to |z₊| ≤ 1 (Carr-Wu's recommendation)
5. OLS regression: y = I² - A², X = [2z₊, z₊z₋]
6. Extract γ, ω² (force ω² ≥ 0)
7. Compute R²

**Note**: R² > 99% in Carr-Wu's empirical work, validating local commonality.

---

### Part VI: Complete Pipeline (Lines 434-518)

#### `ProcessedDay` Dataclass (Line 439)

Complete processed day ready for Neural RDE.

**Fields**:
```python
date: str                    # Trading date
path: np.ndarray             # (n_snapshots, 3)
logsignatures: np.ndarray    # (n_intervals, logsig_dim)
targets: SmileTargets

# Statistics
n_snapshots: int
n_intervals: int
realized_variance: float     # From path
total_levy_area: float       # (S, I) Lévy area
```

**Method**: `compression_ratio()` returns `n_snapshots / n_intervals`

#### `process_day(snapshots, step_size, depth)` (Line 466)

**Main entry point** for processing a day of CBOE data.

**Steps**:
1. Construct path from snapshots
2. Compute log-signatures (the key step)
3. Extract targets from final cross-section
4. Compute statistics (realized variance, Lévy area)
5. Return `ProcessedDay`

---

### Part VII: Hyperparameter Guidance (Lines 520-579)

**Kidger quote**: "Increasing step size will lead to faster (but less informative) training. Increasing depth will lead to slower (but more informative) training."

#### Recommendations for CBOE 5-minute data:

| Setting | step_size | depth | Result |
|---------|-----------|-------|--------|
| Default | 8 | 2 | ~10 intervals/day, 40-min windows |
| Finer | 4 | 2 | ~20 intervals/day, 20-min windows |
| Rougher | 8 | 3 | Captures jump asymmetry |

#### `recommend_hyperparameters(n_snapshots_per_day, data_roughness)` (Line 542)

Returns recommended hyperparameters based on data characteristics.

| Roughness | step_size | depth |
|-----------|-----------|-------|
| 'smooth' | 16 | 2 |
| 'typical' | 8 | 2 |
| 'rough' | 4 | 3 |

---

### Part VIII: Demonstration (Lines 581-694)

#### `demonstrate()` (Line 586)

Complete demonstration with output:

```
1. PATH CONSTRUCTION
   Converting snapshots to control path X = (t, log S, σ)
   Path shape: (3, 3)
   Time range: [0.0128, 0.1026]

2. LOG-SIGNATURE COMPUTATION
   Depth-2 log-signature dimension: 6
   Components:
     Increment (Δt, Δlog S, Δσ): [0.0897, -0.0008, 0.0012]
     Lévy areas (t-S, t-σ, S-σ): [-0.00004, 0.00005, -0.00001]

   The (S, σ) Lévy area = -0.000014
   → Price dropped THEN vol rose (typical crash pattern)

3. CROSS-SECTIONAL TARGETS
   ATM IV: 0.2034
   γ (return-vol covariance): -0.4521
   ω² (vol-of-vol): 0.0876
   R²: 0.9823
```

#### Key Insight (printed at end)

> "The log-signature is not arbitrary feature engineering. It is the OPTIMAL summary of a path for driving a CDE. This is a theorem, not a heuristic."

---

## Usage Example

```python
from kidger_preprocessing import (
    construct_path,
    compute_logsignature,
    preprocess_to_logsignatures,
    process_day,
    recommend_hyperparameters
)

# Load snapshots
snapshots = [pd.read_csv(f) for f in files]

# Construct path
path = construct_path(snapshots)

# Compute log-signature for full path
logsig = compute_logsignature(path, depth=2)

# Or process full day
processed = process_day(snapshots, step_size=8, depth=2)
print(f"Compression: {processed.compression_ratio():.1f}x")
print(f"Lévy area: {processed.total_levy_area:.6f}")
```

---

## Function Reference Table

| Function/Class | Line | Purpose |
|----------------|------|---------|
| `Interval` | 26 | Immutable interval container |
| `signature_dimension` | 43 | Dimension formula |
| `compute_increment` | 81 | Depth-1 signature |
| `compute_levy_area` | 92 | **Depth-2 Lévy area** |
| `compute_logsignature` | 132 | Full log-signature |
| `extract_path_point` | 173 | Extract (t, log S, σ) |
| `construct_path` | 225 | Build full path |
| `preprocess_to_logsignatures` | 261 | **Key preprocessing** |
| `intervals_to_array` | 319 | Convert to array |
| `SmileTargets` | 348 | Carr-Wu targets |
| `fit_smile` | 359 | Smile regression |
| `ProcessedDay` | 439 | Complete processed day |
| `process_day` | 466 | **Main entry point** |
| `recommend_hyperparameters` | 542 | Hyperparameter guidance |
| `demonstrate` | 586 | Full demonstration |
