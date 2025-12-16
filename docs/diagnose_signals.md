# diagnose_signals.py - Signal Debugging Tool

## Overview

This is a **diagnostic utility** that shows all trading signals regardless of threshold, helping users understand why no actionable signals were generated. Useful for debugging and understanding signal behavior.

**Location**: `diagnose_signals.py`
**Lines**: 145
**Dependencies**: main.py, carr_wu_kidger_trading

---

## Purpose

When the trading framework returns no actionable signals, this tool helps diagnose:
1. What the actual signal values are
2. How they compare to thresholds
3. Why signals might be below threshold

---

## Main Function

### `diagnose_signals(data_dir)` (Line 23)

Shows all signals even if below threshold.

**Process**:
1. Load data from specified directory
2. Create pipeline with very low thresholds (0.001) to capture everything
3. For each trading day:
   - Reset pipeline
   - Process all snapshots
   - Display final log-signature
   - Show signal values vs thresholds
   - Display interpretations

**Output format**:
```
======================================================================
Date: 2025-12-01  (78 snapshots)
======================================================================

Spot: $680.45
ATM IV: 18.50%
Gamma (γ): -0.450000
Omega² (ω²): 0.120000

Log-signature shape: (15,)
Log-signature values:
  Depth 1 (first 5): [ 0.025 -0.008 -0.003  0.     0.   ]
  Depth 2 Lévy areas: [ 0.0001 -0.00005  0.  ...]

----------------------------------------------------------------------
SIGNAL VALUES VS THRESHOLDS
----------------------------------------------------------------------
Trade     Signal   Threshold   Strength   Confidence  Status
----------------------------------------------------------------------
vol      +0.0523     0.1000       0.52         0.67  ✗ Below threshold
skew     -0.0012     0.0050       0.24         0.55  ✗ Below threshold
smile    +0.0842     0.1500       0.56         0.72  ✗ Below threshold

Signal interpretations:
  vol: No clear signal → HOLD
  skew: Lévy area neutral → HOLD
  smile: No clear signal → HOLD
```

---

## Thresholds Explained

At the end, the tool prints threshold guidance:

| Trade | Threshold | Meaning |
|-------|-----------|---------|
| Vol | > 0.10 | 10% deviation of realized vs implied variance |
| Skew | > 0.005 | 0.5% Lévy area deviation |
| Smile | > 0.15 | 15% deviation of vol-of-vol vs implied |
| Confidence | ≥ 0.60 | Consistent signal direction |

**Key insight**: A calm day with low volatility will not generate actionable signals. Consider loading data from more volatile days or adjusting thresholds.

---

## Usage

```bash
# Use default data directory
python diagnose_signals.py

# Specify data directory
python diagnose_signals.py data/s3_cache
```

---

## Function Reference

| Function | Line | Purpose |
|----------|------|---------|
| `diagnose_signals` | 23 | Main diagnostic function |
