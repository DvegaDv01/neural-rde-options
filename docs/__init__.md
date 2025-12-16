# __init__.py - Package Interface

## Overview

This is the package initialization file that provides a unified import interface for all Neural RDE components. It re-exports key classes and functions from all submodules with graceful fallback handling.

**Location**: `__init__.py`
**Lines**: 328
**Version**: 0.2.0

---

## Package Structure

The package is organized into four main component groups:

### 1. CORE (neural_rde_options.py)

**Signature computation**:
- `compute_signature`, `compute_signature_depth1`, `compute_signature_depth2`
- `compute_logsignature`, `compute_levy_area`
- `compute_logsignatures_for_intervals`, `logsignature_dimension`
- `batch_compute_logsignatures`

**Model components**:
- `VectorField`, `InitialEncoder`, `OutputDecoder`
- `NeuralRDE`

**Greeks**:
- `SignatureGreeks`, `decompose_pnl_signature`, `compute_libra_greek`

**Loss functions**:
- `compute_pnl_loss`, `compute_forecast_loss`
- `compute_no_arbitrage_loss`, `compute_implied_vol_loss`
- `total_loss`

**Data generation**:
- `generate_jump_diffusion_path`, `generate_training_batch`

**Training**:
- `train_model`, `create_train_step`

**Inference**:
- `price_option`, `black_scholes_call`, `compute_smile_surface`

---

### 2. ENHANCED (enhanced_implementation.py)

**Enhanced model**:
- `EnhancedNeuralRDE`, `EnhancedOutputDecoder`

**Greek mapping**:
- `SignatureGreekMapping`, `extract_greeks_from_signature`

**Extended smile**:
- `compute_smile_formula_extended`, `analyze_atm_skew_explosion`

**Path augmentation**:
- `augment_path_with_jump_indicators`, `compute_jump_signature_features`

**Forecasting**:
- `interpret_depth3_for_forecasting`

**Baseline**:
- `CarrWuBaseline`

**Enhanced loss**:
- `compute_enhanced_total_loss`, `compute_terminal_condition_loss`, `compute_drift_constraint_loss`

---

### 3. PREPROCESSING (cboe_preprocessing.py)

**Configuration**:
- `PreprocessingConfig`

**Data structures**:
- `PathData`, `SmileParameters`, `TrainingBatch`

**Path construction**:
- `CBOEPathConstructor`

**Training data**:
- `TrainingDataBuilder`

**Online processing**:
- `OnlinePathProcessor`

**Utilities**:
- `logsig_dimension`, `analyze_path_roughness`, `recommend_hyperparameters`

---

### 4. TRADING (carr_wu_kidger_trading.py)

**Trade types**:
- `TradeType`

**Portfolio construction**:
- `OptionPosition`, `CarrWuPortfolio`, `CarrWuPortfolioConstructor`

**Timing signals**:
- `TimingSignal`, `KidgerTimingEngine`

**Strategy**:
- `TradingDecision`, `CarrWuKidgerStrategy`

**Pipeline**:
- `TradingPipeline`, `create_trading_pipeline`

---

### 5. INTEGRATION (kidger_cboe_integration.py)

- `CBOEDataset`, `load_cboe_data`
- `create_training_loop`, `prepare_neural_rde_inputs`
- `interpret_logsignature`, `KidgerInterpretation`

---

## Graceful Degradation

Each component group is imported with try/except handling:

```python
try:
    from .enhanced_implementation import (...)
    _HAS_ENHANCED = True
except ImportError as e:
    _HAS_ENHANCED = False
    warnings.warn(f"Enhanced implementation not available: {e}")
```

This allows the package to function with partial dependencies.

---

## Usage Examples

### Basic Usage

```python
# Core model
from neural_rde_options import NeuralRDE, train_model

# Enhanced model with all features
from neural_rde_options import EnhancedNeuralRDE

# CBOE preprocessing
from neural_rde_options import CBOEPathConstructor, TrainingDataBuilder

# Trading
from neural_rde_options import create_trading_pipeline
pipeline = create_trading_pipeline()
decisions = pipeline.process_and_trade(cboe_df, forward=680.0, tau=1/252)
```

### Check Component Availability

```python
from neural_rde_options import check_components

check_components()
# Output:
# Neural RDE Options Package - Component Status
# ==================================================
#   Core (neural_rde_options.py):      ✓ Available
#   Enhanced (enhanced_implementation): ✓ Available
#   Preprocessing (cboe_preprocessing): ✓ Available
#   Trading (carr_wu_kidger_trading):   ✓ Available
#   Integration (kidger_cboe_integration): ✓ Available
# ==================================================
```

---

## Exported Symbols (`__all__`)

The package exports 60+ symbols organized by category:

| Category | Count | Examples |
|----------|-------|----------|
| Signature computation | 8 | `compute_logsignature`, `compute_levy_area` |
| Model components | 4 | `NeuralRDE`, `VectorField` |
| Greeks | 3 | `SignatureGreeks`, `compute_libra_greek` |
| Loss functions | 5 | `total_loss`, `compute_pnl_loss` |
| Data generation | 2 | `generate_jump_diffusion_path` |
| Training | 2 | `train_model`, `create_train_step` |
| Inference | 3 | `price_option`, `black_scholes_call` |
| Enhanced | 12 | `EnhancedNeuralRDE`, `CarrWuBaseline` |
| Preprocessing | 10 | `CBOEPathConstructor`, `PathData` |
| Trading | 10 | `create_trading_pipeline`, `TradeType` |
| Integration | 6 | `load_cboe_data`, `CBOEDataset` |

---

## Function Reference

| Function/Variable | Line | Purpose |
|-------------------|------|---------|
| `__version__` | 223 | Package version (0.2.0) |
| `__author__` | 224 | Attribution |
| `__all__` | 226 | Exported symbols |
| `check_components()` | 318 | Print component status |
| `_HAS_ENHANCED` | 133 | Enhanced module flag |
| `_HAS_PREPROCESSING` | 167 | Preprocessing module flag |
| `_HAS_TRADING` | 197 | Trading module flag |
| `_HAS_INTEGRATION` | 214 | Integration module flag |
