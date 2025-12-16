# carr_wu_kidger_trading.py - Complete Trading Framework

## Overview

This is the **trading application layer** that combines Carr-Wu portfolio construction with Kidger log-signature timing signals. It implements the three Carr-Wu trades (Vol, Skew, Smile) and generates timing signals from log-signatures for optimal entry/exit.

**Location**: `carr_wu_kidger_trading.py`
**Lines**: 2061
**Dependencies**: NumPy, Pandas, SciPy, cboe_preprocessing

**References**:
- Al-Jaaf & Carr (2023) "Vol, Skew, and Smile Trading", J. Derivatives
- Kidger et al. (2021) "Neural Rough Differential Equations"
- Carr & Wu (2020) "Option Profit and Loss Attribution", J. Finance

---

## Core Concept

> "Carr-Wu show WHAT to trade. Kidger shows WHEN to trade. Together: A complete framework for systematic options trading."

---

## Module Structure

### Part 1: Carr-Wu Portfolio Construction (Lines 46-386)

#### `TradeType` Enum (Line 50)

The three Carr-Wu trade types:
| Type | Position | Bets on |
|------|----------|---------|
| VOL | ATM straddle | σ² vs I² |
| SKEW | Risk-reversal | γ vs b |
| SMILE | Butterfly | ω² vs c |

#### `OptionPosition` Dataclass (Line 57)

Single option position with:
- `strike`, `expiry_days`, `option_type`
- `quantity` (normalized by cash gamma)
- `cash_gamma`, `implied_vol`, `moneyness_l_plus`

#### `CarrWuPortfolio` Dataclass (Line 69)

Complete portfolio containing:
- `trade_type`, `positions`
- Carr-Wu parameters: `implied_variance`, `implied_skew`, `implied_smile`
- Greeks: `portfolio_delta`, `portfolio_vega`, `portfolio_cash_vega`
- `expected_gain_rate`, `delta_hedge_shares`, `vega_hedge_straddles`

#### `CarrWuPortfolioConstructor` Class (Line 100)

Constructs Carr-Wu portfolios from options data.

##### `compute_moneyness_measures(strike, forward, iv, tau)` (Line 128)

Computes Carr-Wu moneyness measures:
```
ℓ_± = ln(K/F) ± σ²τ/2
z_± = ℓ_±/(σ√τ)
```

##### `compute_cash_gamma(forward, strike, iv, tau)` (Line 155)

Computes cash gamma:
```
$Γ = K × N'(z_+) / (σ√τ)
```

##### `construct_vol_trade(...)` (Line 175)

**Vol Trade** (Equation 52):
- Position: Long 2/$Γ_a ATM straddles
- Mean gain rate: G_t = σ² - I_a²

##### `construct_skew_trade(...)` (Line 231)

**Skew Trade** (Equations 58, 65):
- Position: Normalized risk-reversal + vega hedge
- η_p = -1/((ℓ_+^c - ℓ_+^p)$Γ_p) (short OTM put)
- η_c = +1/((ℓ_+^c - ℓ_+^p)$Γ_c) (long OTM call)
- Mean gain rate: G_t = γ - b

##### `construct_smile_trade(...)` (Line 312)

**Smile Trade** (Equations 72, 79):
- Position: Normalized butterfly + vega hedge
- η_p = 1/(ℓ̄²_agt $Γ_p) (long OTM put)
- η_a = -2/(ℓ̄²_agt $Γ_a) (short 2 ATM straddles)
- η_c = 1/(ℓ̄²_agt $Γ_c) (long OTM call)
- Mean gain rate: G_t = ω² - c

---

### Part 2: Kidger Timing Signals (Lines 388-691)

#### `TimingSignal` Dataclass (Line 392)

Timing signal from log-signature analysis:
```python
trade_type: TradeType
signal_value: float
signal_strength: float      # Normalized (0-1)
confidence: float           # Based on consistency
logsig_component: float     # Raw log-signature value
implied_component: float    # Implied from smile
threshold: float            # Entry threshold
interpretation: str         # Human-readable
```

#### `KidgerTimingEngine` Class (Line 411)

Generates timing signals from log-signatures.

**Key insight**: "The log-signature is the optimal summary statistic for predicting how paths drive CDEs."

**Default thresholds**:
| Trade | Threshold | Meaning |
|-------|-----------|---------|
| Vol | 0.10 | 10% deviation triggers signal |
| Skew | 0.005 | 0.5% Lévy area deviation |
| Smile | 0.15 | 15% deviation triggers signal |

##### `extract_vol_signal(logsig, implied_variance, tau, d)` (Line 440)

Vol trade timing signal:
```
Signal = S^(log_S, log_S) / (I² × τ) - 1
```

- S^(log_S, log_S) ≈ realized variance over interval
- When Signal > threshold: Realized variance trending above implied

##### `extract_skew_signal(logsig, implied_skew, tau, d)` (Line 506)

**THE KEY INSIGHT from Kidger's framework**:

Skew trade timing signal:
```
Signal = A^(log_S, σ) / √τ - b
```

Where A^(log_S, σ) is the **Lévy area** capturing ORDER of price/vol moves:
- A > 0: Price moved before vol (classic leverage effect)
- A < 0: Vol moved before price (anticipatory)

##### `extract_smile_signal(logsig, implied_smile, tau, d)` (Line 582)

Smile trade timing signal:
```
Signal = S^(σ, σ) / (c × τ) - 1
```

S^(σ, σ) ≈ vol-of-vol over interval.

##### `generate_all_signals(...)` (Line 657)

Generates signals for all three trade types.

---

### Part 3: Combined Trading Strategy (Lines 693-1520)

#### `TradingDecision` Dataclass (Line 697)

Trading decision combining portfolio and timing:
```python
trade_type: TradeType
action: str              # 'LONG', 'SHORT', 'HOLD'
portfolio: CarrWuPortfolio
timing_signal: TimingSignal
position_size: float     # Risk-adjusted
expected_sharpe: float   # Historical + timing improvement
```

#### `GreeksSnapshot` Dataclass (Line 712)

Portfolio Greeks at entry:
- Delta, Gamma, Vega, Theta
- Vanna, Volga
- Cash Gamma, Cash Vega

From Carr-Wu relationships:
```
Cash Vega = $Γ × σ²τ
Cash Vanna = $Γ × ℓ₊
Cash Volga = $Γ × ℓ₋ℓ₊
Theta = -$Γ × I²/2
```

#### `VolatilityView` Dataclass (Line 745)

Volatility posture of the trade:
- vol_posture, skew_posture, smile_posture
- implied_param, realized_param, edge
- bet_description

#### `PnLScenario` Dataclass (Line 769)

P&L estimate under specific scenario:
- spot_move_pct, iv_move_vol, time_days
- estimated_pnl, pnl_pct, explanation

Uses Carr-Wu P&L attribution:
```
dP&L ≈ Δ×dS + ½Γ×dS² + V×dσ + Θ×dt + Vanna×dS×dσ + ½Volga×dσ²
```

#### `EntryTiming` Dataclass (Line 789)

Entry timing guidance:
- signal_strength, signal_age_minutes
- signal_half_life_minutes, entry_window_minutes
- optimal_entry ('NOW', 'WAIT', 'MISSED', 'FADING')
- degraded_sharpe, recommended_size_adjustment
- recalc_conditions

#### `ExitConditions` Dataclass (Line 812)

Exit conditions:
- take_profit_condition, take_profit_threshold
- stop_loss_condition, stop_loss_threshold
- time_exit_condition, time_exit_days
- signal_reversal_threshold

#### `TradeReport` Dataclass (Line 826)

**Comprehensive trade report** combining all components:
- Identification: timestamp, trade_type, underlying, expiry_days
- Core decision: TradingDecision
- Greeks: GreeksSnapshot
- Volatility view: VolatilityView
- Scenarios: List[PnLScenario]
- Entry timing: EntryTiming
- Exit conditions: ExitConditions
- Market snapshot: spot_price, forward_price, atm_iv

Methods:
- `generate_report(width)`: Formatted ASCII report
- `to_dict()`: Dictionary for serialization

#### `compute_portfolio_greeks(portfolio, forward, tau)` (Line 886)

Computes comprehensive Greeks for a Carr-Wu portfolio.

#### `compute_pnl_scenarios(greeks, forward, atm_iv, tau)` (Line 963)

Computes P&L under 6 scenarios:
1. Base case (time decay only)
2. Spot up 1%, IV -0.5vol
3. Spot down 1%, IV +1vol
4. Vol spike (+3vol)
5. Vol crush (-2vol)
6. Gamma scalp (±2% spot)

#### `compute_entry_timing(signal, step_size, ...)` (Line 1074)

Signal decay model:
- Half-life ≈ step_size × snapshot_interval
- Entry window = 2 × half_life
- Decay factor = 0.5^(age/half_life)

#### `construct_volatility_view(trade_type, portfolio, signal)` (Line 1149)

Constructs volatility view for each trade type.

#### `construct_exit_conditions(trade_type, tau, signal)` (Line 1239)

Constructs exit conditions based on trade type and maturity.

#### `format_trade_report(report, width)` (Line 1288)

Formats TradeReport as ASCII box with sections:
- Market conditions
- Position details
- Greeks at entry
- Volatility view
- P&L scenarios
- Entry timing
- Exit conditions

---

### Part 4: CarrWuKidgerStrategy (Lines 1405-1641)

#### Historical Sharpe Ratios (Line 1416)

From Carr-Wu (2023) Table 4:
| Trade | Sharpe | Timing Improvement |
|-------|--------|-------------------|
| Vol | 0.42 | +55% |
| Skew | 1.38 | +34% |
| Smile | 0.89 | +35% |

#### `generate_decisions(...)` (Line 1443)

Generates trading decisions for all three trade types.

#### `generate_trade_reports(...)` (Line 1521)

**Main entry point for live trading signal generation**.

Returns full TradeReport objects with Greeks, P&L scenarios, entry timing, and exit conditions.

---

### Part 5: Trading Pipeline (Lines 1785-2053)

#### `TradingPipeline` Dataclass (Line 1789)

End-to-end pipeline: CBOE data → preprocessing → trading signals.

Components:
- `path_constructor`: CBOEPathConstructor
- `portfolio_constructor`: CarrWuPortfolioConstructor
- `timing_engine`: KidgerTimingEngine
- `strategy`: CarrWuKidgerStrategy
- `config`: PreprocessingConfig

Methods:
- `reset()`: Reset for new trading day
- `update_path(path_point)`: Add observation, recompute log-signature
- `process_snapshot(cboe_df)`: Process single CBOE snapshot
- `generate_signals(smile_params, tau)`: Generate timing signals
- `process_and_trade(cboe_df, forward, tau)`: Complete pipeline

#### `create_trading_pipeline(...)` (Line 1985)

Factory function to create complete trading pipeline.

**Usage**:
```python
pipeline = create_trading_pipeline()
for snapshot in cboe_snapshots:
    decisions = pipeline.process_and_trade(snapshot, forward=680.0, tau=1/252)
    for dec in decisions:
        print(f"{dec.trade_type}: {dec.action} with size {dec.position_size:.2%}")
```

---

## Key Insights

1. **SKEW TRADE has highest Sharpe** (1.38) because it captures path-dependent risk premia. The Lévy area A^(log_S, σ) provides the timing signal.

2. **VOL TRADE** profits when σ² > I², but has negative average return (paying for variance protection). Timing improves Sharpe by ~55%.

3. **SMILE TRADE** captures vol-of-vol risk premium. S^(σ, σ) forecasts when ω² exceeds implied curvature.

4. **SHORT-DATED OPTIONS** (1-month) have highest Sharpes because path-dependence dominates and forecasting accuracy is highest.

5. **LOG-SIGNATURE is OPTIMAL** for timing because option prices solve CDEs.

---

## Function Reference Table

| Function/Class | Line | Purpose |
|----------------|------|---------|
| `TradeType` | 50 | Trade type enum |
| `CarrWuPortfolio` | 69 | Portfolio container |
| `CarrWuPortfolioConstructor` | 100 | Build Carr-Wu portfolios |
| `TimingSignal` | 392 | Timing signal container |
| `KidgerTimingEngine` | 411 | Generate timing signals |
| `TradingDecision` | 697 | Trading decision |
| `GreeksSnapshot` | 712 | Portfolio Greeks |
| `PnLScenario` | 769 | P&L scenarios |
| `EntryTiming` | 789 | Entry timing guidance |
| `TradeReport` | 826 | **Comprehensive report** |
| `compute_portfolio_greeks` | 886 | Compute Greeks |
| `compute_pnl_scenarios` | 963 | Compute scenarios |
| `compute_entry_timing` | 1074 | Entry timing |
| `format_trade_report` | 1288 | Format report |
| `CarrWuKidgerStrategy` | 1405 | Combined strategy |
| `TradingPipeline` | 1789 | End-to-end pipeline |
| `create_trading_pipeline` | 1985 | Factory function |
