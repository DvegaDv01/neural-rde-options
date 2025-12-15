#!/usr/bin/env python3
"""
Diagnostic script to show all trading signals regardless of threshold.
This helps understand why no actionable signals were generated.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from main import load_cboe_data, PipelineConfig
from carr_wu_kidger_trading import (
    create_trading_pipeline,
    CarrWuPortfolioConstructor,
    KidgerTimingEngine,
    TradeType,
)


def diagnose_signals(data_dir: str):
    """Show all signals even if below threshold."""

    print("=" * 70)
    print("SIGNAL DIAGNOSTIC - SHOWING ALL SIGNALS (INCLUDING BELOW THRESHOLD)")
    print("=" * 70)

    # Load data
    print(f"\nLoading data from {data_dir}...")
    snapshots_by_date = load_cboe_data(data_dir)

    config = PipelineConfig()
    preproc_config = config.to_preprocessing_config()

    # Create pipeline with very low thresholds to capture everything
    pipeline = create_trading_pipeline(
        config=preproc_config,
        vol_threshold=0.001,      # Very low
        skew_threshold=0.0001,    # Very low
        smile_threshold=0.001,    # Very low
        risk_budget=0.10,
        min_confidence=0.01       # Very low
    )

    # Also create normal timing engine to show values vs thresholds
    normal_timing = KidgerTimingEngine(
        vol_threshold=0.10,
        skew_threshold=0.005,
        smile_threshold=0.15
    )

    constructor = CarrWuPortfolioConstructor(target_moneyness=0.05)

    for date_str in sorted(snapshots_by_date.keys()):
        snapshots = snapshots_by_date[date_str]

        print(f"\n{'=' * 70}")
        print(f"Date: {date_str}  ({len(snapshots)} snapshots)")
        print("=" * 70)

        # Reset pipeline for new day
        pipeline.reset()

        # Process snapshots
        for snap in snapshots:
            spot = snap['active_underlying_price'].iloc[0]
            path_point, smile_params = pipeline.process_snapshot(snap)

        # Get final log-signature for day
        if pipeline._current_logsig is not None:
            logsig = pipeline._current_logsig

            print(f"\nSpot: ${spot:.2f}")
            print(f"ATM IV: {smile_params.atm_iv:.2%}")
            print(f"Gamma (γ): {smile_params.gamma:.6f}")
            print(f"Omega² (ω²): {smile_params.omega2:.6f}")

            print(f"\nLog-signature shape: {logsig.shape}")
            print(f"Log-signature values:")
            print(f"  Depth 1 (first 5): {logsig[:5]}")
            if len(logsig) > 5:
                print(f"  Depth 2 Lévy areas: {logsig[5:15]}")
            if len(logsig) > 15:
                print(f"  Depth 3 (if any): {logsig[15:]}")

            # Build portfolios
            forward = spot
            tau = 1/252
            atm_iv = smile_params.atm_iv
            put_iv = atm_iv * 1.1
            call_iv = atm_iv * 0.95

            vol_port = constructor.construct_vol_trade(forward, forward, atm_iv, tau)
            skew_port = constructor.construct_skew_trade(
                forward, forward, forward*0.95, forward*1.05,
                atm_iv, put_iv, call_iv, tau
            )
            smile_port = constructor.construct_smile_trade(
                forward, forward, forward*0.95, forward*1.05,
                atm_iv, put_iv, call_iv, tau
            )

            # Get signals from normal timing engine
            signals = normal_timing.generate_all_signals(
                logsig, vol_port, skew_port, smile_port, tau
            )

            print("\n" + "-" * 70)
            print("SIGNAL VALUES VS THRESHOLDS")
            print("-" * 70)
            print(f"{'Trade':<8} {'Signal':>10} {'Threshold':>10} {'Strength':>10} {'Confidence':>10} {'Status':<15}")
            print("-" * 70)

            for trade_type, signal in signals.items():
                status = "✓ ACTIONABLE" if abs(signal.signal_value) > signal.threshold and signal.confidence >= 0.6 else "✗ Below threshold"
                print(f"{trade_type.value:<8} {signal.signal_value:>+10.4f} {signal.threshold:>10.4f} "
                      f"{signal.signal_strength:>10.2f} {signal.confidence:>10.2f} {status:<15}")

            print("\nSignal interpretations:")
            for trade_type, signal in signals.items():
                print(f"  {trade_type.value}: {signal.interpretation}")
        else:
            print("\n  Insufficient data for signals")

    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)
    print("""
To generate actionable signals, the data needs:
- Vol signal > 0.10 (10% deviation of realized vs implied variance)
- Skew signal > 0.005 (0.5% Lévy area deviation)
- Smile signal > 0.15 (15% deviation of vol-of-vol vs implied)
- Confidence ≥ 0.60 (consistent signal direction)

A calm day with low volatility will not generate actionable signals.
Consider loading data from more volatile days or adjusting thresholds.
""")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/s3_cache"
    diagnose_signals(data_dir)
