"""
Neural RDE Options Pricing - Unified Entry Point
=================================================

This module provides a unified entry point for the complete Neural RDE pipeline:

    CBOE Data → Preprocessing → Model Training → Trading Signals

Usage:
------
    # Demo mode (synthetic data, shows full pipeline)
    python main.py --mode demo
    
    # Train on CBOE data
    python main.py --mode train --data /path/to/cboe/data --epochs 100
    
    # Analyze CBOE data without training
    python main.py --mode analyze --data /path/to/cboe/data
    
    # Generate trading signals
    python main.py --mode trade --data /path/to/cboe/data --model trained_model.eqx

Architecture:
-------------
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

Author: Based on theoretical framework by Patrick Kidger, Peter Carr, Liuren Wu
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import pickle

import numpy as np

# JAX imports
try:
    import jax
    import jax.numpy as jnp
    import jax.random as jr
    import equinox as eqx
    import optax
    HAS_JAX = True
except ImportError:
    HAS_JAX = False
    warnings.warn("JAX not available — training and model inference disabled")

# Layer 1: Core Math
try:
    from neural_rde_options import (
        NeuralRDE,
        train_model,
        generate_training_batch,
        compute_logsignature,
        logsignature_dimension,
        SignatureGreeks,
        decompose_pnl_signature,
        price_option,
    )
    HAS_CORE = True
except ImportError as e:
    HAS_CORE = False
    warnings.warn(f"Core module not available: {e}")

# Layer 1: Enhanced
try:
    from enhanced_implementation import (
        EnhancedNeuralRDE,
        CarrWuBaseline,
        SignatureGreekMapping,
        compute_smile_formula_extended,
        augment_path_with_jump_indicators,
    )
    HAS_ENHANCED = True
except ImportError:
    HAS_ENHANCED = False

# Layer 2: Data Pipeline
try:
    from cboe_preprocessing import (
        PreprocessingConfig,
        CBOEPathConstructor,
        TrainingDataBuilder,
        PathData,
        SmileParameters,
        TrainingBatch,
        analyze_path_roughness,
        recommend_hyperparameters,
    )
    HAS_PREPROCESSING = True
except ImportError:
    HAS_PREPROCESSING = False

# Layer 2: Integration Bridge
try:
    from kidger_cboe_integration import (
        load_cboe_data,
        preprocess_cboe_for_neural_rde,
        create_training_loop,
        prepare_neural_rde_inputs,
        CBOEDataset,
        validate_cboe_snapshot,
    )
    HAS_INTEGRATION = True
except ImportError:
    HAS_INTEGRATION = False

# Layer 3: Trading
try:
    from carr_wu_kidger_trading import (
        CarrWuPortfolioConstructor,
        KidgerTimingEngine,
        CarrWuKidgerStrategy,
        TradingPipeline,
        create_trading_pipeline,
        TradeType,
        TradingDecision,
    )
    HAS_TRADING = True
except ImportError:
    HAS_TRADING = False


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline."""
    
    # Preprocessing
    step_size: int = 8              # Snapshots per log-signature interval
    depth: int = 2                  # Log-signature truncation depth
    spot_jump_threshold: float = 0.005   # 0.5% for spot jumps
    vol_jump_threshold: float = 0.02     # 2% IV for vol jumps
    
    # Model
    hidden_dim: int = 64
    mlp_width: int = 128
    mlp_depth: int = 3
    use_enhanced: bool = True       # Use EnhancedNeuralRDE with jump indicators
    
    # Training
    n_epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-3
    
    # Trading
    vol_threshold: float = 0.10
    skew_threshold: float = 0.005
    smile_threshold: float = 0.15
    risk_budget: float = 0.10
    min_confidence: float = 0.6
    
    # Paths
    model_path: Optional[str] = None
    output_dir: str = "./output"
    
    def to_preprocessing_config(self) -> PreprocessingConfig:
        """Convert to PreprocessingConfig."""
        return PreprocessingConfig(
            step_size=self.step_size,
            depth=self.depth,
            spot_jump_threshold=self.spot_jump_threshold,
            vol_jump_threshold=self.vol_jump_threshold,
        )


# =============================================================================
# MODEL SAVE/LOAD
# =============================================================================

def save_model(model: Any, path: str, config: Optional[PipelineConfig] = None) -> None:
    """
    Save a trained model to disk.
    
    Args:
        model: Trained NeuralRDE or EnhancedNeuralRDE model
        path: Path to save the model
        config: Optional configuration to save alongside model
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required to save models")
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save model using equinox serialization
    with open(path, 'wb') as f:
        # Get model hyperparameters for reconstruction
        hyperparams = {
            'input_dim': getattr(model, 'input_dim', 5),
            'hidden_dim': model.hidden_dim,
            'logsig_dim': model.logsig_dim,
            'step_size': model.step_size,
            'depth': model.depth,
            'mlp_width': getattr(model, 'mlp_width', 128),
            'mlp_depth': getattr(model, 'mlp_depth', 3),
            'model_type': type(model).__name__,
        }
        if hasattr(model, 'use_jump_indicators'):
            hyperparams['use_jump_indicators'] = model.use_jump_indicators
        
        # Serialize
        data = {
            'hyperparams': hyperparams,
            'model': eqx.tree_serialise_leaves(model),
            'config': config,
        }
        pickle.dump(data, f)
    
    print(f"Model saved to {path}")


def load_model(path: str) -> Tuple[Any, Optional[PipelineConfig]]:
    """
    Load a trained model from disk.
    
    Args:
        path: Path to the saved model
        
    Returns:
        Tuple of (model, config)
    """
    if not HAS_JAX:
        raise RuntimeError("JAX required to load models")
    
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    hyperparams = data['hyperparams']
    model_type = hyperparams.pop('model_type')
    
    # Create model skeleton
    key = jr.PRNGKey(0)  # Dummy key, will be overwritten
    
    if model_type == 'EnhancedNeuralRDE' and HAS_ENHANCED:
        skeleton = EnhancedNeuralRDE(key=key, **hyperparams)
    else:
        # Remove enhanced-specific params if present
        hyperparams.pop('use_jump_indicators', None)
        skeleton = NeuralRDE(key=key, **hyperparams)
    
    # Deserialize weights
    model = eqx.tree_deserialise_leaves(data['model'], skeleton)
    
    print(f"Model loaded from {path}")
    return model, data.get('config')


# =============================================================================
# MODE: DEMO
# =============================================================================

def run_demo(config: PipelineConfig) -> Dict[str, Any]:
    """
    Run a complete demonstration of the pipeline with synthetic data.
    
    This demonstrates all three layers without requiring real CBOE data.
    """
    print("\n" + "=" * 70)
    print("NEURAL RDE OPTIONS PRICING - DEMONSTRATION")
    print("=" * 70)
    
    if not HAS_JAX or not HAS_CORE:
        raise RuntimeError("JAX and core modules required for demo")
    
    results = {}
    key = jr.PRNGKey(42)
    
    # =========================================================================
    # LAYER 1: Core Math Demo
    # =========================================================================
    print("\n" + "-" * 70)
    print("LAYER 1: Core Math (Signature Computation + Model)")
    print("-" * 70)
    
    # Generate synthetic path
    print("\n1.1 Generating synthetic jump-diffusion path...")
    from neural_rde_options import generate_jump_diffusion_path
    
    key, subkey = jr.split(key)
    path, path_info = generate_jump_diffusion_path(
        subkey, n_steps=500, lambda_jump=10.0
    )
    print(f"    Path shape: {path.shape}")
    print(f"    Jumps detected: {path_info['n_jumps']}")
    print(f"    Realized vol (ann.): {np.sqrt(path_info['realized_sigma2']):.2%}")
    
    # Compute log-signatures
    print("\n1.2 Computing log-signatures...")
    from neural_rde_options import compute_logsignatures_for_intervals
    
    logsigs = compute_logsignatures_for_intervals(
        path, step_size=config.step_size, depth=config.depth
    )
    print(f"    Log-signature shape: {logsigs.shape}")
    print(f"    Intervals: {logsigs.shape[0]}")
    print(f"    Features per interval: {logsigs.shape[1]}")
    
    # Create and train model
    print("\n1.3 Creating Neural RDE model...")
    logsig_dim = logsigs.shape[-1]
    
    key, subkey = jr.split(key)
    # Note: EnhancedNeuralRDE has different interface (computes logsigs internally)
    # For training with train_model(), use base NeuralRDE
    model = NeuralRDE(
        input_dim=3,
        hidden_dim=config.hidden_dim,
        logsig_dim=logsig_dim,
        step_size=config.step_size,
        depth=config.depth,
        mlp_width=config.mlp_width,
        mlp_depth=config.mlp_depth,
        key=subkey
    )
    print(f"    Model: NeuralRDE")
    print(f"    Hidden dim: {config.hidden_dim}")
    print(f"    Log-sig dim: {model.logsig_dim}")
    print(f"    Note: EnhancedNeuralRDE available for inference with extended features")
    
    # Quick training demo
    print("\n1.4 Training model (quick demo - 20 epochs)...")
    key, subkey = jr.split(key)
    trained_model, losses = train_model(
        model,
        n_epochs=min(20, config.n_epochs),
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        key=subkey,
        verbose=False
    )
    print(f"    Initial loss: {losses[0]:.6f}")
    print(f"    Final loss: {losses[-1]:.6f}")
    print(f"    Improvement: {(1 - losses[-1]/losses[0]):.1%}")
    
    results['model'] = trained_model
    results['losses'] = losses
    results['path'] = path
    results['logsigs'] = logsigs
    
    # =========================================================================
    # LAYER 2: Data Pipeline Demo (Synthetic CBOE-like)
    # =========================================================================
    if HAS_PREPROCESSING and HAS_INTEGRATION:
        print("\n" + "-" * 70)
        print("LAYER 2: Data Pipeline (Preprocessing)")
        print("-" * 70)
        
        print("\n2.1 Creating synthetic CBOE-like data...")
        # Use the integration module's sample data creator
        from kidger_cboe_integration import _create_sample_cboe_data
        snapshots_by_date = _create_sample_cboe_data()
        
        n_days = len(snapshots_by_date)
        n_snaps = sum(len(v) for v in snapshots_by_date.values())
        print(f"    Days: {n_days}")
        print(f"    Total snapshots: {n_snaps}")
        
        print("\n2.2 Preprocessing for Neural RDE...")
        preproc_config = config.to_preprocessing_config()
        
        # Try to preprocess - may fail with synthetic data lacking 0DTE options
        try:
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                dataset = preprocess_cboe_for_neural_rde(snapshots_by_date, preproc_config)
            
            print(f"\n    Dataset created:")
            print(f"    Days processed: {len(dataset.days)}")
            for date in dataset.days:
                batch = dataset.batches[date]
                smile = batch.smile_params[0]
                print(f"    {date}: γ={smile.gamma:.3f}, ω²={smile.omega2:.3f}")
            
            results['dataset'] = dataset
        except (ValueError, KeyError) as e:
            print(f"\n    Note: Full preprocessing requires real CBOE data with 0DTE options")
            print(f"    Skipping detailed batch construction (synthetic data limitation)")
            print(f"    Layer 2 path construction demonstrated, batch building skipped")
            
            # Still show path construction works
            constructor = CBOEPathConstructor(preproc_config)
            first_date = list(snapshots_by_date.keys())[0]
            try:
                path_data = constructor.process_day(
                    snapshots_by_date[first_date], compute_smile=False
                )
                print(f"\n    Path construction successful:")
                print(f"      Path shape: {path_data.path.shape}")
                print(f"      Log-sig shape: {path_data.logsignatures.shape}")
                results['path_data'] = path_data
            except Exception:
                pass
    
    # =========================================================================
    # LAYER 3: Trading Demo
    # =========================================================================
    if HAS_TRADING:
        print("\n" + "-" * 70)
        print("LAYER 3: Trading (Carr-Wu + Kidger)")
        print("-" * 70)
        
        print("\n3.1 Creating trading pipeline...")
        # Use preprocessing config if available, otherwise create one
        if HAS_PREPROCESSING:
            trading_preproc_config = config.to_preprocessing_config()
        else:
            trading_preproc_config = None
            
        pipeline = create_trading_pipeline(
            config=trading_preproc_config,
            vol_threshold=config.vol_threshold,
            skew_threshold=config.skew_threshold,
            smile_threshold=config.smile_threshold,
            risk_budget=config.risk_budget,
            min_confidence=config.min_confidence
        )
        print(f"    Pipeline created")
        
        print("\n3.2 Generating trading signals...")
        # Use synthetic market conditions
        forward = 680.0
        atm_strike = 680.0
        put_strike = 650.0
        call_strike = 710.0
        atm_iv = 0.18
        put_iv = 0.22
        call_iv = 0.16
        tau = 21/252  # 21 days
        
        # Generate mock log-signature
        mock_logsig = np.array([
            0.1, 0.05, 0.02, 0.0, 0.0,  # Depth 1
            0.001, 0.002, -0.001, 0.0005, 0.0015,
            0.001, 0.0005, 0.0003, 0.0001  # Depth 2
        ])
        
        decisions = pipeline.strategy.generate_decisions(
            forward=forward,
            atm_strike=atm_strike,
            put_strike=put_strike,
            call_strike=call_strike,
            atm_iv=atm_iv,
            put_iv=put_iv,
            call_iv=call_iv,
            tau=tau,
            logsig=mock_logsig
        )
        
        print(f"\n    Market conditions:")
        print(f"      Forward: ${forward:.2f}")
        print(f"      ATM IV: {atm_iv:.1%}")
        print(f"      τ: {tau*252:.0f} days")
        
        print(f"\n    Trading decisions:")
        for dec in decisions:
            print(f"      {dec.trade_type.value}: {dec.action} "
                  f"(size={dec.position_size:.2%})")
        
        results['decisions'] = decisions
        results['pipeline'] = pipeline
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("""
Next steps:
  1. Run with real CBOE data: python main.py --mode train --data /path/to/data
  2. Analyze your data: python main.py --mode analyze --data /path/to/data
  3. Generate signals: python main.py --mode trade --data /path/to/data
  4. See visualization.py for plotting capabilities
  5. See example_quickstart.py for more code examples
""")
    
    return results


# =============================================================================
# MODE: TRAIN
# =============================================================================

def run_train(
    data_dir: Optional[str],
    config: PipelineConfig,
    save_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Train a Neural RDE model on CBOE data (or synthetic if no data provided).
    
    Args:
        data_dir: Path to CBOE data directory (None for synthetic)
        config: Pipeline configuration
        save_path: Path to save trained model
    """
    print("\n" + "=" * 70)
    print("NEURAL RDE OPTIONS PRICING - TRAINING")
    print("=" * 70)
    
    if not HAS_JAX or not HAS_CORE:
        raise RuntimeError("JAX and core modules required for training")
    
    results = {}
    key = jr.PRNGKey(42)
    
    # =========================================================================
    # Load/Generate Data
    # =========================================================================
    if data_dir is not None and HAS_INTEGRATION:
        print(f"\nLoading CBOE data from {data_dir}...")
        snapshots_by_date = load_cboe_data(data_dir)
        
        print("\nPreprocessing...")
        preproc_config = config.to_preprocessing_config()
        dataset = preprocess_cboe_for_neural_rde(snapshots_by_date, preproc_config)
        
        # Prepare inputs for training
        inputs = prepare_neural_rde_inputs(dataset)
        use_cboe_data = True
        
        print(f"\nDataset prepared:")
        print(f"  Days: {len(dataset.days)}")
        print(f"  Log-signature shape: {inputs['logsignatures'].shape}")
        print(f"  Path shape: {inputs['paths'].shape}")
        
    else:
        print("\nUsing synthetic data for training...")
        use_cboe_data = False
        dataset = None
        inputs = None
    
    # =========================================================================
    # Create Model
    # =========================================================================
    print("\nCreating model...")
    key, subkey = jr.split(key)
    
    if use_cboe_data:
        logsig_dim = inputs['logsignatures'].shape[-1]
    else:
        logsig_dim = logsignature_dimension(3, config.depth)
    
    # Note: EnhancedNeuralRDE has different __call__ interface (computes logsigs internally)
    # For compatibility with train_model(), always use base NeuralRDE for training
    model = NeuralRDE(
        input_dim=3,
        hidden_dim=config.hidden_dim,
        logsig_dim=logsig_dim,
        step_size=config.step_size,
        depth=config.depth,
        mlp_width=config.mlp_width,
        mlp_depth=config.mlp_depth,
        key=subkey
    )
    print(f"  Model: NeuralRDE")
    print(f"  Hidden dim: {config.hidden_dim}")
    print(f"  Log-sig dim: {model.logsig_dim}")
    
    # =========================================================================
    # Train
    # =========================================================================
    print(f"\nTraining for {config.n_epochs} epochs...")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print()
    
    key, subkey = jr.split(key)
    
    if use_cboe_data:
        # Training with CBOE data
        # Convert numpy arrays to JAX arrays
        trained_model, losses = _train_with_cboe_data(
            model, inputs, config, subkey
        )
    else:
        # Training with synthetic data
        trained_model, losses = train_model(
            model,
            n_epochs=config.n_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            key=subkey,
            verbose=True
        )
    
    print(f"\nTraining complete:")
    print(f"  Initial loss: {losses[0]:.6f}")
    print(f"  Final loss: {losses[-1]:.6f}")
    print(f"  Improvement: {(1 - losses[-1]/losses[0]):.1%}")
    
    results['model'] = trained_model
    results['losses'] = losses
    results['dataset'] = dataset
    
    # =========================================================================
    # Save Model
    # =========================================================================
    if save_path is not None:
        save_model(trained_model, save_path, config)
        results['model_path'] = save_path
    
    return results


def _train_with_cboe_data(
    model: Any,
    inputs: Dict[str, np.ndarray],
    config: PipelineConfig,
    key: jr.PRNGKey
) -> Tuple[Any, List[float]]:
    """
    Train model using preprocessed CBOE data.
    
    This adapts the training loop to use real data instead of synthetic.
    """
    from neural_rde_options import total_loss, create_train_step
    
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    
    loss_weights = {
        'lambda_pnl': 1.0,
        'lambda_forecast': 1.0,
        'lambda_arbitrage': 0.1,
        'lambda_iv': 1.0
    }
    
    train_step = create_train_step(model, optimizer, loss_weights)
    
    # Convert inputs to JAX arrays
    paths = jnp.array(inputs['paths'])
    logsigs = jnp.array(inputs['logsignatures'])
    realized_var = jnp.array(inputs['realized_variance'])
    target_iv = jnp.array(inputs['target_iv'])
    moneyness = jnp.array(inputs['moneyness'])
    tau = jnp.array(inputs['tau'])
    
    n_samples = paths.shape[0]
    losses = []
    
    for epoch in range(config.n_epochs):
        key, subkey = jr.split(key)
        
        # Sample batch indices
        if n_samples > config.batch_size:
            indices = jr.choice(subkey, n_samples, (config.batch_size,), replace=False)
        else:
            indices = jnp.arange(n_samples)
        
        # Build batch
        batch = {
            'path': paths[indices],
            'logsigs': logsigs[indices],
            'realized_sigma2': realized_var[indices],
            'realized_gamma': jnp.array(inputs['smile_gamma'])[indices],
            'realized_omega2': jnp.array(inputs['smile_omega2'])[indices],
            'target_iv': target_iv[indices],
            'moneyness': moneyness[indices] if moneyness.ndim > 1 else moneyness,
            'tau': tau[indices] if tau.ndim > 0 else tau,
        }
        
        model, opt_state, loss = train_step(model, opt_state, batch)
        losses.append(float(loss))
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{config.n_epochs}, Loss: {loss:.6f}")
    
    return model, losses


# =============================================================================
# MODE: ANALYZE
# =============================================================================

def run_analyze(data_dir: str, config: PipelineConfig) -> Dict[str, Any]:
    """
    Analyze CBOE data without training.
    
    Provides statistics on:
    - Data quality
    - Path characteristics
    - Smile parameters
    - Roughness indicators
    """
    print("\n" + "=" * 70)
    print("NEURAL RDE OPTIONS PRICING - DATA ANALYSIS")
    print("=" * 70)
    
    if not HAS_PREPROCESSING or not HAS_INTEGRATION:
        raise RuntimeError("Preprocessing modules required for analysis")
    
    results = {}
    
    # Load data
    print(f"\nLoading data from {data_dir}...")
    snapshots_by_date = load_cboe_data(data_dir)
    
    n_days = len(snapshots_by_date)
    n_snaps = sum(len(v) for v in snapshots_by_date.values())
    print(f"  Days: {n_days}")
    print(f"  Total snapshots: {n_snaps}")
    print(f"  Avg snapshots/day: {n_snaps/n_days:.1f}")
    
    # Validate first day
    first_date = sorted(snapshots_by_date.keys())[0]
    first_snap = snapshots_by_date[first_date][0]
    
    print(f"\nData quality ({first_date}):")
    checks = validate_cboe_snapshot(first_snap)
    for check, passed in checks.items():
        status = "✓" if passed else "✗"
        print(f"  {status} {check}")
    
    # Preprocess
    print("\nPreprocessing...")
    preproc_config = config.to_preprocessing_config()
    dataset = preprocess_cboe_for_neural_rde(snapshots_by_date, preproc_config)
    
    # Aggregate statistics
    print("\nSmile parameter statistics:")
    gammas = [dataset.batches[d].smile_params[0].gamma for d in dataset.days]
    omega2s = [dataset.batches[d].smile_params[0].omega2 for d in dataset.days]
    atm_ivs = [dataset.batches[d].smile_params[0].atm_iv for d in dataset.days]
    
    print(f"  γ (skew):     mean={np.mean(gammas):.4f}, std={np.std(gammas):.4f}")
    print(f"  ω² (smile):   mean={np.mean(omega2s):.4f}, std={np.std(omega2s):.4f}")
    print(f"  ATM IV:       mean={np.mean(atm_ivs):.2%}, std={np.std(atm_ivs):.2%}")
    
    print("\nJump statistics:")
    spot_jumps = [dataset.path_data[d].n_spot_jumps for d in dataset.days]
    vol_jumps = [dataset.path_data[d].n_vol_jumps for d in dataset.days]
    
    print(f"  Spot jumps/day: mean={np.mean(spot_jumps):.1f}, max={max(spot_jumps)}")
    print(f"  Vol jumps/day:  mean={np.mean(vol_jumps):.1f}, max={max(vol_jumps)}")
    
    print("\nPath roughness:")
    all_paths = np.vstack([dataset.path_data[d].path for d in dataset.days])
    roughness = analyze_path_roughness(all_paths)
    for key, value in roughness.items():
        print(f"  {key}: {value:.6f}")
    
    print("\nRecommended hyperparameters:")
    n_snaps_day = len(snapshots_by_date[first_date])
    recs = recommend_hyperparameters(n_snaps_day, roughness, 'balanced')
    for key, value in recs.items():
        print(f"  {key}: {value}")
    
    results['dataset'] = dataset
    results['roughness'] = roughness
    results['recommendations'] = recs
    
    return results


# =============================================================================
# MODE: TRADE
# =============================================================================

def run_trade(
    data_dir: str,
    config: PipelineConfig,
    model_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate trading signals from CBOE data.
    
    Args:
        data_dir: Path to CBOE data
        config: Pipeline configuration
        model_path: Path to trained model (optional, uses timing engine without model)
    """
    print("\n" + "=" * 70)
    print("NEURAL RDE OPTIONS PRICING - TRADING SIGNALS")
    print("=" * 70)
    
    if not HAS_PREPROCESSING or not HAS_TRADING:
        raise RuntimeError("Preprocessing and trading modules required")
    
    results = {'decisions_by_day': {}}
    
    # Load model if provided
    model = None
    if model_path is not None:
        print(f"\nLoading model from {model_path}...")
        model, saved_config = load_model(model_path)
        if saved_config is not None:
            config = saved_config
    
    # Load data
    print(f"\nLoading data from {data_dir}...")
    snapshots_by_date = load_cboe_data(data_dir)
    
    # Create pipeline
    print("\nCreating trading pipeline...")
    preproc_config = config.to_preprocessing_config()
    pipeline = create_trading_pipeline(
        config=preproc_config,
        vol_threshold=config.vol_threshold,
        skew_threshold=config.skew_threshold,
        smile_threshold=config.smile_threshold,
        risk_budget=config.risk_budget,
        min_confidence=config.min_confidence
    )
    
    # Process each day
    print("\nGenerating trading signals:")
    print("-" * 70)
    
    for date_str in sorted(snapshots_by_date.keys()):
        snapshots = snapshots_by_date[date_str]
        
        # Reset pipeline for new day
        pipeline.reset()
        
        # Process snapshots
        for snap in snapshots:
            spot = snap['active_underlying_price'].iloc[0]
            path_point, smile_params = pipeline.process_snapshot(snap)
        
        # Get final log-signature for day
        if pipeline._current_logsig is not None:
            # Generate signals using final snapshot's data
            forward = spot  # Simplified: F ≈ S
            tau = 1/252  # 0DTE
            
            # Use smile parameters for implied quantities
            atm_iv = smile_params.atm_iv
            put_iv = atm_iv * 1.1  # Approximate
            call_iv = atm_iv * 0.95
            
            decisions = pipeline.strategy.generate_decisions(
                forward=forward,
                atm_strike=forward,
                put_strike=forward * 0.95,
                call_strike=forward * 1.05,
                atm_iv=atm_iv,
                put_iv=put_iv,
                call_iv=call_iv,
                tau=tau,
                logsig=pipeline._current_logsig
            )
            
            results['decisions_by_day'][date_str] = decisions
            
            print(f"\n{date_str} (spot=${spot:.2f}, IV={atm_iv:.1%}):")
            for dec in decisions:
                print(f"  {dec.trade_type.value:6s}: {dec.action:5s} "
                      f"(size={dec.position_size:.2%})")
        else:
            print(f"\n{date_str}: Insufficient data for signals")
    
    print("\n" + "-" * 70)
    print("Trading signal generation complete")
    
    results['pipeline'] = pipeline
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Neural RDE Options Pricing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode demo                    # Full demo with synthetic data
  python main.py --mode train --epochs 100      # Train on synthetic data
  python main.py --mode train --data ./cboe     # Train on CBOE data
  python main.py --mode analyze --data ./cboe   # Analyze CBOE data
  python main.py --mode trade --data ./cboe     # Generate trading signals
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['demo', 'train', 'analyze', 'trade'],
        default='demo',
        help='Operating mode'
    )
    parser.add_argument(
        '--data', 
        type=str, 
        default=None,
        help='Path to CBOE data directory'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (for trade mode, or to continue training)'
    )
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save trained model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Training batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--step-size',
        type=int,
        default=8,
        help='Log-signature step size'
    )
    parser.add_argument(
        '--depth',
        type=int,
        default=2,
        help='Log-signature truncation depth'
    )
    parser.add_argument(
        '--no-enhanced',
        action='store_true',
        help='Use basic NeuralRDE instead of EnhancedNeuralRDE'
    )
    
    args = parser.parse_args()
    
    # Build configuration
    config = PipelineConfig(
        step_size=args.step_size,
        depth=args.depth,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        use_enhanced=not args.no_enhanced,
        model_path=args.model,
    )
    
    # Check dependencies
    print("\nChecking dependencies...")
    print(f"  JAX:           {'✓' if HAS_JAX else '✗'}")
    print(f"  Core:          {'✓' if HAS_CORE else '✗'}")
    print(f"  Enhanced:      {'✓' if HAS_ENHANCED else '✗'}")
    print(f"  Preprocessing: {'✓' if HAS_PREPROCESSING else '✗'}")
    print(f"  Integration:   {'✓' if HAS_INTEGRATION else '✗'}")
    print(f"  Trading:       {'✓' if HAS_TRADING else '✗'}")
    
    # Run selected mode
    if args.mode == 'demo':
        results = run_demo(config)
        
    elif args.mode == 'train':
        results = run_train(args.data, config, save_path=args.save)
        
    elif args.mode == 'analyze':
        if args.data is None:
            parser.error("--data required for analyze mode")
        results = run_analyze(args.data, config)
        
    elif args.mode == 'trade':
        if args.data is None:
            parser.error("--data required for trade mode")
        results = run_trade(args.data, config, model_path=args.model)
    
    return results


if __name__ == "__main__":
    main()
