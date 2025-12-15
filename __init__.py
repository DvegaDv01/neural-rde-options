"""
Neural RDE Options Pricing Package
===================================

A JAX-based implementation of Neural Rough Differential Equations
for pricing short-dated options, with integrated CBOE data pipeline
and Carr-Wu trading framework.

Main Components:
================

CORE (neural_rde_options.py):
- Signature computation (signature, log-signature, Lévy area)
- Neural RDE architecture (NeuralRDE)
- Signature-based Greeks (including Libra)
- Training utilities

ENHANCED (enhanced_implementation.py):
- EnhancedNeuralRDE with jump indicators
- Extended smile formula with Lévy area term
- Carr-Wu baseline comparison
- No-arbitrage constraints

PREPROCESSING (cboe_preprocessing.py):
- CBOE data loading and path construction
- Log-signature preprocessing
- Smile parameter extraction
- Training batch preparation

TRADING (carr_wu_kidger_trading.py):
- Carr-Wu portfolio construction (vol/skew/smile trades)
- Kidger timing signals from log-signatures
- Combined trading strategy
- End-to-end trading pipeline

Usage:
======
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
"""

import warnings

# =============================================================================
# CORE: neural_rde_options.py
# =============================================================================
from .neural_rde_options import (
    # Signature computation
    compute_signature,
    compute_signature_depth1,
    compute_signature_depth2,
    compute_logsignature,
    compute_levy_area,
    compute_logsignatures_for_intervals,
    logsignature_dimension,
    batch_compute_logsignatures,
    
    # Model components
    VectorField,
    InitialEncoder,
    OutputDecoder,
    NeuralRDE,
    
    # Greeks
    SignatureGreeks,
    decompose_pnl_signature,
    compute_libra_greek,
    
    # Loss functions
    compute_pnl_loss,
    compute_forecast_loss,
    compute_no_arbitrage_loss,
    compute_implied_vol_loss,
    total_loss,
    
    # Data generation
    generate_jump_diffusion_path,
    generate_training_batch,
    
    # Training
    train_model,
    create_train_step,
    
    # Inference
    price_option,
    black_scholes_call,
    compute_smile_surface,
)

# =============================================================================
# ENHANCED: enhanced_implementation.py
# =============================================================================
try:
    from .enhanced_implementation import (
        # Enhanced model
        EnhancedNeuralRDE,
        EnhancedOutputDecoder,
        
        # Greek mapping
        SignatureGreekMapping,
        extract_greeks_from_signature,
        
        # Extended smile
        compute_smile_formula_extended,
        analyze_atm_skew_explosion,
        
        # Path augmentation
        augment_path_with_jump_indicators,
        compute_jump_signature_features,
        
        # Forecasting
        interpret_depth3_for_forecasting,
        
        # Carr-Wu baseline
        CarrWuBaseline,
        
        # Enhanced loss
        compute_enhanced_total_loss,
        compute_terminal_condition_loss,
        compute_drift_constraint_loss,
    )
    _HAS_ENHANCED = True
except ImportError as e:
    _HAS_ENHANCED = False
    warnings.warn(f"Enhanced implementation not available: {e}")

# =============================================================================
# PREPROCESSING: cboe_preprocessing.py
# =============================================================================
try:
    from .cboe_preprocessing import (
        # Configuration
        PreprocessingConfig,
        
        # Data structures
        PathData,
        SmileParameters,
        TrainingBatch,
        
        # Path construction
        CBOEPathConstructor,
        
        # Training data
        TrainingDataBuilder,
        
        # Online processing
        OnlinePathProcessor,
        
        # Utilities
        logsig_dimension,
        compute_logsignature as cboe_compute_logsignature,
        compute_levy_area as cboe_compute_levy_area,
        analyze_path_roughness,
        recommend_hyperparameters,
    )
    _HAS_PREPROCESSING = True
except ImportError as e:
    _HAS_PREPROCESSING = False
    warnings.warn(f"CBOE preprocessing not available: {e}")

# =============================================================================
# TRADING: carr_wu_kidger_trading.py
# =============================================================================
try:
    from .carr_wu_kidger_trading import (
        # Trade types
        TradeType,
        
        # Portfolio construction
        OptionPosition,
        CarrWuPortfolio,
        CarrWuPortfolioConstructor,
        
        # Timing signals
        TimingSignal,
        KidgerTimingEngine,
        
        # Strategy
        TradingDecision,
        CarrWuKidgerStrategy,
        
        # Pipeline (main entry point)
        TradingPipeline,
        create_trading_pipeline,
    )
    _HAS_TRADING = True
except ImportError as e:
    _HAS_TRADING = False
    warnings.warn(f"Trading module not available: {e}")

# =============================================================================
# INTEGRATION: kidger_cboe_integration.py
# =============================================================================
try:
    from .kidger_cboe_integration import (
        CBOEDataset,
        load_cboe_data,
        create_training_loop,
        prepare_neural_rde_inputs,
        interpret_logsignature,
        KidgerInterpretation,
    )
    _HAS_INTEGRATION = True
except ImportError as e:
    _HAS_INTEGRATION = False
    warnings.warn(f"CBOE integration not available: {e}")


# =============================================================================
# VERSION AND METADATA
# =============================================================================
__version__ = "0.2.0"
__author__ = "Based on theoretical framework by Patrick Kidger et al."

__all__ = [
    # === CORE ===
    # Signature computation
    'compute_signature',
    'compute_signature_depth1',
    'compute_signature_depth2',
    'compute_logsignature',
    'compute_levy_area',
    'compute_logsignatures_for_intervals',
    'logsignature_dimension',
    'batch_compute_logsignatures',
    
    # Model components
    'VectorField',
    'InitialEncoder',
    'OutputDecoder',
    'NeuralRDE',
    
    # Greeks
    'SignatureGreeks',
    'decompose_pnl_signature',
    'compute_libra_greek',
    
    # Loss functions
    'compute_pnl_loss',
    'compute_forecast_loss',
    'compute_no_arbitrage_loss',
    'compute_implied_vol_loss',
    'total_loss',
    
    # Data generation
    'generate_jump_diffusion_path',
    'generate_training_batch',
    
    # Training
    'train_model',
    'create_train_step',
    
    # Inference
    'price_option',
    'black_scholes_call',
    'compute_smile_surface',
    
    # === ENHANCED ===
    'EnhancedNeuralRDE',
    'EnhancedOutputDecoder',
    'SignatureGreekMapping',
    'extract_greeks_from_signature',
    'compute_smile_formula_extended',
    'analyze_atm_skew_explosion',
    'augment_path_with_jump_indicators',
    'compute_jump_signature_features',
    'interpret_depth3_for_forecasting',
    'CarrWuBaseline',
    'compute_enhanced_total_loss',
    'compute_terminal_condition_loss',
    'compute_drift_constraint_loss',
    
    # === PREPROCESSING ===
    'PreprocessingConfig',
    'PathData',
    'SmileParameters',
    'TrainingBatch',
    'CBOEPathConstructor',
    'TrainingDataBuilder',
    'OnlinePathProcessor',
    'logsig_dimension',
    'analyze_path_roughness',
    'recommend_hyperparameters',
    
    # === TRADING ===
    'TradeType',
    'OptionPosition',
    'CarrWuPortfolio',
    'CarrWuPortfolioConstructor',
    'TimingSignal',
    'KidgerTimingEngine',
    'TradingDecision',
    'CarrWuKidgerStrategy',
    'TradingPipeline',
    'create_trading_pipeline',
    
    # === INTEGRATION ===
    'CBOEDataset',
    'load_cboe_data',
    'create_training_loop',
    'prepare_neural_rde_inputs',
    'interpret_logsignature',
    'KidgerInterpretation',
]


def check_components():
    """Print status of available components."""
    print("Neural RDE Options Package - Component Status")
    print("=" * 50)
    print(f"  Core (neural_rde_options.py):      ✓ Available")
    print(f"  Enhanced (enhanced_implementation): {'✓ Available' if _HAS_ENHANCED else '✗ Not available'}")
    print(f"  Preprocessing (cboe_preprocessing): {'✓ Available' if _HAS_PREPROCESSING else '✗ Not available'}")
    print(f"  Trading (carr_wu_kidger_trading):   {'✓ Available' if _HAS_TRADING else '✗ Not available'}")
    print(f"  Integration (kidger_cboe_integration): {'✓ Available' if _HAS_INTEGRATION else '✗ Not available'}")
    print("=" * 50)
