"""
Neural RDE Options Pricing Package
===================================

A JAX-based implementation of Neural Rough Differential Equations
for pricing short-dated options.

Main Components:
- Signature computation (signature, log-signature, Lévy area)
- Neural RDE architecture
- Signature-based Greeks (including Libra)
- Training utilities
- Visualization tools
"""

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

__version__ = "0.1.0"
__author__ = "Based on theoretical framework by Patrick Kidger et al."

__all__ = [
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
]
