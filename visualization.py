"""
Visualization utilities for Neural RDE options pricing.

This module provides plotting functions to visualize:
1. Sample paths and their signatures
2. Training progress
3. Implied volatility surfaces
4. P&L attribution
"""

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Optional, List, Tuple
import numpy as np

# Import from main module
from neural_rde_options import (
    NeuralRDE,
    compute_signature,
    compute_logsignature,
    compute_levy_area,
    compute_logsignatures_for_intervals,
    generate_jump_diffusion_path,
    logsignature_dimension,
    price_option,
    train_model,
    SignatureGreeks,
    decompose_pnl_signature
)


def plot_path_with_signatures(
    path: jnp.ndarray,
    step_size: int = 32,
    depth: int = 2,
    figsize: Tuple[int, int] = (14, 10)
):
    """
    Visualize a path along with its signature decomposition.
    
    Args:
        path: Array of shape (n_steps, 3) with (time, log_price, vol)
        step_size: Interval size for log-signatures
        depth: Log-signature depth
        figsize: Figure size
    """
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, 2, figure=fig)
    
    t = path[:, 0]
    log_S = path[:, 1]
    sigma = path[:, 2]
    
    # Plot 1: Log-price path
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, log_S, 'b-', linewidth=0.8)
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Log Price')
    ax1.set_title('Log-Price Path')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Volatility path
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, sigma, 'r-', linewidth=0.8)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Volatility')
    ax2.set_title('Volatility Path')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Price-Vol phase diagram
    ax3 = fig.add_subplot(gs[1, 0])
    colors = np.linspace(0, 1, len(log_S))
    ax3.scatter(log_S, sigma, c=colors, cmap='viridis', s=1, alpha=0.5)
    ax3.plot(log_S, sigma, 'k-', alpha=0.3, linewidth=0.5)
    ax3.set_xlabel('Log Price')
    ax3.set_ylabel('Volatility')
    ax3.set_title('Price-Vol Phase Diagram (color = time)')
    ax3.grid(True, alpha=0.3)
    
    # Compute Lévy area for the full path (price-vol)
    levy_area = compute_levy_area(path[:, 1:])
    ax3.text(0.05, 0.95, f'Lévy Area: {levy_area[0, 1]:.4f}', 
             transform=ax3.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Plot 4: Log-signatures over intervals
    logsigs = compute_logsignatures_for_intervals(path, step_size, depth)
    n_intervals = logsigs.shape[0]
    
    ax4 = fig.add_subplot(gs[1, 1])
    im = ax4.imshow(logsigs.T, aspect='auto', cmap='RdBu', 
                    interpolation='nearest', vmin=-np.max(np.abs(logsigs)), 
                    vmax=np.max(np.abs(logsigs)))
    ax4.set_xlabel('Interval')
    ax4.set_ylabel('Log-signature Component')
    ax4.set_title(f'Log-signatures (depth={depth}, step={step_size})')
    plt.colorbar(im, ax=ax4)
    
    # Plot 5: Signature component time series
    ax5 = fig.add_subplot(gs[2, 0])
    interval_times = np.arange(n_intervals) * step_size * (t[1] - t[0])
    
    # Plot depth-1 components (increments)
    d = path.shape[1]
    for i in range(d):
        ax5.plot(interval_times, logsigs[:, i], label=f'Increment {i}', linewidth=1)
    
    ax5.set_xlabel('Time')
    ax5.set_ylabel('Value')
    ax5.set_title('Depth-1 Log-signature (Increments)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Lévy areas over intervals
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Extract Lévy area components (they start after depth-1)
    levy_start_idx = d
    levy_areas_interval = []
    for i in range(n_intervals):
        interval_start = i * step_size
        interval_end = min((i + 1) * step_size + 1, len(path))
        interval_path = path[interval_start:interval_end, 1:]
        la = compute_levy_area(interval_path)
        levy_areas_interval.append(la[0, 1])  # Price-vol Lévy area
    
    ax6.bar(interval_times, levy_areas_interval, width=interval_times[1]-interval_times[0] if len(interval_times) > 1 else 0.01,
            alpha=0.7, color='purple')
    ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax6.set_xlabel('Time')
    ax6.set_ylabel('Lévy Area')
    ax6.set_title('Price-Vol Lévy Area per Interval')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_training_progress(losses: List[float], figsize: Tuple[int, int] = (10, 4)):
    """
    Plot training loss over epochs.
    
    Args:
        losses: List of loss values
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    epochs = np.arange(1, len(losses) + 1)
    
    # Linear scale
    ax1.plot(epochs, losses, 'b-', linewidth=1)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    
    # Log scale
    ax2.semilogy(epochs, losses, 'b-', linewidth=1)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training Loss (Log Scale)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_implied_vol_surface(
    model: NeuralRDE,
    path: jnp.ndarray,
    spot: float,
    strike_range: Tuple[float, float] = (0.9, 1.1),
    tau_range: Tuple[float, float] = (1/252, 5/252),
    n_strikes: int = 20,
    n_taus: int = 10,
    implied_vol_atm: float = 0.2,
    figsize: Tuple[int, int] = (12, 5)
):
    """
    Plot the implied volatility surface generated by the model.
    
    Args:
        model: Trained NeuralRDE
        path: Recent price/vol history
        spot: Current spot price
        strike_range: (min, max) as fraction of spot
        tau_range: (min, max) maturities in years
        n_strikes: Number of strike points
        n_taus: Number of maturity points
        implied_vol_atm: ATM IV for normalization
        figsize: Figure size
    """
    strikes = np.linspace(spot * strike_range[0], spot * strike_range[1], n_strikes)
    taus = np.linspace(tau_range[0], tau_range[1], n_taus)
    
    # Compute IV surface
    logsigs = compute_logsignatures_for_intervals(path, model.step_size, model.depth)
    
    iv_surface = np.zeros((n_strikes, n_taus))
    for i, K in enumerate(strikes):
        for j, tau in enumerate(taus):
            moneyness = (np.log(K / spot) + 0.5 * implied_vol_atm**2 * tau) / (implied_vol_atm * np.sqrt(tau))
            outputs = model(path, logsigs, jnp.array(moneyness), jnp.array(tau))
            iv_surface[i, j] = float(outputs['implied_vol'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Surface plot
    X, Y = np.meshgrid(taus * 252, strikes / spot)  # Convert to days and moneyness
    
    im = ax1.contourf(X, Y, iv_surface, levels=20, cmap='viridis')
    ax1.set_xlabel('Time to Maturity (days)')
    ax1.set_ylabel('Strike / Spot')
    ax1.set_title('Implied Volatility Surface')
    plt.colorbar(im, ax=ax1, label='IV')
    
    # Smile slices
    colors = plt.cm.viridis(np.linspace(0, 1, n_taus))
    for j, tau in enumerate(taus):
        ax2.plot(strikes / spot, iv_surface[:, j], color=colors[j], 
                 label=f'{tau*252:.1f}d', linewidth=1.5)
    
    ax2.set_xlabel('Strike / Spot')
    ax2.set_ylabel('Implied Volatility')
    ax2.set_title('IV Smile by Maturity')
    ax2.legend(title='Maturity', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_greeks_comparison(
    model: NeuralRDE,
    path: jnp.ndarray,
    spot: float,
    strike_range: Tuple[float, float] = (0.9, 1.1),
    tau: float = 1/252,
    n_strikes: int = 50,
    implied_vol_atm: float = 0.2,
    figsize: Tuple[int, int] = (14, 8)
):
    """
    Plot the signature Greeks across strikes.
    
    Args:
        model: Trained NeuralRDE
        path: Recent price/vol history
        spot: Current spot price
        strike_range: (min, max) as fraction of spot
        tau: Time to maturity
        n_strikes: Number of strike points
        implied_vol_atm: ATM IV
        figsize: Figure size
    """
    strikes = np.linspace(spot * strike_range[0], spot * strike_range[1], n_strikes)
    moneyness_values = []
    
    # Collect Greeks
    thetas, deltas, gammas, vegas, libras = [], [], [], [], []
    
    logsigs = compute_logsignatures_for_intervals(path, model.step_size, model.depth)
    
    for K in strikes:
        moneyness = (np.log(K / spot) + 0.5 * implied_vol_atm**2 * tau) / (implied_vol_atm * np.sqrt(tau))
        moneyness_values.append(moneyness)
        
        outputs = model(path, logsigs, jnp.array(moneyness), jnp.array(tau))
        greeks = outputs['greeks']
        
        thetas.append(float(greeks[0]))
        deltas.append(float(greeks[1]))
        gammas.append(float(greeks[2]))
        vegas.append(float(greeks[3]))
        libras.append(float(greeks[4]))
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
    moneyness_values = np.array(moneyness_values)
    
    # Theta
    axes[0, 0].plot(moneyness_values, thetas, 'b-', linewidth=1.5)
    axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 0].set_xlabel('Moneyness (z₊)')
    axes[0, 0].set_ylabel('Theta (Θ)')
    axes[0, 0].set_title('Theta')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Delta
    axes[0, 1].plot(moneyness_values, deltas, 'g-', linewidth=1.5)
    axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 1].set_xlabel('Moneyness (z₊)')
    axes[0, 1].set_ylabel('Delta (Δ)')
    axes[0, 1].set_title('Delta')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Gamma
    axes[0, 2].plot(moneyness_values, gammas, 'r-', linewidth=1.5)
    axes[0, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[0, 2].set_xlabel('Moneyness (z₊)')
    axes[0, 2].set_ylabel('Gamma (Γ)')
    axes[0, 2].set_title('Gamma')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Vega
    axes[1, 0].plot(moneyness_values, vegas, 'm-', linewidth=1.5)
    axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 0].set_xlabel('Moneyness (z₊)')
    axes[1, 0].set_ylabel('Vega (V)')
    axes[1, 0].set_title('Vega')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Libra (the new Greek!)
    axes[1, 1].plot(moneyness_values, libras, 'purple', linewidth=2)
    axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 1].set_xlabel('Moneyness (z₊)')
    axes[1, 1].set_ylabel('Libra (L)')
    axes[1, 1].set_title('LIBRA (Lévy Area Sensitivity) - New Greek!')
    axes[1, 1].grid(True, alpha=0.3)
    
    # All Greeks normalized
    axes[1, 2].plot(moneyness_values, np.array(thetas) / np.max(np.abs(thetas) + 1e-8), label='Θ', linewidth=1)
    axes[1, 2].plot(moneyness_values, np.array(deltas) / np.max(np.abs(deltas) + 1e-8), label='Δ', linewidth=1)
    axes[1, 2].plot(moneyness_values, np.array(gammas) / np.max(np.abs(gammas) + 1e-8), label='Γ', linewidth=1)
    axes[1, 2].plot(moneyness_values, np.array(vegas) / np.max(np.abs(vegas) + 1e-8), label='V', linewidth=1)
    axes[1, 2].plot(moneyness_values, np.array(libras) / np.max(np.abs(libras) + 1e-8), label='L', linewidth=2)
    axes[1, 2].axhline(y=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 2].axvline(x=0, color='k', linestyle='--', alpha=0.3)
    axes[1, 2].set_xlabel('Moneyness (z₊)')
    axes[1, 2].set_ylabel('Normalized Value')
    axes[1, 2].set_title('All Greeks (Normalized)')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_levy_area_analysis(
    path: jnp.ndarray,
    window_size: int = 50,
    figsize: Tuple[int, int] = (12, 8)
):
    """
    Analyze Lévy area patterns in the path.
    
    Args:
        path: Array of shape (n_steps, 3)
        window_size: Rolling window for Lévy area computation
        figsize: Figure size
    """
    n_steps = len(path)
    t = path[:, 0]
    log_S = path[:, 1]
    sigma = path[:, 2]
    
    # Compute rolling Lévy areas
    levy_areas = []
    window_times = []
    
    for i in range(window_size, n_steps):
        window_path = path[i-window_size:i, 1:]  # Price and vol only
        la = compute_levy_area(window_path)
        levy_areas.append(la[0, 1])
        window_times.append(t[i])
    
    levy_areas = np.array(levy_areas)
    window_times = np.array(window_times)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # Top left: Price path with Lévy area coloring
    ax1 = axes[0, 0]
    ax1.plot(t, log_S, 'b-', linewidth=0.5, alpha=0.5)
    scatter = ax1.scatter(window_times, log_S[window_size:], c=levy_areas, 
                          cmap='RdBu', s=2, vmin=-np.max(np.abs(levy_areas)),
                          vmax=np.max(np.abs(levy_areas)))
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Log Price')
    ax1.set_title('Log Price (colored by Lévy Area)')
    plt.colorbar(scatter, ax=ax1, label='Lévy Area')
    
    # Top right: Lévy area time series
    ax2 = axes[0, 1]
    ax2.plot(window_times, levy_areas, 'k-', linewidth=0.5)
    ax2.fill_between(window_times, levy_areas, where=levy_areas > 0, 
                     color='green', alpha=0.3, label='Price leads Vol')
    ax2.fill_between(window_times, levy_areas, where=levy_areas < 0,
                     color='red', alpha=0.3, label='Vol leads Price')
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Lévy Area')
    ax2.set_title(f'Rolling Lévy Area (window={window_size})')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Bottom left: Lévy area distribution
    ax3 = axes[1, 0]
    ax3.hist(levy_areas, bins=50, density=True, alpha=0.7, color='purple')
    ax3.axvline(x=0, color='k', linestyle='--', alpha=0.5)
    ax3.axvline(x=np.mean(levy_areas), color='r', linestyle='-', 
                label=f'Mean: {np.mean(levy_areas):.4f}')
    ax3.set_xlabel('Lévy Area')
    ax3.set_ylabel('Density')
    ax3.set_title('Lévy Area Distribution')
    ax3.legend()
    
    # Bottom right: Lévy area vs subsequent return
    returns = np.diff(log_S)
    subsequent_returns = returns[window_size:]
    
    ax4 = axes[1, 1]
    ax4.scatter(levy_areas[:-1], subsequent_returns[:-1], alpha=0.3, s=5)
    
    # Fit trend line
    z = np.polyfit(levy_areas[:-1], subsequent_returns[:-1], 1)
    p = np.poly1d(z)
    x_fit = np.linspace(np.min(levy_areas), np.max(levy_areas), 100)
    ax4.plot(x_fit, p(x_fit), 'r-', linewidth=2, label=f'Slope: {z[0]:.6f}')
    
    ax4.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
    ax4.set_xlabel('Lévy Area')
    ax4.set_ylabel('Subsequent Return')
    ax4.set_title('Lévy Area vs Next Return')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def run_visualization_demo():
    """
    Run a complete visualization demonstration.
    """
    print("=" * 70)
    print("Neural RDE Visualization Demo")
    print("=" * 70)
    
    # Generate sample path
    key = jr.PRNGKey(42)
    path, info = generate_jump_diffusion_path(
        key,
        n_steps=2000,
        lambda_jump=15.0,  # High jump intensity
        mu_jump=-0.015,
        sigma_jump=0.025
    )
    
    print(f"\nGenerated path with {info['n_jumps']} jumps")
    
    # Plot 1: Path with signatures
    print("\n1. Plotting path with signature analysis...")
    fig1 = plot_path_with_signatures(path, step_size=32, depth=2)
    fig1.savefig('path_signatures.png', dpi=150, bbox_inches='tight')
    print("   Saved: path_signatures.png")
    
    # Plot 2: Lévy area analysis
    print("\n2. Plotting Lévy area analysis...")
    fig2 = plot_levy_area_analysis(path, window_size=50)
    fig2.savefig('levy_area_analysis.png', dpi=150, bbox_inches='tight')
    print("   Saved: levy_area_analysis.png")
    
    # Train a model for remaining plots
    print("\n3. Training Neural RDE model...")
    input_dim = 3
    logsig_dim = logsignature_dimension(input_dim, depth=2)
    
    model = NeuralRDE(
        input_dim=input_dim,
        hidden_dim=32,
        logsig_dim=logsig_dim,
        step_size=32,
        depth=2,
        key=jr.PRNGKey(123)
    )
    
    trained_model, losses = train_model(
        model,
        n_epochs=30,
        batch_size=16,
        key=jr.PRNGKey(456),
        verbose=True
    )
    
    # Plot 3: Training progress
    print("\n4. Plotting training progress...")
    fig3 = plot_training_progress(losses)
    fig3.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    print("   Saved: training_progress.png")
    
    # Plot 4: Greeks
    spot = float(jnp.exp(path[-1, 1]))
    print(f"\n5. Plotting Greeks (spot={spot:.2f})...")
    fig4 = plot_greeks_comparison(
        trained_model, path, spot,
        strike_range=(0.95, 1.05),
        tau=2/252
    )
    fig4.savefig('greeks_comparison.png', dpi=150, bbox_inches='tight')
    print("   Saved: greeks_comparison.png")
    
    # Plot 5: IV surface
    print("\n6. Plotting implied volatility surface...")
    fig5 = plot_implied_vol_surface(
        trained_model, path, spot,
        strike_range=(0.95, 1.05),
        tau_range=(1/252, 5/252)
    )
    fig5.savefig('iv_surface.png', dpi=150, bbox_inches='tight')
    print("   Saved: iv_surface.png")
    
    print("\n" + "=" * 70)
    print("Visualization demo complete!")
    print("Generated files: path_signatures.png, levy_area_analysis.png,")
    print("                 training_progress.png, greeks_comparison.png,")
    print("                 iv_surface.png")
    print("=" * 70)
    
    plt.show()


if __name__ == "__main__":
    run_visualization_demo()
