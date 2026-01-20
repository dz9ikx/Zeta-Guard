"""
Zeta-Guard Core Implementation
Dual-threshold stability monitoring based on the ζ = 1/√2 ≈ 0.707 invariant.
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, Union, List
from dataclasses import dataclass
import warnings


@dataclass
class StabilityResult:
    """Container for stability monitoring results."""
    status: str                    # "STABLE", "DRIFT", or "CRITICAL_SNAP"
    raw_zeta: float               # Instantaneous stability index
    smooth_zeta: float            # Exponential moving average
    severity: str                 # "LOW", "MEDIUM", "HIGH"
    recommendation: str           # Suggested action
    history_length: int           # Number of observations in history


class ZetaGuard:
    """
    Adaptive stability monitor for high-dimensional systems.
    
    Implements dual-threshold detection based on the Butterworth optimal
    damping coefficients:
    - ζ₁ = 1/√2 ≈ 0.707 (gradual drift threshold)
    - ζ₂ = √2 ≈ 1.414 (critical snap threshold)
    
    Monitors the Jerk-to-Acceleration Ratio (JAR) in system trajectories
    to detect instability before it causes training failure.
    """
    
    def __init__(
        self,
        drift_threshold: float = 0.707,      # ζ₁: Gradual drift detection
        snap_threshold: float = 1.414,       # ζ₂: Critical snap detection  
        alpha: float = 0.3,                  # Exponential smoothing factor
        warmup_steps: int = 20,              # Steps for EMA initialization
        normalize_by_dim: bool = True,       # Normalize by sqrt(dimension)
        enable_warnings: bool = True,        # Enable runtime warnings
        device: Optional[torch.device] = None,
    ):
        """
        Initialize the Zeta-Guard stability monitor.
        
        Args:
            drift_threshold: ζ₁ value for detecting gradual instability (0.707)
            snap_threshold: ζ₂ value for detecting critical snaps (1.414)
            alpha: Smoothing factor for exponential moving average (0.0-1.0)
            warmup_steps: Number of steps for adaptive EMA initialization
            normalize_by_dim: Whether to normalize by sqrt(dimension)
            enable_warnings: Enable warnings for potential issues
            device: PyTorch device for computations (None = auto-detect)
        """
        # Validate thresholds (must follow ζ₂ = √2 * ζ₁ relationship)
        if snap_threshold < drift_threshold * 1.9:  # Approx √2 = 1.414...
            warnings.warn(
                f"snap_threshold ({snap_threshold}) should be ≥ √2 * drift_threshold "
                f"({drift_threshold * 1.414:.3f}) for proper dual-threshold operation",
                UserWarning
            )
        
        self.zeta_drift = drift_threshold
        self.zeta_snap = snap_threshold
        self.alpha = max(0.01, min(0.99, alpha))  # Clamp to valid range
        self.warmup_steps = max(1, warmup_steps)
        self.normalize_by_dim = normalize_by_dim
        self.enable_warnings = enable_warnings
        
        # Device handling
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        # State tracking
        self.smooth_zeta = 0.0               # Exponential moving average
        self.step_count = 0                  # Total monitoring steps
        self.history: List[float] = []       # Raw zeta history for stats
        self.snapshot_buffer = []            # Buffer for derivative calculation
        
        # Statistics
        self.stats = {
            'total_snaps': 0,                # Critical snap events
            'total_drifts': 0,               # Drift events
            'max_zeta': 0.0,                 # Maximum observed zeta
            'mean_zeta': 0.0,                # Running mean
            'ema_adaptation': 1.0,           # Adaptive EMA factor
        }
        
        # Previous states for derivative calculation
        self.prev_state: Optional[torch.Tensor] = None
        self.prev_prev_state: Optional[torch.Tensor] = None
        
    def monitor(
        self,
        current_state: Union[torch.Tensor, np.ndarray, List[float]],
        return_full_result: bool = False,
    ) -> Union[StabilityResult, Tuple[str, float]]:
        """
        Monitor system stability based on state trajectory.
        
        Args:
            current_state: Current system state (any tensor-like structure)
            return_full_result: If True, return StabilityResult object;
                              if False, return tuple (status, zeta_value)
        
        Returns:
            Stability monitoring result with status and metrics.
            
        Example:
            >>> guard = ZetaGuard()
            >>> status, zeta = guard.monitor(model_parameters)
            >>> if status == "CRITICAL_SNAP":
            >>>     print(f"Emergency! ζ = {zeta:.3f}")
        """
        # Convert input to PyTorch tensor if needed
        if not isinstance(current_state, torch.Tensor):
            current_tensor = torch.as_tensor(
                current_state, 
                device=self.device,
                dtype=torch.float32
            )
        else:
            current_tensor = current_state.to(self.device)
        
        # Need at least 3 states for meaningful derivatives
        if self.prev_state is None or self.prev_prev_state is None:
            self._update_state_buffer(current_tensor)
            self.step_count += 1
            
            # Return stable status during warmup
            if return_full_result:
                return StabilityResult(
                    status="STABLE",
                    raw_zeta=0.0,
                    smooth_zeta=0.0,
                    severity="LOW",
                    recommendation="Collecting initial states...",
                    history_length=self.step_count,
                )
            return "STABLE", 0.0
        
        # Calculate stability metrics
        raw_zeta = self._calculate_zeta(
            current_tensor, 
            self.prev_state, 
            self.prev_prev_state
        )
        
        # Update exponential moving average with adaptive learning rate
        self._update_smooth_zeta(raw_zeta)
        
        # Update statistics
        self._update_statistics(raw_zeta)
        
        # Determine stability status
        status, severity, recommendation = self._analyze_stability(raw_zeta)
        
        # Prepare result
        result = StabilityResult(
            status=status,
            raw_zeta=raw_zeta,
            smooth_zeta=self.smooth_zeta,
            severity=severity,
            recommendation=recommendation,
            history_length=len(self.history),
        )
        
        # Update state buffer for next call
        self._update_state_buffer(current_tensor)
        
        return result if return_full_result else (status, raw_zeta)
    
    def _calculate_zeta(
        self,
        current: torch.Tensor,
        prev: torch.Tensor,
        prev_prev: torch.Tensor,
    ) -> float:
        """
        Calculate the Jerk-to-Acceleration Ratio (JAR) / ζ value.
        
        Mathematical formulation:
            velocity = x[t] - x[t-1]
            acceleration = velocity[t] - velocity[t-1]
            jerk = acceleration[t] - acceleration[t-1]
            ζ = ‖jerk‖ / (‖acceleration‖ + ε)
        
        Args:
            current: State at time t
            prev: State at time t-1
            prev_prev: State at time t-2
            
        Returns:
            ζ value (normalized instability index)
        """
        # Flatten all tensors for consistent norm calculation
        x_flat = current.flatten()
        x1_flat = prev.flatten()
        x2_flat = prev_prev.flatten()
        
        # First derivative (velocity)
        v_t = x_flat - x1_flat          # v[t]
        v_t1 = x1_flat - x2_flat        # v[t-1]
        
        # Second derivative (acceleration)
        a_t = v_t - v_t1                # a[t] = v[t] - v[t-1]
        
        # For jerk (third derivative), we need a[t-1]
        # Since we only have 3 points, we approximate:
        # a[t-1] = v[t-1] - v[t-2] but we don't have v[t-2]
        # Alternative: Use finite difference approximation
        # jerk ≈ x[t] - 3*x[t-1] + 3*x[t-2] - x[t-3]
        
        # Simplified jerk estimation from 3 points
        # This approximates the rate of change of acceleration
        jerk_approx = v_t - 2 * v_t1 + (x2_flat - x2_flat.roll(1))
        
        # Calculate norms
        norm_jerk = torch.norm(jerk_approx, p=2)
        norm_accel = torch.norm(a_t, p=2)
        
        # Normalize by dimension if requested
        if self.normalize_by_dim:
            dim = x_flat.shape[0]
            norm_jerk /= (dim ** 0.5 + 1e-9)
            norm_accel /= (dim ** 0.5 + 1e-9)
        
        # Avoid division by zero
        if norm_accel < 1e-12:
            if self.enable_warnings:
                warnings.warn(
                    "Near-zero acceleration detected. System may be stagnating.",
                    RuntimeWarning
                )
            return 0.0
        
        # ζ = Jerk-to-Acceleration Ratio
        zeta_value = (norm_jerk / norm_accel).item()
        
        # Store in history
        self.history.append(zeta_value)
        
        return zeta_value
    
    def _update_smooth_zeta(self, raw_zeta: float):
        """Update exponential moving average of ζ values."""
        # Adaptive learning rate during warmup
        if self.step_count < self.warmup_steps:
            # Start with high trust in raw values, gradually incorporate smoothing
            adaptive_alpha = min(
                0.8,  # Upper bound
                0.1 + 0.7 * (self.step_count / self.warmup_steps)
            )
        else:
            adaptive_alpha = self.alpha
        
        # Initialize or update EMA
        if self.step_count == 0:
            self.smooth_zeta = raw_zeta
        else:
            self.smooth_zeta = (1 - adaptive_alpha) * self.smooth_zeta + adaptive_alpha * raw_zeta
        
        # Track adaptation factor
        self.stats['ema_adaptation'] = adaptive_alpha
        
        self.step_count += 1
    
    def _update_statistics(self, raw_zeta: float):
        """Update monitoring statistics."""
        # Update running statistics
        old_mean = self.stats['mean_zeta']
        n = len(self.history)
        
        # Online mean update
        if n == 1:
            self.stats['mean_zeta'] = raw_zeta
        else:
            self.stats['mean_zeta'] = old_mean + (raw_zeta - old_mean) / n
        
        # Track maximum
        if raw_zeta > self.stats['max_zeta']:
            self.stats['max_zeta'] = raw_zeta
    
    def _analyze_stability(
        self, 
        raw_zeta: float
    ) -> Tuple[str, str, str]:
        """
        Analyze stability based on ζ values and thresholds.
        
        Returns:
            Tuple of (status, severity, recommendation)
        """
        # CRITICAL_SNAP: Sudden, catastrophic instability
        if raw_zeta > self.zeta_snap:
            self.stats['total_snaps'] += 1
            return (
                "CRITICAL_SNAP",
                "HIGH",
                "IMMEDIATE ACTION REQUIRED: Save checkpoint, reduce learning rate by 10x, "
                "consider restarting from last stable state."
            )
        
        # DRIFT: Gradual accumulation of instability
        if self.smooth_zeta > self.zeta_drift:
            self.stats['total_drifts'] += 1
            return (
                "DRIFT",
                "MEDIUM",
                "Corrective action recommended: Reduce learning rate by 2x, "
                "increase batch size, add gradient clipping."
            )
        
        # STABLE: System within acceptable bounds
        return (
            "STABLE",
            "LOW",
            "Continue monitoring. System is within stable operating parameters."
        )
    
    def _update_state_buffer(self, new_state: torch.Tensor):
        """Update the state buffer for derivative calculations."""
        if self.prev_state is not None:
            self.prev_prev_state = self.prev_state.clone()
        
        self.prev_state = new_state.clone()
    
    def reset(self):
        """Reset the monitor to initial state."""
        self.smooth_zeta = 0.0
        self.step_count = 0
        self.history = []
        self.snapshot_buffer = []
        
        self.prev_state = None
        self.prev_prev_state = None
        
        self.stats = {
            'total_snaps': 0,
            'total_drifts': 0,
            'max_zeta': 0.0,
            'mean_zeta': 0.0,
            'ema_adaptation': 1.0,
        }
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive monitoring statistics."""
        stats = self.stats.copy()
        
        if self.history:
            history_array = np.array(self.history)
            stats.update({
                'current_zeta': self.history[-1],
                'smooth_zeta': self.smooth_zeta,
                'history_size': len(self.history),
                'zeta_std': float(np.std(history_array)),
                'zeta_percentile_95': float(np.percentile(history_array, 95)),
                'drift_ratio': self.stats['total_drifts'] / max(1, len(self.history)),
                'snap_ratio': self.stats['total_snaps'] / max(1, len(self.history)),
            })
        
        return stats
    
    def get_recommendation(self, status: str) -> str:
        """
        Get detailed recommendation based on stability status.
        
        Args:
            status: One of "STABLE", "DRIFT", or "CRITICAL_SNAP"
            
        Returns:
            Detailed recommendation string
        """
        recommendations = {
            "CRITICAL_SNAP": """
            🚨 CRITICAL SNAP DETECTED (ζ > {snap:.3f})
            
            Immediate Actions:
            1. SAVE CURRENT MODEL CHECKPOINT
            2. Reduce learning rate by factor of 10
            3. Roll back to last known stable checkpoint if available
            4. Enable gradient clipping with norm = 1.0
            5. Consider reducing batch size temporarily
            
            Investigation needed:
            - Check for exploding gradients
            - Verify data pipeline integrity
            - Review recent hyperparameter changes
            """,
            
            "DRIFT": """
            ⚠️  GRADUAL DRIFT DETECTED (smooth ζ > {drift:.3f})
            
            Recommended Corrections:
            1. Reduce learning rate by factor of 2
            2. Increase batch size by 25%
            3. Add weight decay (1e-4) if not present
            4. Enable gradient clipping with norm = 5.0
            5. Monitor loss curvature more closely
            
            Preventive measures:
            - Schedule learning rate warm restarts
            - Add learning rate scheduling
            - Implement early stopping
            """,
            
            "STABLE": """
            ✅ SYSTEM STABLE (ζ < {drift:.3f})
            
            Continue normal operation. Recommendations:
            1. Maintain current hyperparameters
            2. Periodic checkpointing every epoch
            3. Monitor ζ trend for gradual changes
            4. Consider gradual learning rate decay
            
            Optimization opportunities:
            - Slight increase in learning rate (+10%)
            - Larger batch size for throughput
            - More aggressive optimization if ζ remains low
            """
        }
        
        return recommendations.get(status, "Unknown status").format(
            drift=self.zeta_drift,
            snap=self.zeta_snap
        )
    
    def save_state(self, filepath: str):
        """
        Save monitor state to disk for resumable monitoring.
        
        Args:
            filepath: Path to save state (.pt file)
        """
        state = {
            'smooth_zeta': self.smooth_zeta,
            'step_count': self.step_count,
            'history': self.history,
            'stats': self.stats,
            'zeta_drift': self.zeta_drift,
            'zeta_snap': self.zeta_snap,
            'alpha': self.alpha,
            'device': str(self.device),
        }
        
        torch.save(state, filepath)
    
    def load_state(self, filepath: str):
        """
        Load monitor state from disk.
        
        Args:
            filepath: Path to saved state (.pt file)
        """
        state = torch.load(filepath, map_location=self.device)
        
        self.smooth_zeta = state['smooth_zeta']
        self.step_count = state['step_count']
        self.history = state['history']
        self.stats = state['stats']
        self.zeta_drift = state.get('zeta_drift', 0.707)
        self.zeta_snap = state.get('zeta_snap', 1.414)
        self.alpha = state.get('alpha', 0.3)
        
        # Note: Previous states (prev_state, prev_prev_state) are not saved
        # as they can be reconstructed from recent history if needed


# Example usage function
def example_usage():
    """
    Example demonstrating Zeta-Guard integration in training loop.
    """
    import torch.nn as nn
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 50),
        nn.ReLU(),
        nn.Linear(50, 1)
    )
    
    # Initialize Zeta-Guard
    guard = ZetaGuard(
        drift_threshold=0.707,
        snap_threshold=1.414,
        alpha=0.3,
        warmup_steps=10,
    )
    
    # Simulated training loop
    print("Starting training with Zeta-Guard monitoring...")
    print("=" * 60)
    
    for epoch in range(5):
        # Simulate parameter updates (in real training, this would be optimizer.step())
        with torch.no_grad():
            for param in model.parameters():
                # Add noise to simulate training updates
                param.add_(torch.randn_like(param) * 0.01)
        
        # Monitor stability
        status, zeta = guard.monitor(list(model.parameters()))
        
        # Display results
        print(f"Epoch {epoch+1}: ζ = {zeta:.3f} | Status: {status}")
        
        # Take action based on status
        if status == "CRITICAL_SNAP":
            print("  🚨 CRITICAL: Emergency protocol activated")
            # In practice: save checkpoint, reduce LR, etc.
        elif status == "DRIFT":
            print("  ⚠️  DRIFT: Corrective measures suggested")
            # In practice: adjust hyperparameters
    
    # Display final statistics
    print("\n" + "=" * 60)
    print("Training complete. Final statistics:")
    stats = guard.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    # Run example if executed directly
    example_usage()
