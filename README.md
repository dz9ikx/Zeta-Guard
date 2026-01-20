# Zeta-Guard 🛡️

Zeta-Guard is an adaptive stability supervisor that monitors neural network training in real-time using the Butterworth optimal damping principle. It detects dangerous gradient patterns before they cause training collapse, with dual-threshold protection against both sudden snaps and gradual drift.

🔥 Why Zeta-Guard?
Training modern neural networks is unstable. Models diverge, gradients explode, and weeks of training can be lost in seconds. Zeta-Guard solves this by:

⚡ Detecting critical snaps (ζ > √2 ≈ 1.414) - Instant reaction to sudden divergence

🌀 Catching gradual drift (ζ > 1/√2 ≈ 0.707) - Early warning for creeping instability

🛡️ Auto-recovery protocols - Built-in emergency procedures to save your training

📊 Real-time visualization - See your training stability as it happens


# 🚀 Quick Start
Installation
```bash
pip install zeta-guard
```

Basic Usage
Add protection to your training loop in 3 lines:
```python
import torch
from zetaguard import ZetaGuard

# Initialize the guardian
guard = ZetaGuard()

# Your training loop
for epoch in range(epochs):
    for batch in dataloader:
        # Forward/backward pass
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        # Monitor stability
        result = guard.monitor(model.parameters())
        
        # Automatic protection
        if result.status == "CRITICAL":
            print("🚨 Critical instability detected! Auto-recovery engaged.")
            # guard automatically: saves checkpoint, reduces LR, resets if needed
```

# 📊 How It Works
Zeta-Guard implements the Butterworth optimal damping principle using the ζ (zeta) coefficient:

The Mathematics
```
ζ₁ = 1/√2 ≈ 0.707  # Optimal damping ratio (Butterworth filter)
ζ₂ = √2 ≈ 1.414    # Critical instability threshold

JAR (Jerk-to-Acceleration Ratio) = ‖jerk‖ / ‖acceleration‖
```

Dual-Threshold Detection
```python
# Inside Zeta-Guard:
velocity = x[t] - x[t-1]
acceleration = velocity[t] - velocity[t-1]
jerk = acceleration[t] - acceleration[t-1]

jar = ‖jerk‖ / (‖acceleration‖ + ε)

if jar > 1.414:    # ζ₂ - Critical Snap
    emergency_protocol()
elif jar > 0.707:  # ζ₁ - Gradual Drift  
    corrective_measures()
```

# 🎯 Features
1. Intelligent Monitoring
* Exponential smoothing separates signal from noise

* Adaptive thresholds learn from your training history

* Multi-dimensional analysis handles high-dimensional parameter spaces

2. Auto-Recovery Protocols

```python
# Built-in emergency procedures
guard = ZetaGuard(auto_recover=True)

# When CRITICAL snap detected:
# 1. 📁 Saves model checkpoint
# 2. 📉 Reduces learning rate (LR *= 0.1)
# 3. 🔄 Resets to last stable state if needed
# 4. 📝 Logs incident for analysis
```

3. Real-Time Dashboard

```bash
# Launch the monitoring dashboard
zeta-dashboard --port 8080
```

Open http://localhost:8080 to see:

Live stability metrics

Historical trends

Alert history

System recommendations

4. Framework Integration
```python
# PyTorch Lightning
from pytorch_lightning import Callback
from zetaguard.integrations.pytorch_lightning import ZetaCallback

trainer = Trainer(callbacks=[ZetaCallback()])

# Hugging Face Transformers
from zetaguard.integrations.transformers import ZetaGuardCallback

training_args = TrainingArguments(
    callbacks=[ZetaGuardCallback()]
)
```
# 🔧 Advanced Configuration

Custom Thresholds

```python
guard = ZetaGuard(
    drift_threshold=0.707,   # ζ₁ - Drift detection
    snap_threshold=1.414,    # ζ₂ - Snap detection
    alpha=0.3,               # Smoothing factor (0.1-0.9)
    warmup_steps=50,         # Initial calibration steps
    auto_recover=True,       # Enable emergency protocols
    recovery_mode="aggressive"  # "conservative" | "balanced" | "aggressive"
)
```
Selective Monitoring

```python
# Monitor specific layers
guard.monitor_layers([
    model.attention.layers,
    model.output_projection
])

# Or monitor by pattern
guard.monitor_pattern(".*weight")  # All weight parameters
guard.monitor_pattern(".*bias")    # All bias parameters
```
Custom Recovery Protocols

```python
from zetaguard.protocols import EmergencyProtocol

class MyCustomProtocol(EmergencyProtocol):
    def execute(self, model, optimizer, guard):
        # Your custom recovery logic
        optimizer.param_groups[0]['lr'] *= 0.5
        self.save_checkpoint(model, "emergency_save.pt")
        self.notify_slack("Training instability detected!")
        
guard = ZetaGuard(recovery_protocol=MyCustomProtocol())
```
# 📈 Real-World Examples
1. Protecting GAN Training
```python
# GANs are notoriously unstable
guard = ZetaGuard(snap_threshold=1.2)  # More sensitive for GANs

for epoch in range(epochs):
    # Train discriminator
    # Train generator
    
    result = guard.monitor(generator.parameters())
    
    if result.status == "DRIFT":
        # Adjust training balance
        discriminator.requires_grad = False
        train_generator_extra_steps(2)
```
2. RL Agent Stability
```python
# RL agents often diverge during exploration
guard = ZetaGuard(drift_threshold=0.6)  # Conservative for safety

for episode in range(episodes):
    agent.collect_experience()
    agent.update_policy()
    
    result = guard.monitor(agent.policy_net.parameters())
    
    if result.status != "STABLE":
        # Reduce exploration, increase stability
        agent.entropy_coef *= 0.9
        agent.learning_rate *= 0.8
```
3. Large Language Model Training
```python
# LLM training is expensive - protect it!
guard = ZetaGuard(
    auto_recover=True,
    recovery_mode="conservative"  # Don't lose progress!
)

for step in range(total_steps):
    loss = model(batch)
    loss.backward()
    
    # Gradient clipping with Zeta-Guard intelligence
    if guard.should_clip_gradients():
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            guard.recommended_clip_value()
        )
    
    optimizer.step()
    
    # Periodic stability check
    if step % 100 == 0:
        guard.full_diagnostics(model)
```

# 📊 Dashboard & Visualization

Command Line Monitoring

```bash
# Terminal-based monitoring
zeta-monitor --model checkpoint.pt --interval 5

# Output:
# Epoch 125 | Loss: 0.45 | ζ: 0.32 ✅ STABLE
# Epoch 126 | Loss: 0.47 | ζ: 0.51 ✅ STABLE  
# Epoch 127 | Loss: 1.28 | ζ: 1.62 🚨 CRITICAL
# >>> Auto-recovery engaged: LR reduced, checkpoint saved
```
Integration with TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter
from zetaguard.integrations.tensorboard import ZetaBoard

writer = SummaryWriter()
zeta_board = ZetaBoard(writer)

# In training loop:
zeta_board.log_stability(guard.get_metrics(), global_step)
```

# 🧪 Testing & Validation

Test Your Model's Stability
```bash
# Run stability stress test
zeta-test --model your_model.pt --samples 1000

# Outputs stability report:
# ✅ Stability Score: 8.7/10
# ⚠️  Weak Layers: layer4.conv (ζ=0.68)
# 🚨 Critical Points: 2 detected
# 💡 Recommendations: Increase batch size, add gradient clipping
```

Unit Tests

```python
# Test Zeta-Guard in your CI pipeline
def test_training_stability():
    guard = ZetaGuard()
    unstable_model = create_unstable_model()
    
    # Simulate unstable training
    for _ in range(100):
        unstable_model.unstable_update()
        result = guard.monitor(unstable_model.parameters())
        
        if result.status == "CRITICAL":
            # Test passes - guardian detected instability
            assert guard.recovery_triggered == True
            return
            
    # Test fails - should have detected instability
    assert False, "Failed to detect training instability"
```
# 📚 API Reference

Core Classes
`ZetaGuard`
Main guardian class.

```python
class ZetaGuard:
    def __init__(self, 
                 drift_threshold: float = 0.707,
                 snap_threshold: float = 1.414,
                 alpha: float = 0.3,
                 auto_recover: bool = True,
                 **kwargs):
        ...
    
    def monitor(self, parameters) -> StabilityResult:
        """
        Monitor parameters for instability.
        Returns: StabilityResult(status, metrics, recommendations)
        """
    
    def emergency_protocol(self) -> RecoveryReport:
        """
        Execute emergency recovery procedures.
        """
    
    def get_metrics(self) -> Dict[str, float]:
        """
        Get current stability metrics.
        """
```
`StabilityResult`

```phyton
@dataclass
class StabilityResult:
    status: str  # "STABLE" | "DRIFT" | "CRITICAL"
    zeta_value: float
    raw_jar: float
    smoothed_jar: float
    confidence: float
    recommendations: List[str]
    timestamp: datetime
```

Key Methods

* `monitor(parameters)` - Check parameters for instability

* `monitor_layers(layers)` - Monitor specific layers

* `monitor_pattern(pattern)` - Monitor parameters matching regex

* `reset()` - Reset guardian state

* `save_state(path)` - Save guardian state to disk

* `load_state(path)` - Load guardian state

* `get_statistics()` - Get historical statistics

* `generate_report()` - Generate stability report

# 🔬 The Science Behind ζ=0.707

The ζ (zeta) coefficient represents the damping ratio in control theory:
```
ζ = 1/√2 ≈ 0.707106781186
```

This is the Butterworth optimal - the point of:

* Maximum bandwidth without oscillation
  
* Critical damping (fastest non-oscillatory response)

* Optimal energy transfer between states

In neural networks, we adapt this principle to monitor the "energy" of gradient updates. When the Jerk-to-Acceleration Ratio (JAR) exceeds ζ, the system is moving toward instability.

# 🤝 Contributing

Development Setup

```bash
git clone https://github.com/dz9ikx/zeta-guard.git
cd zeta-guard
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run with coverage
pytest --cov=zetaguard tests/
```

Contribution Areas

1. New Integrations - TensorFlow, JAX, MXNet

2. Additional Protocols - Custom recovery strategies

3. Visualization - Enhanced dashboards, new plots

4. Documentation - Tutorials, examples, API docs

Code Style
```
# Auto-format code
black zetaguard/
isort zetaguard/

# Type checking
mypy zetaguard/

# Linting
flake8 zetaguard/
```

# 🌟 Acknowledgments

* Butterworth filter theory (Stephen Butterworth)

* Control systems engineering community

* PyTorch and TensorFlow teams

* All open-source AI researchers


Remember: Training instability costs time, money, and sanity. Zeta-Guard is your safety net. 🛡️

When your gradients go wild, Zeta-Guard keeps them mild.
