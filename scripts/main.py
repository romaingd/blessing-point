from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


@dataclass
class ExperimentConfig:
    """Configuration for circle diffusion experiments."""

    n_samples: int = 10000
    ambient_dims: list[int] = None  # Will be set to [2, 5, 10, 20, 50, 100]
    circle_radius: float = 1.0
    noise_levels: list[float] = None  # Will be set to [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]
    n_epochs: int = 1000
    learning_rate: float = 1e-3
    batch_size: int = 256

    def __post_init__(self):
        if self.ambient_dims is None:
            self.ambient_dims = [2, 5, 10, 20, 50, 100]
        if self.noise_levels is None:
            self.noise_levels = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]


class CircleManifoldData:
    """Generate and manage circle data embedded in high-dimensional space."""

    def __init__(self, radius: float = 1.0):
        self.radius = radius

    def generate_circle_data(self, n_samples: int, ambient_dim: int, noise_std: float = 0.0) -> torch.Tensor:
        """
        Generate circle data embedded in ambient_dim dimensions.

        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        ambient_dim : int
            Dimensionality of ambient space
        noise_std : float
            Standard deviation of ambient noise

        Returns
        -------
        torch.Tensor
            Shape (n_samples, ambient_dim) with circle data
        """
        # Generate angles uniformly
        angles = torch.rand(n_samples) * 2 * np.pi

        # Create circle in first 2 dimensions
        x = torch.zeros(n_samples, ambient_dim)
        x[:, 0] = self.radius * torch.cos(angles)
        x[:, 1] = self.radius * torch.sin(angles)

        # Add ambient noise if specified
        if noise_std > 0:
            x += torch.randn_like(x) * noise_std

        return x

    def compute_intrinsic_dimension_ratio(self, ambient_dim: int) -> float:
        """Compute r = intrinsic_dim / ambient_dim for circle (intrinsic_dim = 1)."""
        return 1.0 / ambient_dim


class ScoreNetwork(nn.Module):
    """Neural network to learn score function for diffusion model."""

    def __init__(self, dim: int, hidden_dims: list[int] = [128, 128, 128]):
        super().__init__()
        layers = []

        # Input layer
        layers.append(nn.Linear(dim + 1, hidden_dims[0]))  # +1 for time embedding
        layers.append(nn.ReLU())

        # Hidden layers
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_dims[-1], dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict score function.

        Parameters
        ----------
        x : torch.Tensor
            Data points, shape (batch_size, dim)
        t : torch.Tensor
            Time steps, shape (batch_size,)

        Returns
        -------
        torch.Tensor
            Predicted scores, shape (batch_size, dim)
        """
        # Simple time embedding
        t_embed = t.unsqueeze(-1)

        # Concatenate x and time
        input_tensor = torch.cat([x, t_embed], dim=-1)

        return self.network(input_tensor)


class DiffusionModel:
    """Simple diffusion model for score-based generation."""

    def __init__(self, dim: int, beta_schedule: str = "linear"):
        self.dim = dim
        self.score_network = ScoreNetwork(dim)
        self.optimizer = torch.optim.Adam(self.score_network.parameters(), lr=1e-3)

        # Set up noise schedule
        self.T = 1.0
        self.n_timesteps = 100
        if beta_schedule == "linear":
            self.betas = torch.linspace(0.01, 0.2, self.n_timesteps)
        else:
            raise NotImplementedError(f"Schedule {beta_schedule} not implemented")

        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def forward_process(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply forward diffusion process.

        Parameters
        ----------
        x0 : torch.Tensor
            Clean data, shape (batch_size, dim)
        t : torch.Tensor
            Time indices, shape (batch_size,)

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Noisy data and noise
        """
        # Convert discrete time to continuous
        t_continuous = t.float() / self.n_timesteps
        alpha_bar_t = self.alpha_bars[t]

        noise = torch.randn_like(x0)
        x_noisy = torch.sqrt(alpha_bar_t.unsqueeze(-1)) * x0 + torch.sqrt(1 - alpha_bar_t.unsqueeze(-1)) * noise

        return x_noisy, noise, t_continuous

    def train_step(self, x0: torch.Tensor) -> float:
        """Single training step."""
        batch_size = x0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (batch_size,))

        # Apply forward process
        x_noisy, noise, t_continuous = self.forward_process(x0, t)

        # Predict score (negative of noise direction)
        predicted_score = self.score_network(x_noisy, t_continuous)

        # True score is -noise / sqrt(1 - alpha_bar_t)
        alpha_bar_t = self.alpha_bars[t]
        true_score = -noise / torch.sqrt(1 - alpha_bar_t.unsqueeze(-1))

        # MSE loss
        loss = nn.MSELoss()(predicted_score, true_score)

        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate samples using reverse diffusion."""
        # Start from noise
        x = torch.randn(n_samples, self.dim)

        # Reverse diffusion
        for i in reversed(range(self.n_timesteps)):
            t = torch.full((n_samples,), i / self.n_timesteps)

            with torch.no_grad():
                score = self.score_network(x, t)

            # Simple Euler step (could use more sophisticated integrator)
            dt = 1.0 / self.n_timesteps
            x = x + score * dt + np.sqrt(2 * dt) * torch.randn_like(x)

        return x


class QualityMetrics:
    """Compute various quality metrics for generated samples."""

    @staticmethod
    def manifold_recovery_error(generated_samples: torch.Tensor, radius: float = 1.0) -> float:
        """
        Measure how well samples recover the circle manifold.

        Parameters
        ----------
        generated_samples : torch.Tensor
            Generated samples, shape (n_samples, dim)
        radius : float
            Expected circle radius

        Returns
        -------
        float
            Mean squared error from expected radius in first 2 dimensions
        """
        # Project to first 2 dimensions
        x_proj = generated_samples[:, :2]

        # Compute distances from origin
        distances = torch.norm(x_proj, dim=1)

        # MSE from expected radius
        mse = torch.mean((distances - radius) ** 2)

        return mse.item()

    @staticmethod
    def ambient_noise_ratio(generated_samples: torch.Tensor) -> float:
        """
        Measure ratio of variance in manifold dimensions vs ambient dimensions.

        Parameters
        ----------
        generated_samples : torch.Tensor
            Generated samples, shape (n_samples, dim)

        Returns
        -------
        float
            Ratio of ambient to manifold variance
        """
        if generated_samples.shape[1] <= 2:
            return 0.0

        # Variance in first 2 dimensions (manifold)
        manifold_var = torch.var(generated_samples[:, :2]).item()

        # Variance in remaining dimensions (ambient)
        if generated_samples.shape[1] > 2:
            ambient_var = torch.var(generated_samples[:, 2:]).item()
            return ambient_var / (manifold_var + 1e-8)
        else:
            return 0.0

    @staticmethod
    def wasserstein_distance(samples1: torch.Tensor, samples2: torch.Tensor) -> float:
        """
        Approximate 1D Wasserstein distance between two samples.
        Computed on first 2 principal components.
        """
        # Simple approximation: sort samples and compute L2 distance
        if samples1.shape[0] != samples2.shape[0]:
            min_size = min(samples1.shape[0], samples2.shape[0])
            samples1 = samples1[:min_size]
            samples2 = samples2[:min_size]

        # Project to 2D for comparison
        s1_2d = samples1[:, :2]
        s2_2d = samples2[:, :2]

        # Convert to radial coordinates and sort
        angles1 = torch.atan2(s1_2d[:, 1], s1_2d[:, 0])
        angles2 = torch.atan2(s2_2d[:, 1], s2_2d[:, 0])

        angles1_sorted = torch.sort(angles1)[0]
        angles2_sorted = torch.sort(angles2)[0]

        return torch.mean(torch.abs(angles1_sorted - angles2_sorted)).item()


def run_single_experiment(config: ExperimentConfig, ambient_dim: int, noise_level: float) -> dict:
    """
    Run single experiment for given ambient dimension and noise level.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration
    ambient_dim : int
        Ambient space dimension
    noise_level : float
        Noise level for data generation

    Returns
    -------
    Dict
        Results including metrics and model
    """
    # Generate data
    data_gen = CircleManifoldData(config.circle_radius)
    train_data = data_gen.generate_circle_data(config.n_samples, ambient_dim, noise_level)

    # Compute r ratio
    r_ratio = data_gen.compute_intrinsic_dimension_ratio(ambient_dim)

    # Create and train model
    model = DiffusionModel(ambient_dim)

    # Training loop
    losses = []
    for epoch in range(config.n_epochs):
        # Sample batch
        indices = torch.randint(0, config.n_samples, (config.batch_size,))
        batch = train_data[indices]

        # Training step
        loss = model.train_step(batch)
        losses.append(loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Generate samples
    generated_samples = model.sample(config.n_samples)

    # Compute metrics
    metrics = QualityMetrics()

    manifold_error = metrics.manifold_recovery_error(generated_samples, config.circle_radius)
    ambient_ratio = metrics.ambient_noise_ratio(generated_samples)
    wasserstein_dist = metrics.wasserstein_distance(train_data, generated_samples)

    return {
        "r_ratio": r_ratio,
        "ambient_dim": ambient_dim,
        "noise_level": noise_level,
        "final_loss": losses[-1],
        "manifold_error": manifold_error,
        "ambient_noise_ratio": ambient_ratio,
        "wasserstein_distance": wasserstein_dist,
        "generated_samples": generated_samples,
        "training_data": train_data,
        "losses": losses,
    }


def run_phase_diagram_experiment(config: ExperimentConfig) -> dict:
    """
    Run full phase diagram experiment across r ratios and noise levels.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration

    Returns
    -------
    Dict
        Complete experimental results
    """
    results = []

    print("Starting phase diagram experiment...")

    for ambient_dim in config.ambient_dims:
        for noise_level in config.noise_levels:
            print(f"\nRunning: ambient_dim={ambient_dim}, noise_level={noise_level}")

            result = run_single_experiment(config, ambient_dim, noise_level)
            results.append(result)

    return {"results": results, "config": config}


def plot_phase_diagram(experiment_results: dict, metric_name: str = "manifold_error"):
    """
    Plot phase diagram showing metric as function of r and noise level.

    Parameters
    ----------
    experiment_results : Dict
        Results from run_phase_diagram_experiment
    metric_name : str
        Which metric to plot ('manifold_error', 'ambient_noise_ratio', 'wasserstein_distance')
    """
    results = experiment_results["results"]
    config = experiment_results["config"]

    # Extract data for plotting
    r_ratios = []
    noise_levels = []
    metric_values = []

    for result in results:
        r_ratios.append(result["r_ratio"])
        noise_levels.append(result["noise_level"])
        metric_values.append(result[metric_name])

    # Create meshgrid for plotting
    unique_r = sorted(list(set(r_ratios)))
    unique_noise = sorted(list(set(noise_levels)))

    R, N = np.meshgrid(unique_r, unique_noise)
    Z = np.zeros_like(R)

    # Fill in the values
    for i, noise in enumerate(unique_noise):
        for j, r in enumerate(unique_r):
            # Find corresponding result
            for result in results:
                if abs(result["r_ratio"] - r) < 1e-6 and abs(result["noise_level"] - noise) < 1e-6:
                    Z[i, j] = result[metric_name]
                    break

    # Create plot
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(R, N, Z, levels=20, cmap="viridis")
    plt.colorbar(contour, label=metric_name)

    plt.xlabel("r = intrinsic_dim / ambient_dim")
    plt.ylabel("Noise Level")
    plt.title(f"Phase Diagram: {metric_name}")

    # Add contour lines
    plt.contour(R, N, Z, levels=10, colors="white", alpha=0.5, linewidths=0.5)

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Set up experiment
    config = ExperimentConfig(
        n_samples=5000,
        ambient_dims=[2**k for k in range(8, 14)],  # Smaller for initial testing
        noise_levels=[0.01, 0.05, 0.1, 0.2],
        n_epochs=500,  # Reduced for testing
        batch_size=128,
    )

    # Run experiments (comment out for now to avoid long execution)
    results = run_phase_diagram_experiment(config)
    plot_phase_diagram(results, 'manifold_error')

    print("Code ready for experimentation!")
    print("Uncomment the last lines to run the full experiment.")
