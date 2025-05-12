import numpy as np


class ReactionDiffusionSimulator:
    def __init__(self, dim=10, seed=None):
        """
        1D reaction-diffusion system with periodic boundary conditions.

        dx_i/dt = (x_{i-1} - x_i) + (x_{i+1} - x_i) + x_i(1 - x_i)

        :param dim: Number of variables
        :param seed: Random seed for reproducibility
        """
        self.dim = dim
        if seed is not None:
            np.random.seed(seed)

    def f(self, x):
        """
        Compute dx/dt according to the reaction-diffusion ODE.
        Periodic boundary conditions are used.

        :param x: np.ndarray of shape (dim,)
        :return: dx/dt of shape (dim,)
        """
        dx = np.zeros_like(x)
        for i in range(self.dim):
            left = x[(i - 1) % self.dim]
            right = x[(i + 1) % self.dim]
            dx[i] = (left - x[i]) + (right - x[i]) + x[i] * (1 - x[i])
        return dx

    def simulate(self, t_steps=1000, dt=0.01):
        """
        Simulate the system using Euler integration.

        :param t_steps: number of time steps
        :param dt: time step size
        :return: np.ndarray of shape (t_steps, dim)
        """
        x = np.random.rand(self.dim)
        trajectory = [x.copy()]
        for _ in range(t_steps - 1):
            dx = self.f(x)
            x = x + dx * dt
            trajectory.append(x.copy())
        return np.stack(trajectory)  # shape: (T, dim)

    def add_measurement_anomaly(self, data: np.ndarray, std: float) -> np.ndarray:
        """
        Add Gaussian noise to the entire trajectory (measurement anomaly).

        :param data: shape (T, dim)
        :param std: standard deviation of noise
        :return: perturbed data
        """
        noise = np.random.normal(0, std, size=data.shape)
        return data + noise

    def simulate_with_internal_cyber_anomaly(self, i: int, noise_std: float, t_steps=1000, dt=0.01):
        """
        Simulate the system while injecting noise into x[i] during ODE computation
        whenever i appears in (i-1, i, i+1) for any dx_j/dt.

        :param i: target index for cyber anomaly
        :param noise_std: standard deviation of injected noise
        :param t_steps: number of steps
        :param dt: step size
        :return: np.ndarray of shape (T, dim)
        """
        x = np.random.rand(self.dim)
        trajectory = [x.copy()]

        for _ in range(t_steps - 1):
            dx = np.zeros_like(x)
            for j in range(self.dim):
                idxs = {(j - 1) % self.dim, j, (j + 1) % self.dim}
                x_mod = x.copy()
                if i in idxs:
                    x_mod[i] += np.random.normal(0, noise_std)
                left = x_mod[(j - 1) % self.dim]
                right = x_mod[(j + 1) % self.dim]
                dx[j] = (left - x_mod[j]) + (right - x_mod[j]) + x_mod[j] * (1 - x_mod[j])
            x = x + dx * dt
            trajectory.append(x.copy())

        return np.stack(trajectory)
