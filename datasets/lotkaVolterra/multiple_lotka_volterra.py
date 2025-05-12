# Based on https://github.com/smkalami/lotka-volterra-in-python
import numpy as np


class MultiLotkaVolterra:
    def __init__(self, p=10, d=2, alpha=1.2, beta=0.2, gamma=1.1, delta=0.05, sigma=0.1):
        """
        Dynamical multi-species Lotka--Volterra system. The original two-species Lotka--Volterra is a special case
        with p = 1 , d = 1.

        @param p: number of predator/prey species. Total number of variables is 2*p.
        @param d: number of GC parents per variable.
        @param alpha: strength of interaction of a prey species with itself.
        @param beta: strength of predator -> prey interaction.
        @param gamma: strength of interaction of a predator species with itself.
        @param delta: strength of prey -> predator interaction.
        @param sigma: scale parameter for the noise.
        """

        assert p >= d and p % d == 0

        self.p = p
        self.d = d

        # Coupling strengths
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.sigma = sigma

    def simulate(self, t: int, dt=0.01, downsample_factor=10, seed=None):
        if seed is not None:
            np.random.seed(seed)
        xs_0 = np.random.uniform(10, 100, size=(self.p, ))
        ys_0 = np.random.uniform(10, 100, size=(self.p, ))

        ts = np.arange(t) * dt

        # Simulation Loop
        xs = np.zeros((t, self.p))
        ys = np.zeros((t, self.p))
        xs[0, :] = xs_0
        ys[0, :] = ys_0
        for k in range(t - 1):
            xs[k + 1, :], ys[k + 1, :] = self.next(xs[k, :], ys[k, :], dt)

        causal_struct = np.zeros((self.p * 2, self.p * 2))
        signed_causal_struct = np.zeros((self.p * 2, self.p * 2))
        for j in range(self.p):
            # Self causation
            causal_struct[j, j] = 1
            causal_struct[j + self.p, j + self.p] = 1

            signed_causal_struct[j, j] = +1
            signed_causal_struct[j + self.p, j + self.p] = -1

            # Predator-prey relationships
            causal_struct[j, int(np.floor((j + self.d) / self.d) * self.d - 1 + self.p - self.d + 1):int(np.floor((j + self.d) / self.d) * self.d + self.p)] = 1
            causal_struct[j + self.p, int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) / self.d) * self.d)] = 1

            signed_causal_struct[j, int(np.floor((j + self.d) / self.d) * self.d - 1 + self.p - self.d + 1):int(np.floor((j + self.d) / self.d) * self.d + self.p)] = -1
            signed_causal_struct[j + self.p, int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) / self.d) * self.d)] = +1

        return [np.concatenate((xs[::downsample_factor, :], ys[::downsample_factor, :]), 1)], causal_struct, signed_causal_struct

    # Dynamics
    # State transitions using the Runge-Kutta method
    def next(self, x, y, dt):
        xdot1, ydot1 = self.f(x, y)
        xdot2, ydot2 = self.f(x + xdot1 * dt / 2, y + ydot1 * dt / 2)
        xdot3, ydot3 = self.f(x + xdot2 * dt / 2, y + ydot2 * dt / 2)
        xdot4, ydot4 = self.f(x + xdot3 * dt, y + ydot3 * dt)
        # Add noise to simulations
        xnew = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt / 6 + \
               np.random.normal(scale=self.sigma, size=(self.p, ))
        ynew = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt / 6 + \
               np.random.normal(scale=self.sigma, size=(self.p, ))
        # Clip from below to prevent populations from becoming negative
        return np.maximum(xnew, 0), np.maximum(ynew, 0)

    def f(self, x, y):
        xdot = np.zeros((self.p, ))
        ydot = np.zeros((self.p, ))

        for j in range(self.p):
            y_Nxj = y[int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) /
                                                                                                  self.d) * self.d)]
            x_Nyj = x[int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):int(np.floor((j + self.d) /
                                                                                                  self.d) * self.d)]
            xdot[j] = self.alpha * x[j] - self.beta * x[j] * np.sum(y_Nxj) - self.alpha * (x[j] / 200) ** 2
            ydot[j] = self.delta * np.sum(x_Nyj) * y[j] - self.gamma * y[j]
        return xdot, ydot

    def simulate_with_internal_cyber_anomaly(self, t: int, i: int, noise_std: float, dt=0.01, downsample_factor=10, seed=None):
        """
        Simulate the multi-species Lotka–Volterra system with internal cyber anomaly.
        Injects Gaussian noise into x[i] and y[i] when computing derivatives of affected states.

        :param t: number of time steps
        :param i: index of the variable to be perturbed
        :param noise_std: standard deviation of injected noise
        :param dt: time resolution
        :param downsample_factor: downsample output by this factor
        :param seed: random seed
        :return: list with concatenated [x, y] data (T, 2p)
        """
        if seed is not None:
            np.random.seed(seed)

        xs_0 = np.random.uniform(10, 100, size=(self.p,))
        ys_0 = np.random.uniform(10, 100, size=(self.p,))
        xs = np.zeros((t, self.p))
        ys = np.zeros((t, self.p))
        xs[0, :] = xs_0
        ys[0, :] = ys_0

        for k in range(t - 1):
            x = xs[k, :]
            y = ys[k, :]

            # Perturbed f using RK4
            def noisy_f(x_in, y_in):
                xdot = np.zeros((self.p,))
                ydot = np.zeros((self.p,))

                x_noisy = x_in.copy()
                y_noisy = y_in.copy()

                for j in range(self.p):
                    # Find interaction indices
                    y_Nxj = y_noisy[int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):
                                    int(np.floor((j + self.d) / self.d) * self.d)]
                    x_Nyj = x_noisy[int(np.floor((j + self.d) / self.d) * self.d - self.d + 1 - 1):
                                    int(np.floor((j + self.d) / self.d) * self.d)]

                    # Inject noise if i affects j
                    if j == i or i in range(*self._get_neighbor_range(j)):
                        x_noisy[i] += np.random.normal(0, noise_std)
                        y_noisy[i] += np.random.normal(0, noise_std)

                    xdot[j] = self.alpha * x_noisy[j] - self.beta * x_noisy[j] * np.sum(y_Nxj) - self.alpha * (x_noisy[j] / 200) ** 2
                    ydot[j] = self.delta * np.sum(x_Nyj) * y_noisy[j] - self.gamma * y_noisy[j]

                return xdot, ydot

            # Runge-Kutta with noise-injected f
            xdot1, ydot1 = noisy_f(x, y)
            xdot2, ydot2 = noisy_f(x + xdot1 * dt / 2, y + ydot1 * dt / 2)
            xdot3, ydot3 = noisy_f(x + xdot2 * dt / 2, y + ydot2 * dt / 2)
            xdot4, ydot4 = noisy_f(x + xdot3 * dt, y + ydot3 * dt)

            x_next = x + (xdot1 + 2 * xdot2 + 2 * xdot3 + xdot4) * dt / 6 + np.random.normal(scale=self.sigma, size=(self.p,))
            y_next = y + (ydot1 + 2 * ydot2 + 2 * ydot3 + ydot4) * dt / 6 + np.random.normal(scale=self.sigma, size=(self.p,))

            xs[k + 1, :] = np.maximum(x_next, 0)
            ys[k + 1, :] = np.maximum(y_next, 0)

        return [np.concatenate((xs[::downsample_factor, :], ys[::downsample_factor, :]), axis=1)]


    def add_measurement_anomaly(self, system: np.ndarray, std: float) -> np.ndarray:
        """
        Adds Gaussian noise to simulate measurement anomaly.

        :param system: np.ndarray of shape (T, 2p), output of the simulation
        :param std: standard deviation of Gaussian noise
        :return: perturbed system (same shape)
        """
        noise = np.random.normal(0, std, size=system.shape)
        return system + noise
