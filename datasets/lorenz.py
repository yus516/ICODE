import numpy as np
from scipy.integrate import ode


class Lorenz96:
    """
    Lorenz96 dynamical system. The example of a system where the relationships are fixed (determined by F constant).

    Functional form:
    ----------------
    dx_i / dt = (x_{i+1} - x_{i-2}) x_{i-1} - x _i + F

    System parameters:
    ------------------
    Initial states are sampled randomly.
    """
    def __init__(self, dim=10, force=8, seed=None):
        """
        Initialisaes the Lorenz96 system.

        :param dim: number of variables.
        :param force: forcing constant value.
        """
        self.num_sim = 0
        self.t_len = 0
        self.simulations = []
        # integration tool
        self.system_ode = ode(self._system_ode).set_integrator('vode', method='bdf')
        # integration parameters
        self.dim = dim
        self.force = force
        if seed is not None:
            np.random.seed(seed)

    @staticmethod
    def _system_ode(t: float, x: np.array, theta: np.array) -> list:

        dim = int(theta[0])
        force = theta[1]
        # f-form
        f = []
        for n in range(dim):
            f.append((x[(n+1) % dim] - x[(n+dim-2) % dim])
                     * x[(n+dim-1) % dim]
                     - x[n % dim]
                     + force)
        return f
    
    @staticmethod
    def _system_ode_cyber(t: float, x_vec: np.array, theta: np.array, i: int) -> list:

        dim = int(theta[0])
        force = theta[1]
        f = []
        x = x_vec.copy()

        # Inject anomaly by perturbing x[i] temporarily
        x_i_backup = x[i]  # store clean value

        for n in range(dim):
            idxs = {(n - 2) % dim, (n - 1) % dim, n % dim, (n + 1) % dim}
            if i in idxs:
                x[i] = x_i_backup + 1

            dx_n = (x[(n + 1) % dim] - x[(n + dim - 2) % dim]) * x[(n + dim - 1) % dim] - x[n] + force
            f.append(dx_n)

            x[i] = x_i_backup  # reset for next iteration
        return f

    def create_data(self, num_sim, t_len):
        """
        Numerically simulates discrete time series from the initialised Lorenz 96 system.

        :param num_points: number of data points.
        :return: returns the simulated data.
        """
        # generate specified number of points
        self.num_sim = num_sim
        self.t_len = t_len
        for i in range(self.num_sim):
            inp = self.gen_data_point()
            self.simulations.append(np.transpose(inp))
        return self.simulations

    def get_causal_structure(self):
        a = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            a[i, i] = 1
            a[(i + 1) % self.dim, i] = 1
            a[(i + 2) % self.dim, i] = 1
            a[((i - 1) < 0) * (self.dim - 1) + ((i - 1) >= 0) * (i - 1), i] = 1
        return a

    def gen_data_point(self, downsample=True) -> list:
        """
        Integrate the system using an scipy built-in ODE solver.

        :param t_delta_integration: time between integration intervals.
        :return: a numpy array with dimensions [n_times, self.dim].
        """
        # Integration parameters
        t_start = 0.0
        t_end = 10 * (self.t_len - 1) * 0.01
        t_delta_integration = 0.01
        # Generate initial states
        x = list(np.random.rand(self.dim, 1))
        # other
        system = np.copy(x).reshape(self.dim, 1)
        t = [t_start]
        # initialize LV ODE system
        self.system_ode.set_initial_value(x, t_start).set_f_params([self.dim, self.force])
        # integrate until specified
        while self.system_ode.successful() and self.system_ode.t < t_end:
            self.system_ode.integrate(t=self.system_ode.t + t_delta_integration)
            system = np.c_[system, self.system_ode.y.reshape(self.dim, 1)]
            t.append(self.system_ode.t)
        # Downsample the time series
        if downsample:
            system = system[:, ::10]
        return system
    
    def add_measurement_anomaly(self, system: np.ndarray, m: float) -> np.ndarray:
        """
        Adds Gaussian noise to the entire trajectory to simulate measurement anomaly.

        :param system: np.ndarray, shape (dim, T)
        :param std: standard deviation of Gaussian noise
        :return: noisy system
        """
        noise = np.random.normal(m, 1, size=system.shape)
        return system + noise
    
    def simulate_with_cyber_anomaly(self, i: int, std: float, downsample=True) -> np.ndarray:
        """
        Simulate the system while injecting a cyber anomaly during integration.

        This perturbs x[i-1], x[i], x[i+1] with Gaussian noise at every ODE step.

        :param i: index of the target variable (cyber attack target)
        :param std: standard deviation of injected noise
        :param downsample: whether to downsample by 10
        :return: np.ndarray of shape (dim, T)
        """
        # Integration parameters
        t_start = 0.0
        t_end = 10 * (self.t_len - 1) * 0.01
        t_delta_integration = 0.01
        # Generate initial states
        x = list(np.random.rand(self.dim, 1))
        # other
        system = np.copy(x).reshape(self.dim, 1)
        t = [t_start]
        # initialize LV ODE system
        self.system_ode.set_initial_value(x, t_start).set_f_params([self.dim, self.force])
        # integrate until specified
        while self.system_ode.successful() and self.system_ode.t < t_end:
            self.system_ode.integrate(t=self.system_ode.t + t_delta_integration)
            system = np.c_[system, self.system_ode.y.reshape(self.dim, 1)]
            t.append(self.system_ode.t)
        # Downsample the time series
        if downsample:
            system = system[:, ::10]
        return system

