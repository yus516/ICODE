import torch.nn as nn
import torch

from torchdiffeq import odeint


class SENNGC(nn.Module):
    def __init__(self, num_vars: int, order: int, hidden_layer_size: int, num_hidden_layers: int, device: torch.device,
                 method="OLS"):
        """
        Generalised VAR (GVAR) model based on self-explaining neural networks.

        @param num_vars: number of variables (p).
        @param order:  model order (maximum lag, K).
        @param hidden_layer_size: number of units in the hidden layer.
        @param num_hidden_layers: number of hidden layers.
        @param device: Torch device.
        @param method: fitting algorithm (currently, only "OLS" is supported).
        """
        super(SENNGC, self).__init__()

        # Networks for amortising generalised coefficient matrices.
        self.coeff_nets = nn.ModuleList()

        # Instantiate coefficient networks
        for k in range(order):
            modules = [nn.Sequential(nn.Linear(num_vars, hidden_layer_size), nn.ReLU())]
            if num_hidden_layers > 1:
                for j in range(num_hidden_layers - 1):
                    modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size), nn.ReLU()))
            modules.extend(nn.Sequential(nn.Linear(hidden_layer_size, num_vars**2)))
            self.coeff_nets.append(nn.Sequential(*modules))

        # Some bookkeeping
        self.num_vars = num_vars
        self.order = order
        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layer_size = num_hidden_layers

        self.device = device

        self.method = method

    # Initialisation
    def init_weights(self):
        for m in self.modules():
            nn.init.xavier_normal_(m.weight.data)
            m.bias.data.fill_(0.1)

    # Forward propagation,
    # returns predictions and generalised coefficients corresponding to each prediction
    def forward(self, inputs: torch.Tensor):
        if inputs[0, :, :].shape != torch.Size([self.order, self.num_vars]):
            print("WARNING: inputs should be of shape BS x K x p")

        coeffs = None
        if self.method is "OLS":
            preds = torch.zeros((inputs.shape[0], self.num_vars)).to(self.device)
            for k in range(self.order):
                coeff_net_k = self.coeff_nets[k]
                coeffs_k = coeff_net_k(inputs[:, k, :])
                coeffs_k = torch.reshape(coeffs_k, (inputs.shape[0], self.num_vars, self.num_vars))
                if coeffs is None:
                    coeffs = torch.unsqueeze(coeffs_k, 1)
                else:
                    coeffs = torch.cat((coeffs, torch.unsqueeze(coeffs_k, 1)), 1)
                coeffs[:, k, :, :] = coeffs_k
                if self.method is "OLS":
                    preds += torch.matmul(coeffs_k, inputs[:, k, :].unsqueeze(dim=2)).squeeze()
                    preds += inputs[:, -1, :]
        elif self.method is "BFT":
            NotImplementedError("Backfitting not implemented yet!")
        else:
            NotImplementedError("Unsupported fitting method!")

        return preds, coeffs


class ODEFunc(nn.Module):
    def __init__(self, num_vars, hidden_size, num_hidden_layers):
        super().__init__()
        layers = [nn.Linear(num_vars, hidden_size), nn.Tanh()]
        for _ in range(num_hidden_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.Tanh()]
        layers += [nn.Linear(hidden_size, num_vars)]
        self.net = nn.Sequential(*layers)

    def forward(self, t, h):
        return self.net(h)

# this is for the version using neural ode. It takes a long time. If the resolution is small, use the method above. 
class SENNGODE(nn.Module):
    def __init__(self, num_vars, order, hidden_layer_size, num_hidden_layers, device):
        super().__init__()
        self.num_vars = num_vars
        self.order = order
        self.device = device

        # Dynamics module
        self.ode_func = ODEFunc(num_vars, hidden_layer_size, num_hidden_layers)

    def forward(self, inputs: torch.Tensor):
        # inputs: (BS, K, p)
        bs, K, p = inputs.shape
        assert K == self.order and p == self.num_vars

        # Use average of past states as initial condition (or pick inputs[:, 0, :])
        h0 = torch.mean(inputs, dim=1)  # (BS, p)

        # Time steps to integrate over
        t = torch.tensor([0, 1], dtype=torch.float32).to(self.device)

        # Solve ODE: output shape (2, BS, p), take final
        out = odeint(self.ode_func, h0, t, method='dopri5')[-1]

        # Optional residual: add last observed input
        return out, None  
