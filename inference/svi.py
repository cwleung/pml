# Define the SVI class
import torch
from torch import optim


class SVI:
    def __init__(self, model, lr=0.01):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr)

    def elbo(self, x):
        pdf = self.model(x)
        return torch.sum(torch.log(pdf))

    def elbo_beta(self, x, beta=1.0):
        pdf = self.model(x)
        return torch.sum(torch.log(pdf)) - beta * torch.sum(torch.log(1 - pdf))

    def elbo_beta_tc(self, x, beta=1.0, gamma=1.0):
        pdf = self.model(x)
        return torch.sum(torch.log(pdf)) - beta * torch.sum(torch.log(1 - pdf)) + gamma * torch.sum(
            torch.log(pdf / (1 - pdf)))

    def step(self, x):
        self.optimizer.zero_grad()
        loss = -self.elbo(x)
        loss.backward()
        self.optimizer.step()
