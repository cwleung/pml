import torch
import torch.distributions as dist
import numpy as np

import matplotlib.pyplot as plt


class PoissonMixtureADVI(torch.nn.Module):

    def __init__(self, n_components, data_dim):
        super().__init__()
        self.n_components = n_components
        self.data_dim = data_dim

        # Parameters should ensure Poisson rates are positive and adequate
        self.alpha = torch.nn.Parameter(
            torch.ones(n_components) * 10)  # Larger concentration for a more uniform initial mixture
        self.lambdas = torch.nn.Parameter(torch.rand(n_components, data_dim) * 10 + 1)  # Ensure larger initial rates

    def model_log_prob(self, data):
        mix_weights = dist.Dirichlet(self.alpha).rsample()  # Sample mixture weights

        # Expanded rates for each component for each data point
        rates = dist.Gamma(self.lambdas, torch.ones_like(self.lambdas)).rsample()

        # Broadcasting rates across all data points
        log_probs = dist.Poisson(rates[:, None, :]).log_prob(data).sum(-1)

        # Weigh using mixture weights
        log_mix_weights = torch.log(mix_weights)[:, None]
        weighted_log_probs = log_probs + log_mix_weights

        # This reduces across components using log-sum-exp for numerical stability
        return torch.logsumexp(weighted_log_probs, dim=0).sum()


def train(model, data, learning_rate=0.01, epochs=2000):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = -model.model_log_prob(data)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")


def plot_model(model, data):
    # Create a range of values to calculate the Poisson probability distributions
    x_values = torch.arange(0, torch.max(data) + 10, dtype=torch.float)

    # Calculate probabilities using learnt rates
    rates = torch.exp(model.lambdas)  # Ensure rates are not logged or altered in your model
    probabilities = [dist.Poisson(rate).log_prob(x_values[:, None]).exp().detach().numpy() for rate in rates]

    # Plotting the actual data histogram and Poisson fits
    plt.figure(figsize=(10, 5))
    plt.hist(data.numpy(), bins=30, alpha=0.6, label='Data histogram')  # Histogram of data
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown']  # Colors for different components

    for idx, prob in enumerate(probabilities):
        plt.plot(x_values, prob * len(data) * (1 / len(probabilities)),  # Scale probabilities
                 label=f'Component {idx + 1}', color=colors[idx % len(colors)])

    plt.title('Data and Fitted Poisson Distributions')
    plt.xlabel('Data Values')
    plt.ylabel('Frequency / Probability')
    plt.legend()
    plt.show()


# Example setup
data_np = np.random.poisson(lam=(1, 20), size=(1000, 2))
data = torch.tensor(data_np, dtype=torch.float)
model = PoissonMixtureADVI(n_components=2, data_dim=data.shape[1])
train(model, data)

plot_model(model, data)
