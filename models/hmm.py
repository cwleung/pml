import torch
import torch.distributions as dist
import torch.nn as nn
import torch.optim as optim


class HMM(nn.Module):
    def __init__(self, num_states, num_observations):
        super(HMM, self).__init__()
        self.num_states = num_states
        self.num_observations = num_observations

        # Initialize variational parameters
        self.q_initial = nn.Parameter(torch.randn(num_states))
        self.q_transition = nn.Parameter(torch.randn(num_states, num_states))
        self.q_emission = nn.Parameter(torch.randn(num_states, num_observations))

    def forward(self, observations):
        # Convert variational parameters to probabilities
        initial_probs = torch.softmax(self.q_initial, dim=0)
        transition_probs = torch.softmax(self.q_transition, dim=1)
        emission_probs = torch.softmax(self.q_emission, dim=1)

        # Forward algorithm
        alpha = initial_probs * emission_probs[:, observations[0]]
        for t in range(1, len(observations)):
            alpha = (alpha.unsqueeze(1) * transition_probs).sum(dim=0) * emission_probs[:, observations[t]]

        return alpha.sum()

    def sample(self, num_samples):
        initial_probs = torch.softmax(self.q_initial, dim=0)
        transition_probs = torch.softmax(self.q_transition, dim=1)
        emission_probs = torch.softmax(self.q_emission, dim=1)

        states = []
        observations = []

        for _ in range(num_samples):
            state = torch.multinomial(initial_probs, 1).item()
            observation = torch.multinomial(emission_probs[state], 1).item()
            states.append(state)
            observations.append(observation)

            for _ in range(99):  # Generate sequences of length 100
                state = torch.multinomial(transition_probs[state], 1).item()
                observation = torch.multinomial(emission_probs[state], 1).item()
                states.append(state)
                observations.append(observation)

        return torch.tensor(states), torch.tensor(observations)


class ADVIHMM:
    def __init__(self, num_states, num_observations):
        self.model = HMM(num_states, num_observations)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)

    def fit(self, observations, num_iterations=10):
        observations = torch.tensor(observations)

        for _ in range(num_iterations):
            self.optimizer.zero_grad()

            # Compute ELBO
            log_likelihood = self.model(observations)
            kl_divergence = self.compute_kl_divergence()
            elbo = log_likelihood - kl_divergence

            # Maximize ELBO
            loss = -elbo
            loss.backward()
            self.optimizer.step()

    def compute_kl_divergence(self):
        # Compute KL divergence between variational distribution and prior
        kl_initial = dist.kl_divergence(
            dist.Categorical(logits=self.model.q_initial),
            dist.Categorical(logits=torch.zeros_like(self.model.q_initial))
        )

        kl_transition = dist.kl_divergence(
            dist.Categorical(logits=self.model.q_transition),
            dist.Categorical(logits=torch.zeros_like(self.model.q_transition))
        ).sum()

        kl_emission = dist.kl_divergence(
            dist.Categorical(logits=self.model.q_emission),
            dist.Categorical(logits=torch.zeros_like(self.model.q_emission))
        ).sum()

        return kl_initial + kl_transition + kl_emission

    def predict(self, observations):
        with torch.no_grad():
            return self.model(torch.tensor(observations))


# Example usage

if __name__ == "__main__":
    # Generate some sample data
    true_hmm = HMM(num_states=3, num_observations=5)
    states, observations = true_hmm.sample(5)

    # Create and fit the ADVI HMM
    advi_hmm = ADVIHMM(num_states=3, num_observations=5)
    advi_hmm.fit(observations)

    # Make predictions
    log_likelihood = advi_hmm.predict(observations)
    print(f"Log-likelihood of the observed sequence: {log_likelihood.item()}")

    # Compare true and inferred parameters
    print("True initial probabilities:", torch.softmax(true_hmm.q_initial, dim=0))
    print("Inferred initial probabilities:", torch.softmax(advi_hmm.model.q_initial, dim=0))

    print("True transition probabilities:")
    print(torch.softmax(true_hmm.q_transition, dim=1))
    print("Inferred transition probabilities:")
    print(torch.softmax(advi_hmm.model.q_transition, dim=1))

    print("True emission probabilities:")
    print(torch.softmax(true_hmm.q_emission, dim=1))
    print("Inferred emission probabilities:")
    print(torch.softmax(advi_hmm.model.q_emission, dim=1))
