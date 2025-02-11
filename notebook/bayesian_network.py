import random
from collections import defaultdict

import numpy as np


class BayesianNetwork:
    def __init__(self):
        self.nodes = []
        self.parents = {}
        self.dependencies = defaultdict(list)
        self.probabilities = {}
        self.parameter_priors = {}  # Beta priors for each parameter

    def add_node(self, node, parents):
        self.nodes.append(node)
        self.parents[node] = parents
        for parent in parents:
            self.dependencies[parent].append(node)

        # Initialize uniform priors (Beta(1,1)) for all parameters
        num_parents = len(parents)
        num_params = 2 ** num_parents
        self.parameter_priors[node] = [(1, 1) for _ in range(num_params)]

        # Initialize random probabilities
        self.probabilities[node] = [random.random() for _ in range(num_params)]

    def get_parent_configuration_index(self, node, data_point):
        if not self.parents[node]:
            return 0

        index = 0
        for i, parent in enumerate(self.parents[node]):
            if data_point[parent]:
                index += 2 ** i
        return index

    def log_likelihood(self, data):
        log_like = 0
        for data_point in data:
            for node in self.nodes:
                idx = self.get_parent_configuration_index(node, data_point)
                prob = self.probabilities[node][idx]
                if data_point[node]:
                    log_like += np.log(prob)
                else:
                    log_like += np.log(1 - prob)
        return log_like

    def log_prior(self):
        log_prior = 0
        for node in self.nodes:
            for i, prob in enumerate(self.probabilities[node]):
                alpha, beta = self.parameter_priors[node][i]
                log_prior += (alpha - 1) * np.log(prob) + (beta - 1) * np.log(1 - prob)
        return log_prior

    def propose_parameters(self):
        new_probabilities = {}
        for node in self.nodes:
            new_probabilities[node] = []
            for prob in self.probabilities[node]:
                # Add random noise and clip to [0,1]
                new_prob = prob + np.random.normal(0, 0.1)
                new_prob = max(0.001, min(0.999, new_prob))
                new_probabilities[node].append(new_prob)
        return new_probabilities

    def learn_parameters(self, data, num_iterations=10000):
        current_log_prob = self.log_likelihood(data) + self.log_prior()

        samples = []
        log_probs = []

        accepted = 0
        for i in range(num_iterations):
            proposed_probabilities = self.propose_parameters()
            old_probabilities = self.probabilities.copy()
            self.probabilities = proposed_probabilities
            proposed_log_prob = self.log_likelihood(data) + self.log_prior()

            # Accept or reject based on Metropolis-Hastings ratio
            log_ratio = proposed_log_prob - current_log_prob
            if log_ratio > 0 or random.random() < np.exp(log_ratio):
                current_log_prob = proposed_log_prob
                accepted += 1
            else:
                self.probabilities = old_probabilities

            # Store samples after burn-in
            if i >= num_iterations // 2:
                samples.append({node: probs.copy()
                                for node, probs in self.probabilities.items()})
                log_probs.append(current_log_prob)

            # Print progress
            if (i + 1) % 1000 == 0:
                print(f"Iteration {i + 1}, Acceptance rate: {accepted / (i + 1):.3f}")

        final_probs = {}
        for node in self.nodes:
            final_probs[node] = np.mean([[sample[node][i]
                                          for sample in samples]
                                         for i in range(len(self.probabilities[node]))], axis=1)
        self.probabilities = final_probs

        return samples, log_probs


def generate_synthetic_data(n_samples=1000):
    true_rain_prob = 0.2
    true_sprinkler_probs = [0.4, 0.01]  # [P(S|¬R), P(S|R)]
    true_wetgrass_probs = [0.0, 0.8, 0.9, 0.99]  # [P(W|¬R,¬S), P(W|¬R,S), P(W|R,¬S), P(W|R,S)]

    data = []
    for _ in range(n_samples):
        rain = random.random() < true_rain_prob
        sprinkler_prob = true_sprinkler_probs[1] if rain else true_sprinkler_probs[0]
        sprinkler = random.random() < sprinkler_prob

        # Sample WetGrass given Rain and Sprinkler
        idx = (2 * rain) + sprinkler
        wetgrass_prob = true_wetgrass_probs[idx]
        wetgrass = random.random() < wetgrass_prob

        data.append({
            'Rain': rain,
            'Sprinkler': sprinkler,
            'WetGrass': wetgrass
        })

    return data


def main():
    network = BayesianNetwork()
    network.add_node('Rain', [])
    network.add_node('Sprinkler', ['Rain'])
    network.add_node('WetGrass', ['Rain', 'Sprinkler'])

    data = generate_synthetic_data(n_samples=10000)

    print("Learning parameters...")
    samples, log_probs = network.learn_parameters(data, num_iterations=10000)

    print("\nLearned Probabilities:")
    print(f"P(Rain=True) = {network.probabilities['Rain'][0]:.3f}")
    print(f"P(Sprinkler=True|Rain=False) = {network.probabilities['Sprinkler'][0]:.3f}")
    print(f"P(Sprinkler=True|Rain=True) = {network.probabilities['Sprinkler'][1]:.3f}")
    print("P(WetGrass=True|Rain,Sprinkler):")
    for i, p in enumerate(network.probabilities['WetGrass']):
        r = bool(i // 2)
        s = bool(i % 2)
        print(f"  Rain={r}, Sprinkler={s}: {p:.3f}")


if __name__ == "__main__":
    main()
