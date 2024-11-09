import numpy as np


def generate_data(test_size=0.2):
    # Parameters for price distributions
    n_samples_accept = 100
    n_samples_reject = 100

    # Calculate test samples
    n_test_accept = int(n_samples_accept * test_size)
    n_test_reject = int(n_samples_reject * test_size)

    # Parameters for accepted prices
    mean_accept = 200
    sigma_accept = 30

    # Parameters for rejected prices
    mean_reject = 150
    sigma_reject = 50

    # Generate time points (randomly distributed)
    time_accept = np.sort(np.random.uniform(0, 1, n_samples_accept + n_test_accept))
    time_reject = np.sort(np.random.uniform(0, 1, n_samples_reject + n_test_reject))

    # Add time-dependent trends
    def add_time_trend(prices, time, trend_strength=0.1):
        return prices + trend_strength * time * np.mean(prices)

    # Add seasonal patterns
    def add_seasonality(prices, time, seasonal_strength=0.05):
        return prices + seasonal_strength * np.mean(prices) * np.sin(2 * np.pi * time)

    # Generate prices with trends and seasonality
    prices_accept = add_seasonality(
        add_time_trend(
            np.random.normal(mean_accept, sigma_accept, n_samples_accept + n_test_accept),
            time_accept
        ),
        time_accept
    )

    prices_reject = add_seasonality(
        add_time_trend(
            np.random.normal(mean_reject, sigma_reject, n_samples_reject + n_test_reject),
            time_reject
        ),
        time_reject
    )

    # Combine the training data
    X = np.zeros((n_samples_accept + n_samples_reject, 2))
    y = np.zeros(n_samples_accept + n_samples_reject)

    # Training data: Accepted prices (class 1)
    X[:n_samples_accept, 0] = time_accept[:n_samples_accept]
    X[:n_samples_accept, 1] = prices_accept[:n_samples_accept]
    y[:n_samples_accept] = 1

    # Training data: Rejected prices (class 0)
    X[n_samples_accept:, 0] = time_reject[:n_samples_reject]
    X[n_samples_accept:, 1] = prices_reject[:n_samples_reject]
    y[n_samples_accept:] = 0

    # Combine the test data
    X_test = np.zeros((n_test_accept + n_test_reject, 2))
    y_test = np.zeros(n_test_accept + n_test_reject)

    # Test data: Accepted prices (class 1)
    X_test[:n_test_accept, 0] = time_accept[n_samples_accept:]
    X_test[:n_test_accept, 1] = prices_accept[n_samples_accept:]
    y_test[:n_test_accept] = 1

    # Test data: Rejected prices (class 0)
    X_test[n_test_accept:, 0] = time_reject[n_samples_reject:]
    X_test[n_test_accept:, 1] = prices_reject[n_samples_reject:]
    y_test[n_test_accept:] = 0

    return X, y, X_test, y_test


if __name__ == '__main__':
    # plot
    import matplotlib.pyplot as plt

    X, y, _, _ = generate_data()
    plt.scatter(X[y == 1, 0], X[y == 1, 1], c='blue', label='Accepted', alpha=0.6)
    plt.scatter(X[y == 0, 0], X[y == 0, 1], c='red', label='Rejected', alpha=0.6)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Original Price-Time Data')
    plt.legend()
    plt.grid(True)
    plt.show()
