import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


class GPLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GPLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Wider network with non-linearity
        self.hidden_dim = 32
        self.mean_network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim)
        )

        self.log_var_network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, output_dim),
            nn.Softplus()
        )

    def forward(self, x):
        mean = self.mean_network(x)
        log_var = self.log_var_network(x)

        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mean + eps * std, mean, log_var
        return mean, mean, log_var

    def kl_divergence(self):
        kl = 0
        for param in self.mean_network.parameters():
            kl += 0.5 * torch.sum(param.pow(2))
        return kl * 0.01


class DeepGP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super(DeepGP, self).__init__()
        self.layers = nn.ModuleList()

        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(GPLayer(dims[i], dims[i + 1]))

    def forward(self, x):
        kl_div = 0
        current_output = x

        for layer in self.layers:
            current_output, mean, log_var = layer(current_output)
            kl_div += layer.kl_divergence()

        return current_output, kl_div

    def elbo_loss(self, x, y):
        predictions, kl_div = self(x)
        reconstruction_loss = F.mse_loss(predictions, y, reduction='sum')
        beta = 0.01
        loss = reconstruction_loss + beta * kl_div
        return loss / x.size(0)


def train_and_visualize():
    # Generate synthetic data
    np.random.seed(42)
    x = np.linspace(-5, 5, 100).reshape(-1, 1)
    y = np.sin(x) + 0.1 * np.random.randn(*x.shape)

    # Normalize data
    x_mean, x_std = x.mean(), x.std()
    y_mean, y_std = y.mean(), y.std()
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    # Convert to tensors
    X_tensor = torch.FloatTensor(x_norm)
    y_tensor = torch.FloatTensor(y_norm)

    # Split data
    train_size = int(0.8 * len(x))
    X_train = X_tensor[:train_size]
    y_train = y_tensor[:train_size]
    X_test = X_tensor[train_size:]
    y_test = y_tensor[train_size:]

    # Model parameters
    input_dim = 1
    hidden_dims = [32]
    output_dim = 1

    # Initialize model and optimizer
    model = DeepGP(input_dim, hidden_dims, output_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50, factor=0.5)

    # Training parameters
    epochs = 20000
    best_loss = float('inf')
    patience = 100
    patience_counter = 0
    train_losses = []

    # Training loop with progress bar
    pbar = tqdm(range(epochs), desc="Training")
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()

        loss = model.elbo_loss(X_train, y_train)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses.append(loss.item())
        scheduler.step(loss)

        if loss.item() < best_loss:
            best_loss = loss.item()
            patience_counter = 0
        else:
            patience_counter += 1

        # if patience_counter >= patience:
        #     print(f"\nEarly stopping at epoch {epoch}")
        #     break

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    # Visualization
    fig = plt.figure(figsize=(15, 10))

    # 1. Training Loss Plot
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(train_losses, 'b-', label='Training Loss', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # 2. Prediction Plot
    ax2 = fig.add_subplot(2, 1, 2)

    # Generate predictions
    X_full = torch.linspace(X_test.min(), X_test.max(), 200).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        n_samples = 100
        predictions_list = []
        for _ in range(n_samples):
            pred, _ = model(X_full)
            pred = pred * y_std + y_mean
            predictions_list.append(pred.numpy())

        predictions = np.array(predictions_list)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

    # Denormalize data for plotting
    X_full_orig = X_full.numpy() * x_std + x_mean
    X_train_orig = X_train.numpy() * x_std + x_mean
    y_train_orig = y_train.numpy() * y_std + y_mean
    X_test_orig = X_test.numpy() * x_std + x_mean
    y_test_orig = y_test.numpy() * y_std + y_mean

    # Plot data and predictions
    ax2.scatter(X_train_orig, y_train_orig, color='blue', alpha=0.5, label='Training Data', s=30)
    ax2.scatter(X_test_orig, y_test_orig, color='green', alpha=0.5, label='Test Data', s=30)
    ax2.plot(X_full_orig, mean_pred, 'r-', label='Mean Prediction', linewidth=2)
    ax2.fill_between(X_full_orig.flatten(),
                     mean_pred.flatten() - 2 * std_pred.flatten(),
                     mean_pred.flatten() + 2 * std_pred.flatten(),
                     color='red', alpha=0.2, label='95% Confidence')

    # Plot true function
    X_true = np.linspace(X_full_orig.min(), X_full_orig.max(), 200)
    y_true = np.sin(X_true)
    ax2.plot(X_true, y_true, 'k--', label='True Function', alpha=0.8)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Model Predictions and Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.show()

    # Model performance metrics
    with torch.no_grad():
        test_pred, _ = model(X_test)
        test_pred = test_pred.numpy() * y_std + y_mean

    mse = np.mean((y_test_orig - test_pred) ** 2)
    r2 = 1 - np.sum((y_test_orig - test_pred) ** 2) / np.sum((y_test_orig - y_test_orig.mean()) ** 2)

    print("\nModel Performance Metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")


if __name__ == "__main__":
    train_and_visualize()
