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
        self.hidden_dim = 64  # Increased hidden dimension

        # Initialize weights with smaller values
        self.mean_network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),  # Changed to Tanh for better stability
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, output_dim)
        )

        # Separate variance network
        self.log_var_network = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, output_dim),
            nn.Softplus()
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.mean_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.5)
                nn.init.zeros_(module.bias)

        for module in self.log_var_network.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight, gain=0.1)
                nn.init.constant_(module.bias, -2)

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
        return kl * 0.001  # Reduced KL weight

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
        beta = 0.001  # Reduced beta for better reconstruction
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
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=100, factor=0.5)

    # Training parameters
    epochs = 2000
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

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})


    # Visualization
    fig = plt.figure(figsize=(15, 15))

    # 1. Training Loss Plot
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(train_losses, 'b-', label='Training Loss', alpha=0.7)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Progress')
    ax1.legend()
    ax1.grid(True, which="both", ls="-", alpha=0.2)

    # 2. Training Data Predictions
    ax2 = fig.add_subplot(3, 1, 2)

    # Generate predictions for training data
    X_train_full = torch.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
    model.eval()
    with torch.no_grad():
        n_samples = 100
        train_predictions_list = []
        for _ in range(n_samples):
            pred, _ = model(X_train_full)
            pred = pred * y_std + y_mean
            train_predictions_list.append(pred.numpy())

        train_predictions = np.array(train_predictions_list)
        train_mean_pred = np.mean(train_predictions, axis=0)
        train_std_pred = np.std(train_predictions, axis=0)

    # Denormalize training data
    X_train_full_orig = X_train_full.numpy() * x_std + x_mean
    X_train_orig = X_train.numpy() * x_std + x_mean
    y_train_orig = y_train.numpy() * y_std + y_mean

    # Plot training data and predictions with multiple uncertainty levels
    ax2.scatter(X_train_orig, y_train_orig, color='blue', alpha=0.5, label='Training Data', s=30)
    ax2.plot(X_train_full_orig, train_mean_pred, 'r-', label='Mean Prediction', linewidth=2)

    # 2 standard deviations (95% confidence)
    ax2.fill_between(X_train_full_orig.flatten(),
                     train_mean_pred.flatten() - 2 * train_std_pred.flatten(),
                     train_mean_pred.flatten() + 2 * train_std_pred.flatten(),
                     color='red', alpha=0.1, label='2σ (95% Confidence)')

    # 1 standard deviation (68% confidence)
    ax2.fill_between(X_train_full_orig.flatten(),
                     train_mean_pred.flatten() - train_std_pred.flatten(),
                     train_mean_pred.flatten() + train_std_pred.flatten(),
                     color='red', alpha=0.2, label='1σ (68% Confidence)')

    # Plot true function
    X_true = np.linspace(X_train_full_orig.min(), X_train_full_orig.max(), 200)
    y_true = np.sin(X_true)
    ax2.plot(X_true, y_true, 'k--', label='True Function', alpha=0.8)

    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_title('Training Data: Model Predictions and Uncertainty')
    ax2.legend()
    ax2.grid(True, alpha=0.2)

    # 3. Test Data Predictions
    ax3 = fig.add_subplot(3, 1, 3)

    # Generate predictions for test data
    X_test_full = torch.linspace(X_test.min(), X_test.max(), 200).reshape(-1, 1)
    with torch.no_grad():
        test_predictions_list = []
        for _ in range(n_samples):
            pred, _ = model(X_test_full)
            pred = pred * y_std + y_mean
            test_predictions_list.append(pred.numpy())

        test_predictions = np.array(test_predictions_list)
        test_mean_pred = np.mean(test_predictions, axis=0)
        test_std_pred = np.std(test_predictions, axis=0)

    # Denormalize test data
    X_test_full_orig = X_test_full.numpy() * x_std + x_mean
    X_test_orig = X_test.numpy() * x_std + x_mean
    y_test_orig = y_test.numpy() * y_std + y_mean

    # Plot test data and predictions with multiple uncertainty levels
    ax3.scatter(X_test_orig, y_test_orig, color='green', alpha=0.5, label='Test Data', s=30)
    ax3.plot(X_test_full_orig, test_mean_pred, 'r-', label='Mean Prediction', linewidth=2)

    # 2 standard deviations (95% confidence)
    ax3.fill_between(X_test_full_orig.flatten(),
                     test_mean_pred.flatten() - 2 * test_std_pred.flatten(),
                     test_mean_pred.flatten() + 2 * test_std_pred.flatten(),
                     color='red', alpha=0.1, label='2σ (95% Confidence)')

    # 1 standard deviation (68% confidence)
    ax3.fill_between(X_test_full_orig.flatten(),
                     test_mean_pred.flatten() - test_std_pred.flatten(),
                     test_mean_pred.flatten() + test_std_pred.flatten(),
                     color='red', alpha=0.2, label='1σ (68% Confidence)')

    # Plot true function
    X_true = np.linspace(X_test_full_orig.min(), X_test_full_orig.max(), 200)
    y_true = np.sin(X_true)
    ax3.plot(X_true, y_true, 'k--', label='True Function', alpha=0.8)

    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_title('Test Data: Model Predictions and Uncertainty')
    ax3.legend()
    ax3.grid(True, alpha=0.2)

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