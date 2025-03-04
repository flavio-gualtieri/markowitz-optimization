import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# -------------------------------
# Neural Network for GARCH Parameter Prediction
# -------------------------------
class GARCHParameterNN(nn.Module):
    def __init__(self, input_dim):
        super(GARCHParameterNN, self).__init__()
        # A four-layer perceptron with hidden sizes (128, 2048, 2048, 128)
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Output is the predicted α₁
        )
    
    def forward(self, x):
        return self.model(x)

# -------------------------------
# Functions to compute β₁ and α₀ analytically
# -------------------------------
def compute_beta(alpha1, gamma4_emp):
    """
    Compute β₁ from the predicted α₁ and the empirical fourth standardized moment.
    The formula (from equation (11) in the paper) is:
    
        β₁ = sqrt((1 - 2α₁²/(1 - 6α₁²)) * (Γ₄,emp - 3)) - α₁

    To ensure numerical stability, we add a small epsilon to the denominator.
    """
    eps = 1e-8
    denominator = (1 - 6 * (alpha1 ** 2) + eps)
    term = (1 - 2 * (alpha1 ** 2) / denominator) * (gamma4_emp - 3)
    term = np.maximum(term, 0)  # ensure nonnegative before sqrt
    beta = np.sqrt(term) - alpha1
    return beta

def compute_alpha0(alpha1, beta1, sigma2_emp):
    """
    Compute α₀ using equation (12):
    
        α₀ = σ²_emp * (1 - α₁ - β₁)
    
    where σ²_emp is the empirical second moment (variance).
    """
    return sigma2_emp * (1 - alpha1 - beta1)

# -------------------------------
# Training Function for the NN
# -------------------------------
def train_model(train_features, train_targets, input_dim, num_epochs=5000, learning_rate=0.01, patience=100, save_path=None):
    """
    Train the GARCHParameterNN model and optionally save its state using a unique filename.
    
    Parameters:
        train_features (np.ndarray): Input features for training.
        train_targets (np.ndarray): Target values (true α₁).
        input_dim (int): Dimensionality of input features.
        num_epochs (int): Maximum number of training epochs.
        learning_rate (float): Learning rate for the optimizer.
        patience (int): Number of epochs to wait for improvement before early stopping.
        save_path (str, optional): If provided, saves the model state_dict to this path.
    
    Returns:
        model: The trained neural network.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GARCHParameterNN(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_features_tensor = torch.tensor(train_features, dtype=torch.float32).to(device)
    train_targets_tensor = torch.tensor(train_targets, dtype=torch.float32).view(-1, 1).to(device)

    best_loss = np.inf
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_features_tensor)
        loss = criterion(outputs, train_targets_tensor)
        loss.backward()
        optimizer.step()

        current_loss = loss.item()
        if current_loss < best_loss:
            best_loss = current_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        if (epoch+1) % 500 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {current_loss:.6f}")
    
    model.load_state_dict(best_model_state)
    
    # Save the model state to a stock-specific file if a save_path is provided.
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
    
    return model

# -------------------------------
# Prediction Function: Given input features and empirical moments, return the full set of parameters
# -------------------------------
def predict_parameters(model, input_features, gamma4_emp, sigma2_emp):
    """
    Given a trained model and input features (e.g., [σ²_emp, Γ₄,emp]),
    predict α₁ and compute β₁ and α₀.
    
    Returns:
        alpha0, alpha1, beta: Predicted GARCH parameters.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)
        alpha1_pred = model(input_tensor).cpu().numpy().flatten()
    
    beta_pred = compute_beta(alpha1_pred, gamma4_emp)
    alpha0_pred = compute_alpha0(alpha1_pred, beta_pred, sigma2_emp)
    
    return alpha0_pred, alpha1_pred, beta_pred

# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    # For demonstration, we generate dummy training data.
    num_samples = 1000
    sigma2_emp = np.random.uniform(0.0001, 0.01, num_samples)
    gamma4_emp = np.random.uniform(3.1, 5.0, num_samples)
    true_alpha1 = np.random.uniform(0.0, 0.3, num_samples)
    
    # Feature vector: [σ²_emp, Γ₄,emp]
    train_features = np.column_stack((sigma2_emp, gamma4_emp))
    train_targets = true_alpha1  # Replace with targets from an estimator in practice
    
    # For a specific stock, say "AAPL", save the model under a unique name.
    stock_symbol = "AAPL"
    save_filename = f"garch_nn_{stock_symbol}.pth"
    
    # Train the network and save it to a stock-specific file
    model = train_model(train_features, train_targets, input_dim=train_features.shape[1], save_path=save_filename)
    print("Training completed.")
    
    # Example prediction on new data:
    new_sigma2 = np.array([0.005])
    new_gamma4 = np.array([3.8])
    new_features = np.column_stack((new_sigma2, new_gamma4))
    alpha0, alpha1, beta = predict_parameters(model, new_features, new_gamma4, new_sigma2)
    
    print("Predicted Parameters:")
    print("α₀:", alpha0)
    print("α₁:", alpha1)
    print("β₁:", beta)
