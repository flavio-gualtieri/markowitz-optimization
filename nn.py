import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class GARCHParameterNN(nn.Module):

    def __init__(self, input_dim):
        super(GARCHParameterNN, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2048),
            nn.ReLU(),
            nn.Linear(2048, 2048),
            nn.ReLU(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    

    def forward(self, x):
        return self.model(x)



def compute_beta(alpha1, gamma4_emp):
    eps = 1e-8
    denominator = (1 - 6 * (alpha1 ** 2) + eps)
    term = (1 - 2 * (alpha1 ** 2) / denominator) * (gamma4_emp - 3)
    term = np.maximum(term, 0)  # ensure nonnegative before sqrt
    beta = np.sqrt(term) - alpha1
    return beta



def compute_alpha0(alpha1, beta1, sigma2_emp):
    return sigma2_emp * (1 - alpha1 - beta1)



def train_model(train_features, train_targets, input_dim, num_epochs=5000, learning_rate=0.01, patience=100, save_path=None):
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



def predict_parameters(model, input_features, gamma4_emp, sigma2_emp):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_features, dtype=torch.float32).to(device)
        alpha1_pred = model(input_tensor).cpu().numpy().flatten()
    
    beta_pred = compute_beta(alpha1_pred, gamma4_emp)
    alpha0_pred = compute_alpha0(alpha1_pred, beta_pred, sigma2_emp)

    alpha0_pred = np.maximum(alpha0_pred, 1e-8) # Ensure omega > 0 (small positive value)
    alpha1_pred = np.maximum(alpha1_pred, 0)     # Ensure alpha1 >= 0
    beta_pred = np.maximum(beta_pred, 0)         # Ensure beta1 >= 0

    # Stationarity constraint (alpha1 + beta1 < 1). If violated, scale down.
    sum_alpha_beta = alpha1_pred + beta_pred
    if sum_alpha_beta >= 1:
        scaling_factor = 0.99 / sum_alpha_beta # Scale down to sum to 0.99 (or less than 1)
        alpha1_pred *= scaling_factor
        beta_pred *= scaling_factor
    
    return alpha0_pred, alpha1_pred, beta_pred

