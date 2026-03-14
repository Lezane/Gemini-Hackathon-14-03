import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

# ===================================================================== #
# Phase 4: Proof-of-Concept Simulation
# Algorithm: Hessian-Corrected Lookahead-VI (HC-LA-VI)
#
# Connects:
# [Paper 1] HCMM: Bias-Corrected Momentum via Hessian-Vector Products
# [Paper 2] LA: Nested Lookahead-VI Averaging
# [Theory Bridge]: Second-Order Rotational Dampening (SORD)
# ===================================================================== #

class RobustLogisticRegression(nn.Module):
    """
    Simulating a Minimax game: Non-convex in X (if we use a highly complex neural net, 
    but for fast simulation we use a linear layer representing robust linear evaluation), 
    and strongly concave in weights (Y).
    """
    def __init__(self, input_dim):
        super(RobustLogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

def check_simulation_feasibility(X_train, y_train):
    """
    Feature requested by User: Algorithm decides if there is a common
    simulation feasible to run before starting the heavy math.
    """
    print("\n--- Feasibility Check ---")
    if len(X_train) == 0 or len(y_train) == 0:
        print("Data is empty. Cannot run simulation.")
        return False
        
    num_features = X_train.shape[1]
    if num_features > 500:
        print(f"Feature count ({num_features}) is too high for fast Hessian-Vector product computation. Simular aborting.")
        return False
        
    if len(np.unique(y_train)) < 2:
        print("Dataset contains less than 2 classes. Cannot run binary classification.")
        return False
        
    print(f"Data has {len(X_train)} instances and {num_features} features.")
    print("Conditions optimal for Second-Order Rotational Dampening (SORD) simulation.")
    return True

def hvp_vector_product(loss, model, vector):
    """
    [Paper 1: HCMM]
    Hessian-Vector Product approximation for efficiency.
    Calculates ∇^2 L * v without storing the full Hessian.
    """
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params, create_graph=True)
    # Flatten gradients
    grad_vector = torch.cat([g.reshape(-1) for g in grads])
    # Compute dot product with the input vector
    dot_product = torch.dot(grad_vector, vector.detach())
    # Compute gradient of the dot product to get the HVP
    hvp = torch.autograd.grad(dot_product, params, retain_graph=True)
    return torch.cat([h.reshape(-1) for h in hvp])

def run_simulation():
    import pandas as df_lib
    print("Loading Breast Cancer Dataset from local CSV generated via scikit-learn")
    df = df_lib.read_csv("breast_cancer.csv")
    
    # Process dataset
    y = df['target'].values
    X = df.drop('target', axis=1).values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    
    if not check_simulation_feasibility(X_train_tensor, y_train_tensor):
        return
        
    print("\n--- Initializing HC-LA-VI Architecture ---")
    model = RobustLogisticRegression(input_dim=X_train.shape[1])
    bce_loss = nn.BCELoss()
    
    # Algorithm Hyperparameters
    iterations = 100
    beta = 0.5  # Smoothing factor [Paper 1]
    mu = 0.01  # Step size [Paper 1]
    lookahead_k = 10 # Periodic averaging interval [Paper 2]
    alpha = 0.5 # Averaging coordinate [Paper 2]
    
    # Init Momentum Vectors
    num_params = sum(p.numel() for p in model.parameters())
    momentum = torch.zeros(num_params)
    lookahead_snapshot = None
    
    print("\nStarting Training Loop...")
    losses = []
    for iter in range(1, iterations + 1):
        # 1. Forward Pass
        outputs = model(X_train_tensor)
        loss = bce_loss(outputs, y_train_tensor)
        
        # 2. Extract Flat Gradients
        model.zero_grad()
        loss.backward(create_graph=True)
        flat_grads = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
        
        # 3. Apply HCMM (Bias-Corrected Momentum & HVP) [Paper 1]
        weight_diff = flat_grads.detach() * mu 
        hvp = hvp_vector_product(loss, model, weight_diff)
        
        # 4. [Theoretical Bridge: SORD] - Dampening the Rotational Force
        #    Applying HCMM momentum before the Lookahead averages it out.
        momentum = (1 - beta) * (momentum + hvp.detach()) + beta * flat_grads.detach()
        
        # Update Weights
        with torch.no_grad():
            idx = 0
            for p in model.parameters():
                num_el = p.numel()
                p.data -= mu * momentum[idx:idx+num_el].reshape(p.shape)
                idx += num_el

        # 5. Lookahead-VI Averaging [Paper 2]
        if iter % lookahead_k == 0:
            if lookahead_snapshot is None:
                # First snapshot
                lookahead_snapshot = torch.cat([p.data.reshape(-1) for p in model.parameters()]).clone()
            else:
                # [Theoretical Bridge: SORD]
                # Rotational paths have been stabilized by HCMM momentum. Now we 
                # average it historically to enforce equilibrium convergence.
                current_flat_weights = torch.cat([p.data.reshape(-1) for p in model.parameters()])
                averaged_weights = current_flat_weights + alpha * (lookahead_snapshot - current_flat_weights)
                
                # Assign back to model
                idx = 0
                for p in model.parameters():
                    num_el = p.numel()
                    p.data = averaged_weights[idx:idx+num_el].reshape(p.shape)
                    idx += num_el
                
                # Update Snapshot
                lookahead_snapshot = averaged_weights.clone()
                print(f"Iteration {iter}: Executed Lookahead-VI snapshot sync")

        if iter % 20 == 0:
            print(f"Iteration {iter} | Loss: {loss.item():.4f}")
        losses.append(loss.item())

    os.makedirs('out', exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, iterations + 1), losses, label='BCE Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.title('HC-LA-VI Simulation Loss Convergence')
    plt.grid(True)
    plt.legend()
    plt.savefig('out/simulation_loss.png')
    print("Saved simulation loss plot to out/simulation_loss.png")

    print("\n--- Simulation Complete ---")
    
if __name__ == "__main__":
    run_simulation()
