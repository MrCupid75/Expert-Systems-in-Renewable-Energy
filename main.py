import numpy as np
from scipy.optimize import minimize

# Step 1: Linguistic scale mapped to Intuitionistic Fuzzy Numbers (μ, ν)
linguistic_scale = {
    "Equal importance": (0.5, 0.4),
    "Moderate importance": (0.6, 0.3),
    "Strong importance": (0.7, 0.2),
    "Very strong importance": (0.8, 0.1),
    "Extreme importance": (0.9, 0.05)
}

# Step 2: Simulate 5 expert IFN matrices (3x3)
n_criteria = 3
n_experts = 5
linguistic_values = list(linguistic_scale.values())
expert_matrices = []

for i in range(n_experts):
    matrix = np.zeros((n_criteria, n_criteria, 2))
    for r in range(n_criteria):
        for c in range(n_criteria):
            if r == c:
                matrix[r, c] = (1.0, 0.0)  # Identity
            else:
                μ, ν = linguistic_values[np.random.randint(0, len(linguistic_values))]
                matrix[r, c] = (μ, ν)
    expert_matrices.append(matrix)

# Step 3: IFWA Aggregation
def ifwa_aggregation(expert_matrices):
    return np.mean(expert_matrices, axis=0)

# Step 4: Score Function S = μ - ν
def score_function(ifn_matrix):
    return ifn_matrix[:, :, 0] - ifn_matrix[:, :, 1]

# Step 5: Normalize Scores to get Weights
def normalize_scores(S):
    col_sum = np.sum(S, axis=0)
    norm_matrix = S / col_sum
    weights = np.mean(norm_matrix, axis=1)
    return weights

# Step 6: Consistency Check (approx.)
def check_consistency(S):
    CI = np.max(np.abs(S - S.T))
    print(f"Consistency Index (approx): {CI:.4f}")
    if CI < 0.1:
        print("Consistency acceptable.")
    else:
        print("Consistency may need revision.")

# === Run IC-FAHP ===
aggregated_ifn = ifwa_aggregation(np.array(expert_matrices))
S = score_function(aggregated_ifn)
initial_weights = normalize_scores(S)

print("\nInitial IC-FAHP Weights:\n", initial_weights)
check_consistency(S)

# === Step 7: Construct Best (A_B) and Worst (A_W) Vectors ===
# Simulating expert input (normally you collect this)
A_B = np.random.uniform(1, 9, size=n_criteria)  # Best to others
A_W = np.random.uniform(1, 9, size=n_criteria)  # Others to Worst

print("\nA_B vector (Best Comparisons):", A_B)
print("A_W vector (Worst Comparisons):", A_W)

# === Step 8: BBWM Optimization for Refined Weights ===
def bbwm_objective(W):
    W_B = max(W)  # assume best weight is highest
    W_W = min(W)  # assume worst weight is lowest
    diffs = [abs(W_B / W[i] - A_B[i]) for i in range(len(W))] + [abs(W[i] / W_W - A_W[i]) for i in range(len(W))]
    return max(diffs)

# Constraints: weights sum to 1 and all >= 0
constraints = ({'type': 'eq', 'fun': lambda W: np.sum(W) - 1})
bounds = [(0, 1) for _ in range(n_criteria)]

# Initial guess: IC-FAHP weights
result = minimize(bbwm_objective, initial_weights, method='SLSQP', bounds=bounds, constraints=constraints)

refined_weights = result.x
print("\nRefined Weights after BBWM Optimization:\n", refined_weights)

# === Step 9: Monte Carlo Simulation for Robust Decision Weights ===
def monte_carlo_refinement(A_B, A_W, n_simulations=10000):
    samples = []
    for _ in range(n_simulations):
        noise = np.random.normal(0, 0.05, n_criteria)  # small noise to simulate Bayesian prior adjustment
        sample_weight = np.clip(refined_weights + noise, 0, 1)
        sample_weight /= np.sum(sample_weight)  # Normalize
        samples.append(sample_weight)
    samples = np.array(samples)
    return np.mean(samples, axis=0)

final_weights = monte_carlo_refinement(A_B, A_W)
print("\nFinal Robust Weights after Monte Carlo Simulation:\n", final_weights)
