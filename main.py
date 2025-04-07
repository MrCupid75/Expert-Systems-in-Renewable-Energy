import numpy as np

# Step 1: Linguistic scale mapped to Intuitionistic Fuzzy Numbers (μ, ν)
linguistic_scale = {
    "Equal importance": (0.5, 0.4),
    "Moderate importance": (0.6, 0.3),
    "Strong importance": (0.7, 0.2),
    "Very strong importance": (0.8, 0.1),
    "Extreme importance": (0.9, 0.05)
}

# Step 2: Simulate 5 expert IFN matrices (size: 3x3 criteria)
n_criteria = 3
n_experts = 5

# Random assignment from linguistic scale
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

# Step 4: Score function S = μ - ν
def score_function(ifn_matrix):
    return ifn_matrix[:, :, 0] - ifn_matrix[:, :, 1]

# Step 5: Normalize to get final weights
def normalize_scores(S):
    col_sum = np.sum(S, axis=0)
    norm_matrix = S / col_sum
    weights = np.mean(norm_matrix, axis=1)
    return weights

# Step 6: Consistency Check (basic)
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
weights = normalize_scores(S)

# === Output ===
print("Aggregated IFN Matrix (μ, ν):\n", aggregated_ifn)
print("\nScore Matrix (μ - ν):\n", S)
print("\nNormalized Criteria Weights:\n", weights)
check_consistency(S)
