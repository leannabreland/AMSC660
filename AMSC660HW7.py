import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import matplotlib.pyplot as plt


#Part 1a
data = pd.read_csv('MovieRankingData2024.csv')

data_numeric = data.iloc[:, 2:].apply(pd.to_numeric, errors='coerce')

A = data_numeric.to_numpy()

def P_Omega(A, mask):
    return A * mask

def low_rank_matrix_completion(A, ranks, lambdas, max_iter=100):
    mask = ~np.isnan(A)
    A_filled = np.nan_to_num(A)
    m, n = A.shape
    results = {}

    for rank in ranks:
        for lambda_reg in lambdas:
            X = np.random.randn(m, rank)
            Y = np.random.randn(n, rank)
            residuals = []

            for iteration in range(max_iter):
                for i in range(m):
                    observed_indices = mask[i, :]
                    Y_obs = Y[observed_indices, :]
                    A_obs = A_filled[i, observed_indices]
                    if Y_obs.shape[0] > 0:
                        X[i, :] = lstsq(Y_obs.T @ Y_obs + (lambda_reg/2) * np.eye(rank), Y_obs.T @ A_obs, rcond=None)[0]

                for j in range(n):
                    observed_indices = mask[:, j]
                    X_obs = X[observed_indices, :]
                    A_obs = A_filled[observed_indices, j]
                    if X_obs.shape[0] > 0:
                        Y[j, :] = lstsq(X_obs.T @ X_obs + (lambda_reg/2) * np.eye(rank), X_obs.T @ A_obs, rcond=None)[0]

                A_hat = X @ Y.T
                error = np.sqrt(np.sum(((P_Omega(A_filled, mask) - P_Omega(A_hat, mask)) ** 2)))
                residuals.append(error)
                print(f"Rank {rank}, Lambda {lambda_reg}, Iteration {iteration + 1}, Error: {error:.4f}")

            results[(rank, lambda_reg)] = residuals

    fig, axes = plt.subplots(4, 2, figsize=(10, 10))
    axes = axes.flatten()

    for idx, rank in enumerate(ranks):
        ax = axes[idx]
        for lambda_reg in lambdas:
            ax.plot(results[(rank, lambda_reg)], label=f"Lambda {lambda_reg}")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Residual Norm")
        ax.set_title(f"Residual Norms for Rank {rank}")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(True)

    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.show()

    return results

ranks = [1, 2, 3, 4, 5, 6, 7]
lambdas = [0.1, 1, 10]
max_iter = 30

results = low_rank_matrix_completion(A, ranks, lambdas, max_iter)

#Part 1b

def soft_threshold(M, lambda_val):
    U, sigma, Vt = np.linalg.svd(M, full_matrices=False)
    sigma_thresholded = np.maximum(sigma - lambda_val, 0)
    return U @ np.diag(sigma_thresholded) @ Vt


def nuclear_norm_matrix_completion(A, lambda_val, max_iter=100):
    mask = ~np.isnan(A)
    A_filled = np.nan_to_num(A)

    M = A_filled.copy()

    residuals = []

    for iteration in range(max_iter):
        M_obs = M + mask * (A_filled - M)

        M_new = soft_threshold(M_obs, lambda_val)

        error = np.linalg.norm(mask * (A_filled - M_new), 'fro')
        residuals.append(error)

        print(f"Iteration {iteration + 1}, Lambda {lambda_val}, Error: {error:.4f}")

        M = M_new

    return M, residuals


lambda_values = [0.1, 1, 10]
max_iter = 30

results_nuclear_norm = {}
for lambda_val in lambda_values:
    M_completed, residuals = nuclear_norm_matrix_completion(A, lambda_val, max_iter)
    results_nuclear_norm[lambda_val] = residuals

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for lambda_val in lambda_values:
    plt.plot(results_nuclear_norm[lambda_val], label=f"Lambda {lambda_val}")
plt.xlabel("Iteration")
plt.ylabel("Residual Norm")
plt.title("Residual Norms for Different Lambda Values (Nuclear Norm Penalization)")
plt.legend()
plt.grid(True)
plt.show()
