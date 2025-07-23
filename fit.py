import numpy as np
import pandas as pd
from hamiltonian import hamiltonian_from_params, compute_sparse_matrix, compute_hamiltonian, sparse_matrix


def load_target_eigenvals():
    """Load target eigenvalues from a random row of bandstructures.csv, columns 6-45"""

    df = pd.read_csv('bandstructure.csv', header=None)
    target_vals = df.iloc[1000, 5:45].values
    return target_vals.astype(float)


def gauss_newton_step(J, residuals):
    """Compute the Gauss-Newton step."""
    JTJ = J.T @ J
    JTr = J.T @ residuals
    delta_p = np.linalg.solve(JTJ, JTr)
    return delta_p


def compile_partial_eigenenergies(D_matrices, evect):
    """Compute the partial derivatives of eigenenergies with respect to parameters."""

    partial = np.zeros((len(evect), len(D_matrices)), dtype=complex)

    for j in range(len(evect)):
        # print(f"Eigenvectors: {evect}")
        inner_products = [np.vdot(evect[j], D_matrices[i] @ evect[j]) for i in range(len(D_matrices))]
        # print(inner_products)
        partial[j] = np.array(inner_products)
    return partial


def fit_params(p0, target_eigvals, h_expression, max_iters=20, tol=1e-6):
    D_expression = sparse_matrix(H_expression)
    print("DEBUG: Derivative Symbolic Form has been generated and saved at sparse_matrix_D.csv:")

    p = p0.copy()
    
    for i in range(max_iters):
        print(f"DEBUG: Iteration {i}, current parameters: {p}")

        H = compute_hamiltonian(h_expression, p)
        print(f"DEBUG: Hamiltonian computed for iteration {i}.")

        eig, evect = np.linalg.eigh(H)

        print(f"DEBUG: Eigenvalues computed for iteration {i}: {eig}")
        print(f"DEBUG: Eigenvectors computed for iteration {i}: {evect.shape}")

        residuals = eig - target_eigvals
        print(f"DEBUG: Residuals computed for iteration {i}: {residuals}")
        
        D_matrices = compute_sparse_matrix(D_expression, p)
        print(f"DEBUG: Derivative matrix computed for iteration {i}.")
        J = compile_partial_eigenenergies(D_matrices, evect)
        print(f"DEBUG: Jacobian matrix computed for iteration {i}.")


        norm = np.linalg.norm(residuals)
        print(f"Iteration {i}, residual norm = {norm}")

        if norm < tol:
            break
        
        # Acquire the inverse of the Jacobian product
        j_prod = J.T @ J
        jn_prod_inv = -np.linalg.inv(j_prod)

        #Acquire the jacobian product with the residuals
        jtr = J.T @ residuals

        # Sum for the step update
        step = jn_prod_inv @ jtr
        p = p + step
        
        print(f"DEBUG: Step computed for iteration {i}: {step}")
        print(f"DEBUG: Updated parameters for iteration {i}: {p}")
        print(f"DEBUG: Old parameters for iteration {i}: {p0}")
       
    return p




# Generate an initial guess for the slater-koster parameters
p0 = np.random.rand(29) # has shape (29,)
print("DEBUG: Initial parameters:", p0)

# Generate the Hamiltonian expression from the parameters
H_expression = hamiltonian_from_params()
print("DEBUG: Hamiltonian Symbolic Form has been generated and saved at Hamiltonian.csv:")

e_target = load_target_eigenvals()
print("DEBUG: Target eigenvalues for the Γ point have been loaded successfully:", e_target)


p_final = fit_params(p0, e_target, H_expression)

exit()

print("Final parameters:", p_final)
