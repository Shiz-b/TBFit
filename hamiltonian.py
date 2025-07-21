"""
This module uses Sympy to generate a symbolic hamiltonian, whos elements are some (random) polynomial
combination of parameters p0-p28 (29 parameters in total). 

Finally, a sparse matrix of partial derivatives of H, with respect to p0-p28 (to give 40x40x29) are created.

Please run "pip install -r requirements.txt" to install the required packages before running this script.
"""

import sympy as sp
import numpy as np
import pandas as pd
import random


def hamiltonian_from_params(params, dimension=40):
    """Create Hermitian Hamiltonian using polynomial combinations of parameters.
    
    Args:
        params (list): List of parameters to use in the Hamiltonian. Function expects 20 parameters.
        dimension (int): Dimension of the Hamiltonian matrix, default is 40.
    Returns:
        sympy.Matrix: A symbolic Hamiltonian matrix of size dimensionxdimension.
    """

    # Initialise empty 40x40 Hamiltonian matrix
    hamiltonian = []
    
    # Fill the matrix with polynomial combinations of parameters (as expressions)
    for i in range(dimension):
        row = []
        for j in range(dimension):
            # Set random parameters to use for polynomial combinations.
            random_index = random.randint(0, len(params)-1)  
            random_index2 = random.randint(0, len(params)-1)

            # When i == j, you are looking at diagonal elements.
            if i == j:
                # Example expression
                row.append(params[random_index]*params[random_index-1]**2 + params[random_index2])
                
            else:
                # To make it Hermitian, ensure that the off-diagonal elements are complex conjugates
                if j < i:
                    row.append(np.conj(hamiltonian[j][i]))
                else:
                    # Another random number to use as a power for polynomial
                    random_power = np.random.randint(1, 4)

                    # Example expression off-diagonal
                    row.append(params[random_index]**(random_power) + params[random_index2])
        hamiltonian.append(row)

    # Convert to sympy Matrix and save as a CSV file, then return the matrix expression
    hamiltonian_expresssion = sp.Matrix(hamiltonian)
    pd.DataFrame(hamiltonian).to_csv('hamiltonian.csv', index=False)
    return hamiltonian_expresssion


def sparse_matrix(H, p):
    """Find partial derivatives of input matrix H, w.r.t. to each parameter in p. This is the sparse matrix D.
    
    Args:
        H (sympy.Matrix): The Hamiltonian matrix to differentiate.
        p (list): List of parameters with respect to which to differentiate.
    Returns:
        list: A list of sympy.Matrix expressions representing the partial derivatives of H with respect to each
        parameter in p. This would be 40x40x29.
    """

    D = [sp.diff(H, param) for param in p] 
    pd.DataFrame(D).to_csv('sparse_matrix_D.csv', index=False)
    return D


# initialise 29 parameter variables p0-p28
p = sp.symbols('p0:29')
H = hamiltonian_from_params(p)
D = sparse_matrix(H, p)