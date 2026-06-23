import numpy as np # pyright: ignore[reportMissingImports]

# AX = b
def solve_system_of_equations (A,b) :
    try:
        solution = np.linalg.solve(A,b)
        return solution
    except np.linalg.LinAlgError:
        return "System has no unique solutions"
    
def gaussian_elimination(A, b):
    n = len(A)
    # Augmented matrix
    augmented = np.column_stack((A, b))
    augmented = augmented.astype(float)
    # Forward elimination
    for i in range(n):
        # Partial pivoting
        pivot_row = i
        max_val = abs(augmented[i][i])
        for j in range(i + 1, n):
            if abs(augmented[j][i]) > max_val:
                max_val = abs(augmented[j][i])
                pivot_row = j
        if pivot_row != i:
            augmented[i], augmented[pivot_row] = (
                augmented[pivot_row].copy(),
                augmented[i].copy()
            )
        if abs(augmented[i][i]) == 0:
            return "System has no unique solutions"
        # Normalize pivot row
        pivot = augmented[i][i]
        augmented[i] = augmented[i] / pivot

        # Eliminate below pivot
        for k in range(i + 1, n):
            factor = augmented[k][i]
            augmented[k] = augmented[k] - factor * augmented[i]

    # Back substitution
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):
        x[i] = augmented[i][-1]

        for j in range(i + 1, n):
            x[i] -= augmented[i][j] * x[j]

    return x

A = np.array([ [3,2,-1], [4,2,0], [0,-2,3] ] )
b = np.array( [2, 8, 9])

# Entry point
if __name__ == "__main__":
    print ( solve_system_of_equations(A,b))
