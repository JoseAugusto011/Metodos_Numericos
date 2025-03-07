import numpy as np

class GaussSeidelSolver:
    """
    A class to solve a system of linear equations using the Gauss-Seidel iterative method.

    Attributes:
        A (numpy.ndarray): The coefficient matrix of the system.
        B (numpy.ndarray): The constant terms of the system.
        X0 (numpy.ndarray): The initial guess for the solution.
        tol (float): The tolerance for the error in the final answer.
        max_iterations (int): The maximum number of iterations allowed.
    """

    def __init__(self, A, B, X0=None, tol=1e-6, max_iterations=1000):
        """
        Initializes the GaussSeidelSolver with the given parameters.

        Parameters:
            A (numpy.ndarray): The coefficient matrix of the system.
            B (numpy.ndarray): The constant terms of the system.
            X0 (numpy.ndarray, optional): The initial guess for the solution. Defaults to a vector of ones.
            tol (float, optional): The tolerance for the error in the final answer. Defaults to 1e-6.
            max_iterations (int, optional): The maximum number of iterations allowed. Defaults to 1000.
        """
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float).flatten()
        self.tol = tol
        self.max_iterations = max_iterations

        # Check if A is a square matrix
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Matrix A must be a square matrix.")

        # Check if B is a column vector with the same number of rows as A
        if self.B.shape[0] != self.A.shape[0]:
            raise ValueError("Matrix B must be a column vector with the same number of rows as A.")

        # Initialize X0 if not provided
        if X0 is None:
            self.X0 = np.ones(self.A.shape[0])
        else:
            self.X0 = np.array(X0, dtype=float).flatten()
            if self.X0.shape[0] != self.A.shape[0]:
                raise ValueError("Initial guess X0 must have the same number of rows as A.")

        # Separate A into D, L, and U
        self.D = np.diag(np.diag(self.A))
        self.L = np.tril(self.A) - self.D
        self.U = np.triu(self.A) - self.D

        # Check for convergence
        self._check_convergence()

    def _check_convergence(self):
        """
        Checks if the Gauss-Seidel method will converge for the given matrix A.
        """
        inv_D_plus_L = np.linalg.inv(self.D + self.L)
        spectral_radius = np.max(np.abs(np.linalg.eigvals(-inv_D_plus_L @ self.U)))
        if spectral_radius >= 1:
            raise ValueError("The Gauss-Seidel method will not converge for this matrix.")

    def solve(self):
        """
        Solves the system of linear equations using the Gauss-Seidel method.

        Returns:
            numpy.ndarray: The solution vector X.
            int: The number of iterations performed.
        """
        X = self.X0.copy()
        for iteration in range(self.max_iterations):
            X_new = np.zeros_like(X)
            for i in range(self.A.shape[0]):
                X_new[i] = (self.B[i] - np.dot(self.A[i, :i], X_new[:i]) - np.dot(self.A[i, i+1:], X[i+1:])) / self.A[i, i]
            if np.linalg.norm(X_new - X, np.inf) < self.tol:
                return X_new, iteration + 1
            X = X_new
        raise RuntimeError("Gauss-Seidel method did not converge within the maximum number of iterations.")

class TDMASolver:
    """
    A class to solve a system of linear equations using the Tridiagonal Matrix Algorithm (TDMA).

    Attributes:
        A (numpy.ndarray): The tridiagonal coefficient matrix of the system.
        B (numpy.ndarray): The constant terms of the system.
    """

    def __init__(self, A, B):
        """
        Initializes the TDMASolver with the given tridiagonal matrix A and constant vector B.

        Parameters:
            A (numpy.ndarray): The tridiagonal coefficient matrix of the system.
            B (numpy.ndarray): The constant terms of the system.
        """
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float).flatten()

        # Check if A is a square matrix
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Input matrix A must be square.")

        # Check if B is a column vector with the same number of rows as A
        if self.B.shape[0] != self.A.shape[0]:
            raise ValueError("Matrix B must be a column vector with the same number of rows as A.")

        # Extract the diagonals and make them modifiable
        self.d = np.diag(self.A).copy()           # Main diagonal (copy to make it modifiable)
        self.b = np.diag(self.A, -1).copy()       # Sub-diagonal (below the main diagonal)
        self.a = np.diag(self.A, 1).copy()        # Super-diagonal (above the main diagonal)

        # Pad the sub-diagonal and super-diagonal with zeros for easier indexing
        self.b = np.insert(self.b, 0, 0)   # Add a zero at the beginning
        self.a = np.append(self.a, 0)      # Add a zero at the end

    def solve(self):
        """
        Solves the system of linear equations using the Tridiagonal Matrix Algorithm (TDMA).

        Returns:
            numpy.ndarray: The solution vector X.
        """
        n = len(self.d)
        c = self.B.copy()  # Copy of B to use for forward elimination

        # Forward Elimination
        for i in range(1, n):
            factor = self.b[i] / self.d[i - 1]  # Compute the factor for elimination
            self.d[i] -= factor * self.a[i - 1]  # Update diagonal elements
            c[i] -= factor * c[i - 1]            # Update constant vector

        # Backward Substitution
        X = np.zeros(n)
        X[-1] = c[-1] / self.d[-1]  # Compute the last element of X
        for i in range(n - 2, -1, -1):
            X[i] = (c[i] - self.a[i] * X[i + 1]) / self.d[i]  # Compute remaining elements of X

        return X

class LUSolver:
    """
    A class to solve a system of linear equations using LU Decomposition.

    Attributes:
        A (numpy.ndarray): The coefficient matrix of the system.
        B (numpy.ndarray): The constant terms of the system.
    """

    def __init__(self, A, B):
        """
        Initializes the LUSolver with the given coefficient matrix A and constant vector B.

        Parameters:
            A (numpy.ndarray): The coefficient matrix of the system.
            B (numpy.ndarray): The constant terms of the system.
        """
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float).flatten()

        # Check if A is a square matrix
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Coefficient matrix A must be square.")

        # Check if the dimensions of A and B are compatible
        if self.A.shape[0] != self.B.shape[0]:
            raise ValueError("Coefficient matrix A and vector B must have compatible dimensions.")

    def decompose(self):
        """
        Performs LU decomposition on the coefficient matrix A.

        Returns:
            L (numpy.ndarray): Lower triangular matrix.
            U (numpy.ndarray): Upper triangular matrix.
        """
        n = self.A.shape[0]
        L = np.eye(n)  # Initialize L as identity matrix
        U = np.zeros_like(self.A)  # Initialize U as zeros

        for i in range(n):
            # Compute U's i-th row
            U[i, i:] = self.A[i, i:] - L[i, :i] @ U[:i, i:]
            # Compute L's i-th column
            if i < n - 1:
                L[i+1:, i] = (self.A[i+1:, i] - L[i+1:, :i] @ U[:i, i]) / U[i, i]

        return L, U

    def solve(self):
        """
        Solves the system of linear equations using LU decomposition.

        Returns:
            X (numpy.ndarray): The solution vector.
            L (numpy.ndarray): Lower triangular matrix.
            U (numpy.ndarray): Upper triangular matrix.
        """
        n = self.A.shape[0]
        L, U = self.decompose()

        # Forward substitution: Solve L * C = B
        C = np.zeros(n)
        for i in range(n):
            C[i] = self.B[i] - L[i, :i] @ C[:i]

        # Backward substitution: Solve U * X = C
        X = np.zeros(n)
        for i in range(n-1, -1, -1):
            X[i] = (C[i] - U[i, i+1:] @ X[i+1:]) / U[i, i]

        return X, L, U

class GaussEliminationSolver:
    """
    A class to solve a system of linear equations using Gaussian Elimination.

    Attributes:
        A (numpy.ndarray): The coefficient matrix of the system.
        B (numpy.ndarray): The constant terms of the system.
    """

    def __init__(self, A, B):
        """
        Initializes the GaussEliminationSolver with the given coefficient matrix A and constant vector B.

        Parameters:
            A (numpy.ndarray): The coefficient matrix of the system.
            B (numpy.ndarray): The constant terms of the system.
        """
        self.A = np.array(A, dtype=float)
        self.B = np.array(B, dtype=float).flatten()

        # Check if A is a square matrix
        if self.A.shape[0] != self.A.shape[1]:
            raise ValueError("Coefficient matrix A must be square.")

        # Check if the dimensions of A and B are compatible
        if self.A.shape[0] != self.B.shape[0]:
            raise ValueError("Coefficient matrix A and vector B must have compatible dimensions.")

    def elimination(self, X, i, j):
        """
        Performs Gaussian elimination on the augmented matrix X for the pivot element at (i, j).

        Parameters:
            X (numpy.ndarray): The augmented matrix [A | B].
            i (int): Row index of the pivot element.
            j (int): Column index of the pivot element.

        Returns:
            numpy.ndarray: The updated augmented matrix after elimination.
        """
        nX, mX = X.shape
        a = X[i, j]  # Pivot element

        # Normalize the pivot row
        X[i, :] /= a

        # Eliminate other rows
        for k in range(nX):
            if k == i:
                continue  # Skip the pivot row
            X[k, :] -= X[i, :] * X[k, j]  # Subtract to make the element zero

        return X

    def solve(self):
        """
        Solves the system of linear equations using Gaussian Elimination.

        Returns:
            numpy.ndarray: The solution vector.
        """
        n = self.A.shape[0]
        X = np.hstack((self.A, self.B.reshape(-1, 1)))  # Augmented matrix [A | B]

        # Perform Gaussian elimination
        for i in range(n):
            if X[i, i] == 0:
                raise ValueError("Diagonal element zero. Pivoting failed.")
            X = self.elimination(X, i, i)

        # Extract the solution vector
        C = X[:, -1]
        return C


# Example usage:
if __name__ == "__main__":

    print("-=-"*10+"\tTHOMAS\t"+"-=-"*10)

    # Example tridiagonal matrix A and vector B
    A = np.array([
        [4, -1, 0, 0],
        [-1, 4, -1, 0],
        [0, -1, 4, -1],
        [0, 0, -1, 4]
    ])
    B = np.array([15, 10, 10, 10])

    solver = TDMASolver(A, B)
    X = solver.solve()
    print(f"Solution: {X}")

    print("-=-"*10+"\tGauss Seidel\t"+"-=-"*10)

    A = np.array([[4, -1, 0], [-1, 4, -1], [0, -1, 4]])
    B = np.array([15, 10, 10])
    solver = GaussSeidelSolver(A, B)
    X, iterations = solver.solve()
    print(f"Solution: {X}")
    print(f"Iterations: {iterations}")

    print("-=-"*10+"\tDecomposição LU\t"+"-=-"*10)
     # Example matrix A and vector B
    A = np.array([
        [4, -1, 0],
        [-1, 4, -1],
        [0, -1, 4]
    ])
    B = np.array([15, 10, 10])

    solver = LUSolver(A, B)
    X, L, U = solver.solve()
    print(f"Solution X: {X}")
    print(f"Lower triangular matrix L:\n{L}")
    print(f"Upper triangular matrix U:\n{U}")

    print("-=-"*10+"\tEliminação de Gauss\t"+"-=-"*10)

     # Example matrix A and vector B
    A = np.array([
        [1, 2],
        [4, 5]
    ])
    B = np.array([-1, 4])

    solver = GaussEliminationSolver(A, B)
    C = solver.solve()
    print(f"Solution C: {C}")