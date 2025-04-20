
import numpy as np
class simplex:
    
    
    
    def store_tableau(self, tableau, var_count):
        """
        Stores a formatted string of the current simplex tableau in self.tableau_log.

        Parameters:
            tableau (2D numpy array): The simplex tableau
            var_count (int): Number of original decision variables (not including slack variables)
        """
        if not hasattr(self, 'tableau_log'):
            self.tableau_log = []

        m, n = tableau.shape
        slack_count = n - var_count - 1

        # Create headers
        headers = [f"x{i+1}" for i in range(var_count)] + \
                [f"s{i+1}" for i in range(slack_count)] + ["RHS"]

        lines = []
        lines.append("Tableau:")
        lines.append(" | ".join(f"{h:>6}" for h in headers))
        lines.append("-" * (8 * len(headers)))

        # Constraint rows
        for row in range(m - 1):
            lines.append(" | ".join(f"{tableau[row, col]:6.2f}" for col in range(n)))

        lines.append("-" * (8 * len(headers)))
        # Objective function row
        lines.append("Z: " + " | ".join(f"{tableau[-1, col]:6.2f}" for col in range(n)))
        lines.append("")  # blank line between tableaux

        # Save the formatted string
        self.tableau_log.append("\n".join(lines))
        self.steps = len(self.tableau_log)

    def show_steps(self, count = None):
        if count is None:
            count = self.steps  # Use self.steps as the default
        for tableau in self.tableau_log[:count]:
            print(tableau)

            
    def show_range(self, start=0, end=None):
        """
        Displays a specific range of tableau steps.

        Parameters:
            start (int): Start index (inclusive).
            end (int or None): End index (exclusive). If None, shows until the last step.
        """

        if end is None:
            end = self.steps

        for tableau in self.tableau_log[start:end]:
            print(tableau)


    def __init__(self, c, A, b):

        """
        Solves a linear programming problem using the Simplex algorithm.
        
        Maximize:     c^T optimal
        Subject to:   Ax <= b
                    optimal >= 0

        Parameters:
            c (1D numpy array): Coefficients for the objective function (length n)
            A (2D numpy array): Constraint coefficient matrix (m optimal n)
            b (1D numpy array): Right-hand side of constraints (length m)

        Returns:
            optimal (1D array): Optimal values of decision variables
            value (float): Maximum value of the objective function
        """
          
       
        c = np.array(c)
        A = np.array(A)
        b = np.array(b)
        
        m, n = A.shape  # m constraints, n decision variables
        
     
        # Initialize the simplex tableau:
        # It has (m+1) rows (m constraints + 1 objective) and (n + m + 1) columns:
        # - n for original variables
        # - m for slack variables (1 per constraint)
        # - 1 for RHS values
        tableau = np.zeros((m + 1, n + m + 1))

        # Fill constraint rows
        tableau[:m, :n] = A                    # Original coefficients
        tableau[:m, n:n+m] = np.eye(m)         # Slack variable coefficients (identity matrix)
        tableau[:m, -1] = b                   # Right-hand side (RHS)

        # Fill the objective function (last row): -c^T for maximization
        
        tableau[-1, :n] = -c   # Negative since simplex does minimization by default

        # Main loop of the Simplex method
        while True:
            self.store_tableau(tableau, n)  # keep track of each iteration

            # Step 1: Identify entering variable (most negative value in objective row)
            col = np.argmin(tableau[-1, :-1])  # Ignore RHS column
            if tableau[-1, col] >= 0:
                break  # Optimal solution reached if all entries are non-negative

            # Step 2: Identify leaving variable (minimum ratio test)
            ratios = []
            for i in range(m):
                if tableau[i, col] > 0:
                    ratios.append(tableau[i, -1] / tableau[i, col])  # RHS / pivot column
                else:
                    ratios.append(np.inf)  # Cannot be pivot row if pivot column entry is <= 0

            # Pick the row with the smallest positive ratio
            row = np.argmin(ratios)
            if tableau[row, col] <= 0 or all(r == np.inf for r in ratios):
                raise Exception("Linear program is unbounded")  # No valid pivot → LP is unbounded

            # Step 3: Perform pivoting
            pivot = tableau[row, col]               # Get pivot value
            tableau[row, :] /= pivot                # Normalize pivot row

            # Zero out the pivot column in all other rows
            for i in range(m + 1):
                if i != row:
                    tableau[i, :] -= tableau[i, col] * tableau[row, :]

        # Extract optimal solution from the tableau
        self.optimal = np.zeros(n)
        for i in range(m):
            # Look for a basic variable (column has only one 1 and the rest 0s)
            pivot_col = np.argmax(tableau[i, :n])  # Find largest entry in the row (should be 1)
            if np.count_nonzero(tableau[:, pivot_col]) == 1:
                # Assign RHS value if it's a basic variable
                self.optimal[pivot_col] = tableau[i, -1]

        # Get the optimal objective function value (bottom-right of tableau)
        self.value = tableau[-1, -1]
        
