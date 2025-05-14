import numpy as np
import re

class simplex:

    def _parse_objective_string(self, expression):
        """
        This method will parse the objective function string (e.g., "max -6+7+4").
        
        Parameters:
            expression (str): The string representation of the objective function, 
                            like "max -6+7+4" or "min 1+2-3".
        
        Returns:
            c_sign (str): Either 'max' or 'min', indicating the type of optimization.
            coefficients (list): List of coefficients parsed from the string.
        """
        # Check if the expression contains 'max' or 'min'
        if "max" in expression.lower():
            c_sign = "max"
            expression = expression.lower().replace("max", "").strip()
        elif "min" in expression.lower():
            c_sign = "min"
            expression = expression.lower().replace("min", "").strip()
        else:
            raise ValueError("Expression must contain 'max' or 'min' for the objective function.")
        
        # Ensure the expression starts with a '+' or '-'
        if expression and not expression.startswith(('+', '-')):
            expression = '+' + expression
        
        # Extract the terms (coefficients) with signs
        parts = re.findall(r'[+-]?\d+', expression)  # This will handle terms like -6, +7, etc.
        coefficients = [float(part) for part in parts]
        
        return c_sign, coefficients
        
    
    def _split_with_signs(self, expression):
        if not expression.startswith(('+', '-')):
            expression = '+' + expression
        parts = re.findall(r'[+-]\d+', expression)
        return [float(part) for part in parts]

    def _format_text(self, ab):
        ab = list(ab)
        A = []
        b = []
        signs = []
        for line in ab:
            # Handle both string format and list format
            if isinstance(line, str):
                parts = line.split(" ")
                lhs = parts[0]
                operator = parts[1]
                rhs = parts[2]
            elif isinstance(line, list):
                # Assuming format: [lhs, operator, rhs]
                lhs, operator, rhs = line
            else:
                raise TypeError("Constraint must be either a string or a list")
                
            A.append(self._split_with_signs(lhs))
            b.append(float(rhs))
            signs.append(operator)
        return A, b, signs

    def store_tableau(self, tableau, var_count, phase=None):
        """
        Stores a formatted string of the current simplex tableau in self.tableau_log.

        Parameters:
            tableau (2D numpy array): The simplex tableau
            var_count (int): Number of original decision variables (not including slack/artificial)
            phase (str, optional): Indicator for which phase of the algorithm we're in
        """
        if not hasattr(self, 'tableau_log'):
            self.tableau_log = []

        m, n = tableau.shape
        slack_and_artificial_count = n - var_count - 1

        # Create headers
        headers = [f"x{i+1}" for i in range(var_count)]
        
        # Add appropriate headers based on phase and remaining columns
        if phase == "Phase II":
            # In Phase II, artificial variables are removed, so we only show slack variables
            slack_headers = [f"s{i+1}" for i in range(n - var_count - 1)]  # All non-original, non-RHS variables
            headers.extend(slack_headers)
        elif hasattr(self, 'slack_count') and hasattr(self, 'artificial_count'):
            # In Phase I, show both slack and artificial variables
            slack_headers = [f"s{i+1}" for i in range(self.slack_count)]
            art_headers = [f"a{i+1}" for i in range(self.artificial_count)]
            headers.extend(slack_headers + art_headers)
        else:
            # Fallback if counts aren't tracked
            extra_headers = [f"s{i+1}" for i in range(slack_and_artificial_count)]
            headers.extend(extra_headers)
            
        headers.append("RHS")

        lines = []
        if phase:
            lines.append(f"Tableau: {phase}")
        else:
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

    def __init__(self, c, Ab, c_sign="max"):
        """
        Solves a linear programming problem using the Simplex algorithm.
        Handles all constraint types: <=, >= and =
        
        Maximize/Minimize: c^T x
        Subject to:   Ax (<=, =, >=) b
                     x >= 0

        Parameters:
            c (list): Coefficients for the objective function
            Ab (list of strings): Constraints in the format "ax1+bx2+... (<=,=,>=) c"
            c_sign (str): "max" for maximization, "min" for minimization
        """
        c_sign, c = self._parse_objective_string(c)
        c = np.array(c)
        A, b, signs = self._format_text(Ab)
        A = np.array(A)
        b = np.array(b)
        fix_sign = 1
        # Handle negative RHS values by multiplying the constraint by -1
        for i in range(len(b)):
            if b[i] < 0:
                fix_sign = -1
                b[i] *= -1
                A[i] *= -1 
                if signs[i] == "<=":
                    signs[i] = ">="
                elif signs[i] == ">=":
                    signs[i] = "<="
                # "=" stays the same
           
  
            
        m, n = A.shape  # m constraints, n decision variables
        
        # Count variables of each type
        self.slack_count = sum(1 for sign in signs if sign == "<=")
        self.artificial_count = sum(1 for sign in signs if sign in [">=", "="])
      

        # Determine if we need Phase I (artificial variables)
        need_phase_one = self.artificial_count > 0
        
        # Initialize the simplex tableau for Phase I or directly for Phase II
        if need_phase_one:
            # For Phase I, we include original variables, slack variables, artificial variables,
            # and the RHS column
            total_cols = n + self.slack_count + self.artificial_count + 1
            tableau = np.zeros((m + 1, total_cols))
            
            # Fill constraint rows
            slack_idx = n
            artificial_idx = n + self.slack_count
            artificial_rows = []  # Keep track of rows with artificial variables
            
            for i in range(m):
                # Add original variables
                tableau[i, :n] = A[i]
                
                # Add slack and artificial variables based on constraint type
                if signs[i] == "<=":
                    tableau[i, slack_idx] = 1.0
                    slack_idx += 1
                elif signs[i] == ">=":
                    tableau[i, slack_idx] = -1.0  # Surplus variable (negative slack)
                    tableau[i, artificial_idx] = 1.0  # Artificial variable
                    artificial_rows.append(i)
                    slack_idx += 1
                    artificial_idx += 1
                elif signs[i] == "=":
                    tableau[i, artificial_idx] = 1.0  # Only artificial variable
                    artificial_rows.append(i)
                    artificial_idx += 1
                
                # Add RHS
                tableau[i, -1] = b[i]
            
            # Create Phase I objective function (minimize sum of artificial variables)
            # This is done by setting coefficients of artificial variables to 1 in the objective row
            phase1_obj_row = np.zeros(total_cols)
            artificial_start = n + self.slack_count
            phase1_obj_row[artificial_start:artificial_start + self.artificial_count] = 1.0
            
            # Adjust objective row by subtracting artificial rows
            for i in artificial_rows:
                phase1_obj_row -= tableau[i]
            
            tableau[-1] = phase1_obj_row
            
            # Run Phase I
            tableau = self._solve_tableau(tableau, n, "Phase I")
            
            # Check if all artificial variables are zero (feasible solution)
            if abs(tableau[-1, -1]) > 1e-10:  # If not close to zero
                raise Exception("No feasible solution exists")
            
            # Prepare for Phase II by removing artificial columns and setting up original objective
            # Keep track of which columns to keep (all except artificial)
            keep_cols = list(range(n + self.slack_count)) + [total_cols - 1]  # Original vars, slack vars, RHS
            tableau = tableau[:, keep_cols]
            
            # Update artificial count to reflect removed columns
            self.artificial_count = 0
            
            # Reset objective row for Phase II
            tableau[-1, :] = 0
            tableau[-1, :n] = c * (-1 if c_sign.lower() == "max" else 1)
            
            # Zero out basic variables in objective row
            for i in range(m):
                # Find the basic variable in this row
                basic_var = -1
                for j in range(tableau.shape[1] - 1):  # Skip RHS
                    if abs(tableau[i, j] - 1.0) < 1e-10 and sum(abs(tableau[k, j]) for k in range(m) if k != i) < 1e-10:
                        basic_var = j
                        break
                
                if basic_var >= 0 and basic_var < n:  # If it's an original variable
                    tableau[-1] -= tableau[-1, basic_var] * tableau[i]
            
            # Run Phase II
            tableau = self._solve_tableau(tableau, n, "Phase II")
            
        else:
            # Direct Phase II if no artificial variables needed
            total_cols = n + self.slack_count + 1  # Original vars + slack vars + RHS
            tableau = np.zeros((m + 1, total_cols))
            
            # Fill constraint rows
            slack_idx = n
            for i in range(m):
                # Add original variables
                tableau[i, :n] = A[i]
                
                # Add slack variable for <= constraints
                if signs[i] == "<=":
                    tableau[i, slack_idx] = 1.0
                    slack_idx += 1
                
                # Add RHS
                tableau[i, -1] = b[i]
            
            # Fill objective row for maximization
            sign = -1 if c_sign.lower() == "max" else 1
            tableau[-1, :n] = c * sign
            
            # Run Phase II directly
            tableau = self._solve_tableau(tableau, n, "Phase II")
        
        # Extract optimal solution
        self.optimal = np.zeros(n)
        # Find basic variables
        for j in range(n):
            # Check if column j has exactly one 1 and rest zeros
            one_count = 0
            one_row = -1
            for i in range(m):
                if abs(tableau[i, j] - 1.0) < 1e-10:
                    one_count += 1
                    one_row = i
                elif abs(tableau[i, j]) > 1e-10:
                    one_count = 0
                    break
            
            if one_count == 1:
                self.optimal[j] = tableau[one_row, -1]
        
        # Get optimal objective value
        self.value = tableau[-1, -1] * fix_sign
        self.tableau_final_value = tableau[-1, -1] 

    def _solve_tableau(self, tableau, original_var_count, phase=None):
        """
        Solves the simplex tableau using the simplex algorithm.
        
        Parameters:
            tableau (numpy.ndarray): The simplex tableau
            original_var_count (int): Number of original decision variables
            phase (str, optional): Which phase of the algorithm we're in
            
        Returns:
            numpy.ndarray: The optimized tableau
        """
        m = tableau.shape[0] - 1  # Number of constraints (excluding objective row)

    

        # Main loop of the Simplex method
        max_iterations = 100  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            self.store_tableau(tableau, original_var_count, phase)
            
            # Step 1: Identify entering variable (most negative value in objective row)
            col = np.argmin(tableau[-1, :-1])  # Ignore RHS column
            if tableau[-1, col] >= -1e-10:  # Using small tolerance for floating-point issues
                break  # Optimal solution reached
            
            # Step 2: Identify leaving variable (minimum ratio test)
            ratios = []
            for i in range(m):
                if tableau[i, col] > 1e-10:  # Positive coefficient in pivot column
                    ratios.append(tableau[i, -1] / tableau[i, col])
                else:
                    ratios.append(np.inf)
                    
            if all(r == np.inf for r in ratios):
                raise Exception("Linear program is unbounded")
                
            # Find minimum positive ratio
            row = np.argmin(ratios)

            
            # Step 3: Perform pivoting
            pivot = tableau[row, col]
            tableau[row] /= pivot  # Normalize pivot row
            
            # Zero out the pivot column in all other rows
            for i in range(tableau.shape[0]):
                if i != row:
                    tableau[i] -= tableau[i, col] * tableau[row]

        if iteration >= max_iterations:
            print("Warning: Maximum iterations reached, solution may not be optimal")
            
        return tableau

