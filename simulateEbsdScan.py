import numpy as np
import pandas as pd

def create_test_array(m, n):
    """
    Create a test linear array representing IQ values for an m x n grid.
    Args:
        m (int): Number of rows.
        n (int): Number of columns.
    Returns:
        np.ndarray: Linear array with values from 1 to m*n.
    """
    return np.arange(1, m * n + 1)

def reshape_array(linear_array, m, n, order):
    """
    Reshape a linear array into a 2D array with the specified order.
    Args:
        linear_array (np.ndarray): Input linear array.
        m (int): Number of rows.
        n (int): Number of columns.
        order (str): 'C' for row-major order, 'F' for column-major order.
    Returns:
        np.ndarray: Reshaped 2D array.
    """
    return np.reshape(linear_array, (m, n), order=order)

def linearize_array(array, order):
    """
    Flatten a 2D array back into a linear array with the specified order.
    Args:
        array (np.ndarray): Input 2D array.
        order (str): 'C' for row-major order, 'F' for column-major order.
    Returns:
        np.ndarray: Linearized 1D array.
    """
    return array.flatten(order=order)

def create_comparison_table(initial_array, c_order_array, f_order_array, c_linearized, f_linearized):
    """
    Create a table comparing the arrays and their linearizations.
    Args:
        initial_array (np.ndarray): Initial linear array.
        c_order_array (np.ndarray): Reshaped array using C-order.
        f_order_array (np.ndarray): Reshaped array using Fortran-order.
        c_linearized (np.ndarray): Linearized array from C-order reshaping.
        f_linearized (np.ndarray): Linearized array from Fortran-order reshaping.
    Returns:
        pd.DataFrame: Comparison table.
    """
    data = {
        'Initial Linear Array': initial_array,
        'C-order Reconstructed': c_order_array.flatten(),
        'F-order Reconstructed': f_order_array.flatten(),
        'C-order Linearized': c_linearized,
        'F-order Linearized': f_linearized
    }
    return pd.DataFrame(data)

def main_test():
    # Parameters
    m, n = 3, 5  # Grid size

    # Step 1: Create a test linear array
    initial_array = create_test_array(m, n)

    # Step 2: Reshape using both orders
    c_order_array = reshape_array(initial_array, m, n, order='C')
    f_order_array = reshape_array(initial_array, m, n, order='F')

    # Step 3: Linearize back into 1D arrays
    c_linearized = linearize_array(c_order_array, order='C')
    f_linearized = linearize_array(f_order_array, order='F')

    # Step 4: Create a comparison table
    comparison_table = create_comparison_table(
        initial_array, c_order_array, f_order_array, c_linearized, f_linearized
    )

    # Output the table
    print(comparison_table)
    return comparison_table

# Run the main test function
comparison_table = main_test()
