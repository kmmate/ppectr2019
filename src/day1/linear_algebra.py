"""
Principles of Programmming in Econometrics. 2019 block 0.
Linear algebra functionalities.

@author: Mate Kormos
"""

import numpy as np
import sympy

def getpivot(x):
    """
    Returns the column indices of pivot (nonzero-element) position of array x.
    If a row has no nonzero element, the returned pivot index for it is k.

    Inputs
    ------
    :x: array
        n-by-k array

    Returns
    -------
    n-array with

    """
    n, k = x.shape
    pivots = np.zeros(n)
    for row in range(n):
        pivot_idx = 0
        while x[row, pivot_idx] == 0:
            pivot_idx += 1
            if pivot_idx >= k:  # quit while if col 
                break
        pivots[row] = pivot_idx
    return [int(i) for i in pivots]

def gauss_eliminate(x):
    """
    Performs Gauss elimination and computes reduced row echelon form of x.

    Inputs
    ------
    :x: array
        n-by-k array

    Returns
    -------
    reduced row echelon form of array x

    """
    n, k = x.shape
    x_rref = np.array(x.copy(), dtype=float)  # convert to float for correct precision
    # use each, but the last, row to eliminate the elemets below
    for row in range(n):        
        # sort rows by increasing index of pivot positions
        pivots = getpivot(x_rref)  # indices of pivot positions
        sorted_pivots = np.sort(pivots, kind='stable')  # pivot position indices sorted in increasing order
        x_rref = x_rref[np.argsort(pivots),:]  # sort rows by pivot position indices
        # normalise the pivot to 1 in the actual row, dividing the whole row by its value
        try:
            x_rref[row, :] = x_rref[row, :] / x_rref[row, sorted_pivots[row]]
        except IndexError:  # quit if rank(a) < n and we have reached the last nonzero row
            return x_rref
        # use the row, if not the last, to eliminate elements in the rows below, in the column where the pivot is
        if row != n - 1:
            for row_below in range(row + 1, n):
                x_rref[row_below, :] = x_rref[row_below, :] - x_rref[row_below, sorted_pivots[row]] * x_rref[sorted_pivots[row], :]
    return x_rref


def get_nonzerorows(x):
    """
    Boolean array, true entry when the row of array x not full zero.

    Inputs
    ------
    :x: array
        n-by-k
    Returns
    -------
    boolean array
    """
    n, k = x.shape
    bool_array = []
    for row in range(n):
        non_zero = sum([i != 0 for i in x[row, :]]) > 0
        bool_array.append(non_zero)
    return bool_array

def linearsolver_uppertriangle(a, b):
    """
    Solves the linear equation system ax = b for x.

    Inputs
    ------
    :a: array
        n-by-n full-rank upper-triangular coefficient matrix
    :b: array, constants

    Returns
    -------
    Solution, x, for the system

    Examples
    --------
    >>> a = np.eye(3)  # 3-by-3 identity matrix
    >>> b = 4 * np.ones(3)
    >>> x = linearsolver_square(a, b)
    [4, 4, 4]
    
    """
    n = a.shape[0]
    x = np.zeros(n)  # preallocation
    for i in range(n-1, -1, -1):
        if i == n - 1:
            x[i] = b[i] / a[i, i]
        else:
            # sum of the coeffs times already-solved-for-x's
            s = a[i, i+1:].dot(x[i+1:])
            x[i] = (b[i] - s) / a[i, i]
    return x


def linearsolver(a, b):
    """
    Solves the linear equation system ax = b for x.

    Inputs
    ------
    :a: array
        n-by-k coefficient matrix.
    :b: array, constants

    Returns
    -------
    Solution, x, for the system
    """
    n, k = a.shape 
    augmented = np.c_[a, b]  # augment coeff matrix with constants
    # ranks
    rank_a = np.linalg.matrix_rank(a)
    rank_augmented = np.linalg.matrix_rank(augmented)
    # no solution
    if rank_augmented > rank_a:
        print("There is no solution.")
        return [np.NaN for _ in range(k)]
    # exactly one solution with n=k
    elif n == k and n == rank_a:
        print("There is exactly one solution (n = k case).")
        augmented_rref = gauss_eliminate(augmented)  # do gauss elimination
        # rref coeff matrix and constants
        a_rref = augmented_rref[:, :-1]
        #print("a_rref = \n", a_rref)
        b_rref = augmented_rref[:, -1]
        x = linearsolver_uppertriangle(a_rref, b_rref)
        return x
    # exactly one solution with n > k
    elif n > k:
        print("There is exactly one solution (n > k case).")
        # get rid off superflous rows/equations
        augmented_rref = gauss_eliminate(augmented)  # do gauss elimination
        # rref coeff matrix and constansts
        a_rref = augmented_rref[:, :-1]
        b_rref = augmented_rref[:, -1]
        # the part of the rref  matrix where the rows are not all-zeros (cut down all-zero rows at the end)
        nonzero_rows = get_nonzerorows(a_rref)
        a_rref_nonzero = a_rref[nonzero_rows, :]
        b_rref_nonzero = b_rref[nonzero_rows]
        # solve the system with the derived square rref matrix
        x = linearsolver_uppertriangle(a_rref_nonzero, b_rref_nonzero)
        return x
    # infinitely many solutions
    else:
        print("There are infinitely many solutions.")
        augmented_rref = gauss_eliminate(augmented)  # do gauss elimination
        # rref coeff matrix and constansts
        a_rref = augmented_rref[:, :-1]
        b_rref = augmented_rref[:, -1]
        # the part of the rref  matrix where the rows are not all-zeros (cut down all-zero rows at the end)
        nonzero_rows = get_nonzerorows(a_rref)
        a_rref_nonzero = a_rref[nonzero_rows, :]
        b_rref_nonzero = b_rref[nonzero_rows]
        number_nonzerorows = sum(nonzero_rows)
        pivots = getpivot(a_rref_nonzero)
        # initialise symbolic variables
        x = np.array(sympy.symbols('x0:%d' % k))
        # go up row-by-row and figure out free variables
        for row in range(number_nonzerorows - 1, -1, -1):
            for col in range(k - 1, -1, -1):
                if col == pivots[row]:  # x_col is a pivot, compute its value, else x_col is a free variable, leave it as is
                    try:
                        s = a_rref_nonzero[row, col + 1:].dot(x[col + 1:])
                    except IndexError:  # need this for the last row/column
                        s = 0
                    x[col] = b_rref_nonzero[col] - s  # no division as pivots nomalised to one
        return x
