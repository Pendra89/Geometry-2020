import numpy as np
from numpy import array
from numpy import transpose
from numpy import dot
from numpy import argmax
from numpy import abs
from numpy.linalg import det
from numpy.linalg import matrix_rank
from numpy.linalg import eig
from numpy.linalg import solve
from numpy.linalg import inv


def linear_equations(matrix, vector) -> array:
    """
    this function resolve a system of linear equations
    :param matrix: matrix of coefficients
    :param vector: vector of constant terms

    >>> linear_equations(np.eye(2),np.array([1,1]))
    The system has a single unique solution.
    [1. 1.]

    >>> linear_equations(np.array([[1,0],[1,0]]),np.array([1,0]))
    The system has no solution.

    """
    B = np.c_[matrix, vector]
    rank_A = matrix_rank(matrix)
    rank_B = matrix_rank(B)
    if rank_A == rank_B:
        if rank_A == len(matrix):
            print(f'\n The system has a single unique solution.\n {solve(matrix, vector)}\n ')
            return solve(matrix, vector)
        else:
            print('\n The system has infinitely many solutions. \n')
            if input('Do you want the matrix after the gauss_elimination elimination? [y/n]\n') == 'y':
                S = gauss_elimination(B)
                print(S)
                return S
    else:
        print('\n The system has no solution.\n')
        return None


# esercizio 2


def linear_dependence(matrix: array) -> int:
    """
    This function answer to the question "Are these vectors linearly independent?"

    :param matrix: matrix with vectors as rows
    :return: the number of linearly independent vectors
    """
    rank = matrix_rank(matrix)
    if rank == matrix.shape[0]:
        print('The vectors are linearly independents')
    else:
        print(f'The vectors are linearly dependents and only {rank} of them are linearly independents')
        if input('Do you want the matrix after the gauss_elimination elimination? [y/n]\n') == 'y':
            S = gauss_elimination(matrix)
            print(S)
    return rank


# esercizio3


def cartesian_representation_line(vec_1: np.array, vec_2: np.array, type: int = 1) -> None:
    """
    This function print the cartesian presentation of a line
    a: numpy-array of the
    b: numpy-array of the

    :param vec_1: first point
    :param vec_2: direction (type = 0) or the second point (type = 1)
    :param type: it switches between two points and one point and a direction
    """
    if type:
        vec_2 = vec_2 - vec_1
    for i in range(len(vec_1)):
        print(f' x_{i + 1} = {vec_1[i]} + {vec_2[i]}t')
    return None


def gauss_elimination(matrix) -> np.array:
    """
    This function compute Gauss elimination process
    :param matrix: generic matrix
    :return: matrix after the Gauss elimination
    """
    import sympy
    return np.array(sympy.Matrix(matrix).rref()[0])




def conic_section_classification(coeff: list) -> None:
    """
    This function provides a classification of a conic section

    :param coeff: list of the coefficient of the equation of the conic section

    if the equation is

    A x^2 + B xy + C y^2 + D x + E y + F = 0

    then the array coeff is

    [A,B,C,D,E,F]
    """

    A = array([[coeff[0], coeff[1] / 2, coeff[3] / 2], [coeff[1] / 2, coeff[2], coeff[4] / 2],
               [coeff[3], coeff[4] / 2, coeff[5]]])
    rank = matrix_rank(A)
    if rank == 3:
        d = det(A[:2, :2])
        # remember that we have a finite precision on floats, for this reason we consider 1e-09 as tolerance
        if d > 1e-09:
            print('This conic section is an ellipse')
        elif d < -1e-09:
            print('This conic section is a hyperbola')
        else:
            print('This conic section is a parabola')


    elif rank == 2:
        print('This conic section is a degenerate conic, ', end="")
        d = det(A[:2, :2])
        if d > 1e-09:
            print('in particular we have one point')
        elif d < -1e-09:
            print('in particular we have two incident lines')
        else:
            print('in particular we have two parallel lines')


    else:
        print('This conic section is a degenerate conic, in particular we have two coincident lines')
    return None


if __name__ == '__main__':
    linear_dependence(np.array([[1,2],[2,4]]))

    # linear_equations(np.eye(2),np.array([1,1]))
    # pass
