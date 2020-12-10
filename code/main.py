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


# esercizio 1

def linear_equations(matrix, vector) -> None:
    """
    this function resolve a system of linear equations
    :param matrix: matrix of coefficients
    :param vector: vector of constant terms

    >>> linear_equations(np.eye(2),array([1,1]))
    [1,1]

    """
    B = np.c_appen[matrix, vector]
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
    else:
        print('\n The system has no solution.\n')
        return None


# esercizio 2


def linear_dependence(A):
    '''
    This function answer to the question "Are these vectors linearly independent?"

    A : numpy-array matrix with vectors as rows
    '''
    rank = matrix_rank(A)
    if rank == A.shape[0]:
        print('The vectors are linearly independents')
    else:
        print(f'The vectors are linearly dependents and only {rank} of them are linearly independents')
        if input('Do you want the matrix after the gauss_elimination elimination? [y/n]\n') == 'y':
            S = gauss_elimination(A)
            print(S)


# esercizio3


def cartesian_representation_line(a, b, type=1):
    '''
    This function print the cartesian presentation of a line
    a: numpy-array of the first point
    b: numpy-array of the direction (type = 0) or of the second point (type = 1)
    '''
    if type:
        b = b - a
    for i in range(len(a)):
        print(f' x_{i + 1} = {a[i]} + {b[i]}t')



def gauss_elimination(matrix):
    '''
    This function compute  Gauss elimination process
    matrix: numpy-array
    '''

    n = len(matrix)
    m = len(matrix[0])
    np.array(matrix, dtype=float)

    for _ in range(matrix_rank(matrix)):
        pivot = np.argmax(np.transpose(matrix)[_][_:])
        matrix[pivot + _], matrix[_] = matrix[_], matrix[pivot + _]
        for i in range(_ + 1, n):
            coeff = (matrix[i][_] / matrix[_][_])
            for j in range(_, m):
                matrix[i][j] -= coeff * matrix[_][j]
    return matrix


def conic_section_classification(coeff=[]):
    '''
    This function provides a classification of a conic section

    coeff: list of the coefficient of the equation of the conic section

    if the equation is

    A x^2 + B xy + C y^2 + D x + E y + F = 0

    then the array coeff is

    [A,B,C,D,E,F]
    '''

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


if __name__ == '__main__':
    pass


