import pytest

from .arithmetics import (
    add,
    multiply,
    to_echelon_form,
    to_reduced_echelon_form,
    transpose,
    to_augmented_matrix,
    inverse,
)
from .matrix import Matrix


def test_add_raises_exception_different_row_size():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError) as e:
        add(a, b)


def test_add_raises_exception_different_column_size():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[1, 2, 3], [3, 4, 5]])
    with pytest.raises(ValueError) as e:
        add(a, b)


def test_add():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[5, 6], [7, 8]])
    result = add(a, b)
    assert result.equals(Matrix([[6, 8], [10, 12]]))


def test_add_raises_exception_different_row_size():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError) as e:
        add(a, b)


def test_add_raises_exception_different_column_size():
    a = Matrix([[1, 2], [3, 4]])
    b = Matrix([[1, 2, 3], [3, 4, 5]])
    with pytest.raises(ValueError) as e:
        add(a, b)


def test_multiply_valid_matrices():
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[2, 0], [1, 2]])
    result = multiply(matrix1, matrix2)
    expected = Matrix([[4, 4], [10, 8]])
    assert result.equals(expected)


def test_multiply_invalid_dimensions():
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError) as e:
        multiply(matrix1, matrix2)


def test_multiply_with_zero_matrix():
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[0, 0], [0, 0]])
    result = multiply(matrix1, matrix2)
    expected = Matrix([[0, 0], [0, 0]])
    assert result.equals(expected)


def test_multiply_identity_matrix():
    matrix1 = Matrix([[1, 2], [3, 4]])
    identity = Matrix([[1, 0], [0, 1]])
    result = multiply(matrix1, identity)
    assert result.equals(matrix1)


def test_multiply_non_square_matrices():
    matrix1 = Matrix([[1, 2, 3], [4, 5, 6]])
    matrix2 = Matrix([[7, 8], [9, 10], [11, 12]])
    result = multiply(matrix1, matrix2)
    expected = Matrix([[58, 64], [139, 154]])
    assert result.equals(expected)


def test_transpose_non_square_matrix():
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    result = transpose(matrix)
    expected = Matrix([[1, 4], [2, 5], [3, 6]])
    assert result.equals(expected)


def test_to_echelon_form_simple_2x2():
    matrix = Matrix([[1, 2], [3, 4]])
    expected = Matrix([[1, 2], [0, -2]])
    result = to_echelon_form(matrix)
    assert result.equals(expected)


def test_to_echelon_form_3x3():
    matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = Matrix([[1, 2, 3], [0, -3, -6], [0, 0, 0]])
    result = to_echelon_form(matrix)
    # print(result.get_row(2))
    assert result.equals(expected)


def test_to_echelon_form_already_in_echelon():
    matrix = Matrix([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
    expected = Matrix([[1, 2, 3], [0, 1, 4], [0, 0, 1]])
    result = to_echelon_form(matrix)
    assert result.equals(expected)


def test_to_echelon_form_zero_matrix():
    matrix = Matrix([[0, 0], [0, 0]])
    expected = Matrix([[0, 0], [0, 0]])
    result = to_echelon_form(matrix)
    assert result.equals(expected)


def test_to_reduced_echelon_form():
    matrix = Matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    expected = Matrix([[1, 0, -1], [0, 1, 2], [0, 0, 0]])
    result = to_reduced_echelon_form(matrix)
    assert result.equals(expected)


def test_to_reduced_echelon_form_already_reduced():
    matrix = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    expected = Matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    result = to_reduced_echelon_form(matrix)
    assert result.equals(expected)


def test_to_reduced_echelon_form_zero_matrix():
    matrix = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    expected = Matrix([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    result = to_reduced_echelon_form(matrix)
    assert result.equals(expected)


def test_to_reduced_echelon_form_single_row():
    matrix = Matrix([[2, 4, 6]])
    expected = Matrix([[1, 2, 3]])
    result = to_reduced_echelon_form(matrix)
    assert result.equals(expected)


def test_to_reduced_echelon_form_single_column():
    matrix = Matrix([[2], [4], [6]])
    expected = Matrix([[1], [0], [0]])
    result = to_reduced_echelon_form(matrix)
    assert result.equals(expected)


def test_to_augmented_matrix_simple():
    # Test case 1: Simple 2x2 matrices
    matrix_a = Matrix([[1, 2], [3, 4]])
    matrix_b = Matrix([[5, 6], [7, 8]])
    expected = Matrix([[1, 2, 5, 6], [3, 4, 7, 8]])
    result = to_augmented_matrix(matrix_a, matrix_b)
    assert result.equals(expected)


def test_to_augmented_matrix_different_sizes():
    # Test case 2: 3x1 and 3x2 matrices
    matrix_a = Matrix([[1], [2], [3]])
    matrix_b = Matrix([[4, 5], [6, 7], [8, 9]])
    expected = Matrix([[1, 4, 5], [2, 6, 7], [3, 8, 9]])
    result = to_augmented_matrix(matrix_a, matrix_b)
    assert result.equals(expected)


def test_to_augmented_matrix_raises_value_error():
    # Test case 3: Matrices with different row counts (should raise ValueError)
    matrix_a = Matrix([[1, 2], [3, 4]])
    matrix_b = Matrix([[5, 6]])
    with pytest.raises(ValueError):
        to_augmented_matrix(matrix_a, matrix_b)


def test_inverse_simple():
    # Test case 1: Simple 2x2 matrix
    matrix = Matrix([[4, 7], [2, 6]])
    expected = Matrix([[0.6, -0.7], [-0.2, 0.4]])
    result = inverse(matrix)
    assert result.equals(expected)


def test_inverse_identity():
    # Test case 2: Identity matrix
    matrix = Matrix([[1, 0], [0, 1]])
    expected = Matrix([[1, 0], [0, 1]])
    result = inverse(matrix)
    assert result.equals(expected)


def test_inverse_raises_value_error():
    # Test case 3: Non-square matrix (should raise ValueError)
    matrix = Matrix([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError):
        inverse(matrix)


def test_inverse_singular_matrix():
    # Test case 4: Singular matrix (should raise ValueError or handle appropriately)
    matrix = Matrix([[1, 2], [2, 4]])
    with pytest.raises(ValueError):
        inverse(matrix)
