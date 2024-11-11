import pytest

from .arithmetics import (
    add,
    multiply,
    to_echelon_form,
    to_reduced_echelon_form,
    transpose,
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
