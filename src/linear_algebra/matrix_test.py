import pytest

from .matrix import Matrix


def test_matrix_raises_exception_different_row_size():
    with pytest.raises(ValueError) as e:
        Matrix([[1, 2], [3, 4, 5]])


def test_matrix_add_row_raises_exception_different_row_size():
    matrix = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError) as e:
        matrix.add_row([1, 2, 3])


def test_matrix_add_row():
    matrix = Matrix([[1, 2], [3, 4]])
    matrix.add_row([5, 6])
    assert matrix.get_row(2) == [5, 6]
    assert matrix.get_row(1) == [3, 4]


def test_matrix_get_row_count():
    matrix = Matrix([[1, 2], [3, 4]])
    assert matrix.get_row_count() == 2
    matrix.add_row([5, 6])
    assert matrix.get_row_count() == 3


def test_matrix_replace_row():
    matrix = Matrix([[1, 2], [3, 4]])
    matrix.replace_row(1, [5, 6])
    assert matrix.get_row(1) == [5, 6]
    assert matrix.get_row(0) == [1, 2]
    assert matrix.get_row_count() == 2


def test_matrix_replace_row_raises_exception_different_row_size():
    matrix = Matrix([[1, 2], [3, 4]])
    with pytest.raises(ValueError) as e:
        matrix.replace_row(1, [5, 6, 7])


def test_matrix_get_column():
    matrix = Matrix([[1, 2], [3, 4]])
    assert matrix.get_column(0) == [1, 3]
    assert matrix.get_column(1) == [2, 4]
    matrix.add_row([5, 6])
    assert matrix.get_column(0) == [1, 3, 5]
    assert matrix.get_column(1) == [2, 4, 6]


def test_matrix_equals_true():
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[1, 2], [3, 4]])
    assert matrix1.equals(matrix2) is True


def test_matrix_equals_false_different_dimensions():
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[1, 2, 3], [4, 5, 6]])
    assert matrix1.equals(matrix2) is False


def test_matrix_equals_false_different_values():
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[1, 2], [4, 3]])
    assert matrix1.equals(matrix2) is False


def test_matrix_equals_false_different_row_count():
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[1, 2]])
    assert matrix1.equals(matrix2) is False


def test_matrix_equals_false_different_column_count():
    matrix1 = Matrix([[1, 2], [3, 4]])
    matrix2 = Matrix([[1], [3]])
    assert matrix1.equals(matrix2) is False
