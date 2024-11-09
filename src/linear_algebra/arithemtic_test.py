import pytest

from .arithmetics import add
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
