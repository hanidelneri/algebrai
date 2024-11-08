import pytest
from arithmetic import greatest_common_divisor, least_common_multiple


@pytest.mark.parametrize("a, b, expected", [(10, 5, 5), (5, 10, 5), (10, 0, 10), (2, 3, 1)])
def test_greatest_common_divisor(a: int, b: int, expected: int):
    assert greatest_common_divisor(a, b) == expected


@pytest.mark.parametrize("a, b, expected", [(10, 5, 10), (5, 10, 10), (10, 0, 0), (2, 3, 6), (4, 6, 12), (-3, -6, -6)])
def test_least_common_multiple(a: int, b: int, expected: int):
    assert least_common_multiple(a, b) == expected
