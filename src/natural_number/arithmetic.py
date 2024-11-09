def greatest_common_divisor(a: int, b: int) -> int:
    if b == 0:
        return a

    return greatest_common_divisor(b, a % b)


def least_common_multiple(a: int, b: int) -> int:
    gcd = greatest_common_divisor(a, b)
    if gcd == 0:
        return 0
    return a*b//gcd
