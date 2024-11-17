from .matrix import Matrix
from natural_number import least_common_multiple


def add(a: Matrix, b: Matrix) -> Matrix:
    if (a.get_row_count() != b.get_row_count()):
        raise ValueError("Matrices must have the same number of rows")
    if (a.get_column_count() != b.get_column_count()):
        raise ValueError("Matrices must have the same number of columns")

    result = Matrix()
    for i in range(a.get_row_count()):
        row = []
        for j in range(a.get_column_count()):
            row.append(a.get_row(i)[j] + b.get_row(i)[j])
        result.add_row(row)

    return result


def multiply(a: Matrix, b: Matrix) -> Matrix:
    if (a.get_column_count() != b.get_row_count()):
        raise ValueError(
            "The number of columns in the first matrix must be equal to the number of rows in the second matrix")

    result = Matrix()
    for i in range(a.get_row_count()):
        row = []
        for j in range(b.get_column_count()):
            sum = 0
            for k in range(a.get_column_count()):
                sum += a.get_row(i)[k] * b.get_column(j)[k]
            row.append(sum)
        result.add_row(row)

    return result


def transpose(a: Matrix) -> Matrix:
    result = Matrix()
    for i in range(a.get_column_count()):
        row = a.get_column(i)
        result.add_row(row)
    return result


def to_echelon_form(a: Matrix) -> Matrix:
    result = a.clone()
    row_count = a.get_row_count()

    for i in range(row_count):
        pivot_row = result.get_row(i)
        for j in range(i+1, row_count):
            row = result.get_row(j)
            lcm = least_common_multiple(pivot_row[i], row[i])
            if (lcm == 0):
                continue

            # Scale pivot_row and row by their respective factors
            scaled_pivot_row = [lcm / pivot_row[i] * x for x in pivot_row]
            scaled_row = [lcm / row[i] * x for x in row]

            # Subtract scaled pivot row from scaled row
            new_row = [scaled_row[k] - scaled_pivot_row[k]
                       for k in range(len(row))]

            result.replace_row(j, new_row)

    return result


def to_reduced_echelon_form(a: Matrix) -> Matrix:
    result = a.clone()
    for i in range(a.get_row_count()):
        pivot_row = result.get_row(i)
        if i >= a.get_column_count():
            continue
        pivot_element = pivot_row[i]
        if pivot_element == 0:
            continue
        pivot_row = [x / pivot_element for x in pivot_row]
        result.replace_row(i, pivot_row)

        for j in range(a.get_row_count()):
            if i == j:
                continue
            row = result.get_row(j)
            # Scale pivot_row and row by their respective factors
            factor = row[i]
            new_row = [
                row[k] - factor * pivot_row[k] for k in range(a.get_column_count())
            ]
            result.replace_row(j, new_row)

    return result


def to_augmented_matrix(a: Matrix, b: Matrix) -> Matrix:
    if a.get_row_count() != b.get_row_count():
        raise ValueError("Matrices must have the same number of rows")
    result = Matrix()
    for i in range(a.get_row_count()):
        row = a.get_row(i) + b.get_row(i)
        result.add_row(row)
    return result


def identity_matrix(size: int) -> Matrix:
    result = Matrix()
    for i in range(size):
        row = [0] * size
        row[i] = 1
        result.add_row(row)
    return result


def inverse(a: Matrix) -> Matrix:
    if a.get_row_count() != a.get_column_count():
        raise ValueError("Matrix must be square")

    augmented_matrix = to_augmented_matrix(a, identity_matrix(a.get_row_count()))
    echelon_form = to_echelon_form(augmented_matrix)

    # Extract the right half of the augmented matrix
    rows = echelon_form.get_rows()
    col_count = a.get_column_count()
    inverse_matrix_data = [row[col_count:] for row in rows]

    return Matrix(inverse_matrix_data)
