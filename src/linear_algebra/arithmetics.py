from .matrix import Matrix


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
