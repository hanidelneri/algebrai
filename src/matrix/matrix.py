from typing import List


class Matrix:
    def __init__(self, data: List[List[float]] = [[]]) -> None:
        self.validate_dimensions(data)
        self.__data = data

    def validate_dimensions(self, data: List[List[float]]) -> bool:
        row_length = len(data[0])
        for row in data:
            if len(row) != row_length:
                raise ValueError("All rows must have the same length")

    def add_row(self, row: List[float]) -> None:
        row_length = len(self.__data[0])
        if len(row) != row_length:
            raise ValueError("Row must have the same length as the others")
        self.__data.append(row)

    def get_row(self, index: int) -> List[float]:
        return self.__data[index]

    def get_row_count(self) -> int:
        return len(self.__data)

    def replace_row(self, index: int, row: List[float]) -> None:
        row_length = len(self.__data[0])
        if len(row) != row_length:
            raise ValueError("Row must have the same length as the others")
        self.__data[index] = row

    def get_column(self, index: int) -> List[float]:
        return [row[index] for row in self.__data]
