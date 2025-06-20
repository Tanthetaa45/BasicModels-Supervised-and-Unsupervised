from typing import List, Tuple
from typing import Callable
import math

# Define Matrix as a type alias for a list of lists of floats (or ints if preferred)
Matrix = List[List[float]]

def shape(A: Matrix) -> Tuple[int, int]:
    num_rows = len(A)
    num_cols = len(A[0]) if A else 0
    return num_rows, num_cols

# Test the function
matrix = [[3, 5, 6], [7, 8, 9]]
result = shape(matrix)
print(result)  # This will print: (2, 3)


def make_matrix(num_rows:int,
                num_cols:int,
                entry_fn:Callable[[int,int],float])->Matrix:
    return [[entry_fn(i,j)
             for j in range(num_cols)]
             for i in range(num_rows)]
def identity_matrix_entry(i: int, j: int) -> float:
    return 1.0 if i == j else 0.0

# Creating a 3x3 identity matrix
identity_matrix = make_matrix(10, 10, identity_matrix_entry)
print("Identity Matrix:")
for row in identity_matrix:
    print(row)

#CREATES A MATRIX WHICH TAKES INPUT FROM THE USER

def input_matrix(num_rows: int, num_cols: int) -> Matrix:
    print(f"Enter the elements of the matrix (rows: {num_rows}, columns: {num_cols}):")
    def entry_fn(i: int, j: int) -> float:
        return float(input(f"Element [{i}][{j}]: "))
    
    return make_matrix(num_rows, num_cols, entry_fn)

# Main program to create and print a matrix from keyboard input
if __name__ == "__main__":
    num_rows = int(input("Enter the number of rows: "))
    num_cols = int(input("Enter the number of columns: "))
    
    matrix = input_matrix(num_rows, num_cols)
    
    print("The entered matrix is:")
    for row in matrix:
        print(row)
    
    matrix_shape = shape(matrix)
    print("Shape of the entered matrix:", matrix_shape)


    #ALL THIS BULLSHIT IS AVOIDED BY SIMPLY USING NumPy 