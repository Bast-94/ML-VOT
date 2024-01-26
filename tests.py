import numpy as np
from scipy.optimize import linear_sum_assignment

if __name__ == "__main__":
    cost = np.array([[4, 1, 3], [2, 0, -10], [3, 2, 2]])
    row_ind, col_ind = linear_sum_assignment(cost)
    print(row_ind)
    print(col_ind)
