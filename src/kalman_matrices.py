import numpy as np
SQUARE_MATRIX_SHAPE = (4, 4)
def get_Q(dt:float, std_acc:float):
    return np.array(
        [
            [(dt**4) / 4, 0, (dt**3) / 2, 0],
            [0, (dt**4) / 4, 0, (dt**3) / 2],
            [(dt**3) / 2, 0, dt**2, 0],
            [0, (dt**3) / 2, 0, dt**2],
        ]
    ) * std_acc**2

def get_B(dt:float):
    return np.array(
        [
            [(dt**2) / 2, 0],
            [0, (dt**2) / 2],
            [dt, 0],
            [0, dt],
        ]
    )
def get_A(dt:float):
    return np.array(
        [
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

def get_H():
    return np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
def get_R(x_std_meas:float, y_std_meas:float):
    return np.array([[x_std_meas**2, 0], [0, y_std_meas**2]])

def get_P():
    return np.eye(SQUARE_MATRIX_SHAPE[0])