import numpy as np


def get_Q(dt:float, std_acc:float):
    return np.array(
        [
            [(dt**4) / 4, 0, (dt**3) / 2, 0],
            [0, (dt**4) / 4, 0, (dt**3) / 2],
            [(dt**3) / 2, 0, dt**2, 0],
            [0, (dt**3) / 2, 0, dt**2],
        ]
    ) * std_acc**2

class KalmanFilter:
    def __init__(
        self,
        dt: float,
        u_x: float,
        u_y: float,
        std_acc: float,
        x_std_meas: float,
        y_std_meas: float,
    ):
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc_x = x_std_meas
        self.std_acc_y = y_std_meas
        self.std_acc = std_acc
        self.time_state = 0
        self.x = np.array([[0, 0, 0, 0]]).T
        self.A = np.array(
            [[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]]
        )
        self.B = np.array(
            [
                [(self.dt**2) / 2, 0],
                [0, (self.dt**2) / 2],
                [self.dt, 0],
                [0, self.dt],
            ]
        )
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.Q = get_Q(self.dt, self.std_acc)
        self.R = np.array([[self.std_acc_x**2, 0], [0, self.std_acc_y**2]])
        self.P = np.eye(self.A.shape[0])

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(
            self.B, np.array([[self.u_x, self.u_y]]).T
        )
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        self.time_state += self.dt
        return self.x

    def update(self, z):
        if z.shape == (2,):
            z = z.reshape(-1, 1)

        S = self.R + self.H @ self.P @ self.H.T
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        self.S = S
        self.K = K
        return self.x.T
