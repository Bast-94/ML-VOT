import numpy as np

from src.kalman_matrices import get_A, get_B, get_H, get_P, get_Q, get_R


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

        self.time_state = 0
        self.x = np.array([[0, 0, 0, 0]]).T
        self.A = get_A(self.dt)
        self.B = get_B(self.dt)
        self.H = get_H()
        self.Q = get_Q(self.dt, std_acc)
        self.R = get_R(x_std_meas, y_std_meas)
        self.P = get_P()

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(
            self.B, np.array([[self.u_x, self.u_y]]).T
        )

        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        self.time_state += self.dt
        return self.x

    def update(self, z):
        z = z.reshape(-1, 1) if z.shape == (2,) else z

        S = self.R + self.H @ self.P @ self.H.T
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)

        self.P = (np.eye(self.P.shape[0]) - K @ self.H) @ self.P
        self.S = S
        self.K = K
        return self.x.T

    @classmethod
    def tracking_kalman_filter(cls):
        return cls(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
