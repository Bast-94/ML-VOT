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
        self.x_k = np.array([[0, 0, 0, 0]]).T
        self.A = get_A(self.dt)
        self.B = get_B(self.dt)
        self.H = get_H()
        self.Q = get_Q(self.dt, std_acc)
        self.R = get_R(x_std_meas, y_std_meas)
        self.P_k = get_P()

    def predict(self):
        self.x_k = np.dot(self.A, self.x_k) + np.dot(
            self.B, np.array([[self.u_x, self.u_y]]).T
        )

        self.P_k = np.dot(np.dot(self.A, self.P_k), self.A.T) + self.Q
        self.time_state += self.dt
        return self.x_k

    def update(self, z):
        z = z.reshape(-1, 1) if z.shape == (2,) else z

        S_k = self.R + self.H @ self.P_k @ self.H.T
        K_k = self.P_k @ self.H.T @ np.linalg.inv(S_k)
        self.x_k = self.x_k + K_k @ (z - self.H @ self.x_k)

        self.P_k = (np.eye(self.P_k.shape[0]) - K_k @ self.H) @ self.P_k
        self.S = S_k
        self.K = K_k
        return self.x_k.T

    @classmethod
    def tracking_kalman_filter(cls):
        return cls(dt=0.1, u_x=1, u_y=1, std_acc=1, x_std_meas=0.1, y_std_meas=0.1)
