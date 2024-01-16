import numpy as np
class KalmanFilter():
    def __init__(self, dt:float,u_x:float,u_y:float,x_std_meas:float,y_std_meas:float):
        self.dt = dt
        self.u_x = u_x
        self.u_y = u_y
        self.std_acc_x = y_std_meas
        self.std_acc_y = x_std_meas
        self.time_state = 0
        self.x = np.array([[0, 0, 0, 0]]).T
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.B = np.array([[(self.dt**2)/2, 0],
                           [0, (self.dt**2)/2],
                            [self.dt, 0],
                            [0, self.dt]])
        self.H = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0]])
        self.Q = np.array([[(self.dt**4)/4, 0, (self.dt**3)/2, 0],
                            [0, (self.dt**4)/4, 0, (self.dt**3)/2],
                            [(self.dt**3)/2, 0, self.dt**2, 0],
                            [0, (self.dt**3)/2, 0, self.dt**2]]) * self.std_acc_x**2
        self.R = np.array([[self.std_acc_x**2, 0],
                            [0, self.std_acc_y**2]])
        self.P = np.eye(self.A.shape[1])
    
    def predict(self):
        self.x = np.dot(self.A, self.x_k) + np.dot(self.B, np.array([[self.u_x, self.u_y]]).T)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        self.time_state += self.dt
        return self.x
    
    def update(self, z):
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = (self.x + np.dot(K, (z - np.dot(self.H, self.x))), 2)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)