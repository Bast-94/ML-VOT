import numpy as np
import pytest
from scipy.optimize import linear_sum_assignment

from secretdir.KalmanFilter import KalmanFilter as KF2
from src.kalman_filter import KalmanFilter as KF1

NB_TESTS = 200
nb_params = 6
random_params = np.random.rand(NB_TESTS, nb_params)


@pytest.mark.parametrize("dt, u_x, u_y, std_acc, x_std_meas, y_std_meas",
                            [random_params[i] for i in range(NB_TESTS)])
def test_kalman_filter(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas):
    kf1 = KF1(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    kf2 = KF2(dt, u_x, u_y, std_acc, x_std_meas, y_std_meas)
    assert np.allclose(kf1.x, kf2.x)
    assert np.allclose(kf1.A, kf2.A)
    assert np.allclose(kf1.B, kf2.B)
    assert np.allclose(kf1.H, kf2.H)
    assert kf1.Q.shape == kf2.Q.shape
    assert np.allclose(kf1.Q, kf2.Q) 
    assert kf1.R.shape == kf2.R.shape
    assert np.allclose(kf1.R, kf2.R)
    assert np.allclose(kf1.P, kf2.P)
    assert np.allclose(kf1.predict(), kf2.predict())
    update_1 = kf1.update(np.array([1, 2]))
    update_2 = kf2.update(np.array([1, 2]))
    assert np.allclose(kf1.S, kf2.S)
    assert np.allclose(kf1.K, kf2.K)
    assert np.allclose(kf1.x, kf2.x)
    
    assert np.allclose(update_1, update_2) , "update_1: {}, update_2: {}".format(update_1, update_2)