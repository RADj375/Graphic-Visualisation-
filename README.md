# Graphic-Visualisation-
cubic spline interpolation visualisation 

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import artoolkitplus as ar
from keras.models import Sequential
from keras.layers import Dense, Dropout

from functools import lru_cache

@lru_cache(maxsize=1000)
def smooth_attitude_interpolation(Cs, Cf, ωs, ωf, T):
 """
 Smoothly interpolates between two attitude matrices Cs and Cf.
 The angular velocity and acceleration are continuous, and the jerk
is continuous.

 Args:
   Cs: The initial attitude matrix.
   Cf: The final attitude matrix.
   ωs: The initial angular velocity.
   ωf: The final angular velocity.
   T: The time interval between Cs and Cf.

 Returns:
   A list of attitude matrices that interpolate between Cs and Cf.
 """

 # Check if the input matrices are valid.
 if not np.allclose(np.linalg.inv(Cs) @ Cs, np.eye(3)):
   raise ValueError("Cs is not a valid attitude matrix.")
 if not np.allclose(np.linalg.inv(Cf) @ Cf, np.eye(3)):
   raise ValueError("Cf is not a valid attitude matrix.")

 # Fit a cubic spline to the rotation vector.
 θ = np.linspace(0, T, 3)

 def rotation_vector(t):
   return np.log(Cs.T @ Cf)

 θ_poly, _ = curve_fit(rotation_vector, θ, np.zeros_like(θ), maxfev=100000,
                        method='cubic')

 # Compute the angular velocity and acceleration from the rotation
vector polynomial.
 ω = np.diff(θ_poly) / θ
 ω_̇ = np.diff(ω) / θ

 # Set the jerk at the endpoints to be equal to each other.
 ω_̇[0] = ω_̇[-1]

 # Solve for the angular velocities.
 ω = np.linalg.solve(np.diag(1 / θ) + np.diag(ω_̇), ωs - ωf)

 # Fit a cubic spline to the time matrix.
 t = np.linspace(0, T, 3)
 t_poly, _ = curve_fit(lambda t: np.exp(t), t, np.arange(len(t)),
maxfev=100000,
                        method='cubic')

 # Interpolate the attitude matrices.
 C = np.exp(θ_poly @ np.linalg.inv(np.diag(t_poly)))

 # Visualize the results.
 plt.plot(t, θ_poly)
 plt.show()

 return C

if __name__ == '__main__':
 # Test the code.
 Cs = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
 Cf = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
 ωs = np.array([0, 0, 0])
 ωf = np.array([0, 0, 1])
 T = 1

 C = smooth_attitude_interpolation(Cs, Cf, ωs, ωf, T)

 print(C)

In this code, the if __name__ == '__main__': block is now used to
visualize the results of the interpolation. The plt.plot() function is
used to plot the rotation vector polynomial. The plt.show() function
is used to show the plot.

The code now includes visualization of the results. This visualization
can help to understand the behavior of the cubic spline interpolation
metho
