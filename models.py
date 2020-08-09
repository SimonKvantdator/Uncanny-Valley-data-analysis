# Models for using in my_own_data_analysis.py.
# These models I have written down are all linear,
# but they needn't necessarily be.
import numpy as np

def predict_y_from_polynomial(theta, x):
	"""
	Returns predicted y-values given parameters theta & vector x.
	Uses a polynomial model.
	"""
	X_d = np.array([x**n for n in range(len(theta))]).T # design matrix
	y_pred = X_d @ theta
	return y_pred

def predict_y_from_cosines(theta, x):
	"""
	Returns predicted y-values given parameters theta & vector x.
	Uses a trigonometric sum model.
	"""
	T = max(x) - min(x)
	t = np.pi * (x - min(x)) / T

	X_d = np.array([np.cos(n * t) for n in range(0, len(theta))]).T # design matrix
	y_pred = X_d @ theta
	return y_pred

def relu(x):
	"""
	Rectified linear unit used in predict_y_from_piecewise_linear.
	"""
	return x * (x > 0.0)

def predict_y_from_piecewise_linear(theta, x):
	"""
	Returns predicted y-values given parameters theta & vector x.
	Uses a piecewise linear model. theta[0] is y-offset.
	"""
	T = max(x) - min(x)
	t = (x - min(x)) / T

	if len(theta) > 1:
		X_d = np.array([relu(t - n / len(theta[1:])) for n in range(0, len(theta[1:]))]).T # design matrix
		y_pred = X_d @ theta[1:] + theta[0]
	else:
		y_pred = theta[0] * np.ones_like(x)
	return y_pred