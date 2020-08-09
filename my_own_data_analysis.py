import numpy as np
from scipy.optimize import minimize

import os
import re # regular expressions
import datetime

# for reading csv & json
import pandas as pd
import json
with open('config.json', 'r') as config_file: # load model & figure configurations from file
	configs = json.loads(config_file.read())
	config_file.close()

# for plotting nice plots
import matplotlib as mpl
import matplotlib.pyplot as plt
import corner # for corner plots
fsize = 16
font = {'size': fsize}
mpl.rc('font', **font)
mpl.rc('xtick', labelsize=fsize)
mpl.rc('ytick', labelsize=fsize)
mpl.rc('text', usetex=configs['use_tex'])

# for MCMC
from emcee import EnsembleSampler as ensembleSampler
from multiprocessing import Pool

### define model ###
exec('from models import ' + configs['model'] + ' as predict_y') # a bit bodgy, but it works


### load experiment data ###
# this is the data shown in the uncanny valley wikipedia page
filename = "2015-06-21_expt_1B_1C_face_means.csv"
data = pd.read_csv(filename)
x_data = np.array(data["composite"])
y_data = np.array(data["like.mean"])
sigmas = np.array(data["like.sd"])


### define pdf:s ###
def log_likelihood(theta):
	"""
	Returns the log of the probability density of x_data & y_data being observed from
	a linear model y = theta[0] * x**0 + theta[1] * x**1 + ... + errors where errors
	are normally distributed random variables centered at 0 with variances sigmas**2
	"""
	y_pred = predict_y(theta, x_data)
	tmp_sum = (np.log(2.0 * np.pi * sigmas**2) + (y_data - y_pred)**2 / sigmas**2).sum()
	return -(1.0 / 2.0) * tmp_sum

def log_flat_prior(theta):
	"""
	Returns log of a flat pdf
	"""
	return 0.0

def log_gauss_prior(theta):
	"""
	Returns log of a gaussian pdf
	"""
	width = 5.0
	return -((theta**2) / (2.0 * width**2)).sum() - (len(theta)) / 2.0 * np.log(2.0 * np.pi * width**2)

def log_posterior(theta):
	"""
	Returns log of posterior pdf using a flat prior pdf
	"""
	return log_flat_prior(theta) + log_likelihood(theta) # choose your prior

### define MCMC sampler ###
def my_sampling(dim, log_posterior, nbr_walkers=configs['nbr_walkers'], nbr_warmup=configs['nbr_warmup'], nbr_samples=configs['nbr_samples']):
	"""
	Returns sample chain from MCMC given dimension dim of problem and a logarithmic pdf
	log_posterior.
	"""
	initial_positions = 0.01 * np.random.rand(nbr_walkers, dim)
	sampler = ensembleSampler(nbr_walkers, dim, log_posterior, pool=Pool())

	pos, tr, pr = sampler.run_mcmc(initial_positions, nbr_warmup)
	sampler.reset()
	sampler.run_mcmc(pos, nbr_samples);
		
	return sampler.flatchain
    
### run MCMC sampler ###
samples = my_sampling(configs['nbr_parameters'], log_posterior)


# make histogram corner plot of MCMC result
if configs['show_figs'] or configs['save_figs']:
	tex_labels = [r'$a_{' + str(i) + r'}$' for i in range(configs['nbr_parameters'])]
	fig1 = corner.corner(samples,
		quantiles=[0.16, 0.5, 0.84],
		bins = configs['nbr_walkers'],
		show_titles=True,
		title_fmt='.4f',
		labels=tex_labels)


### fit curve to data ###
x_span = np.linspace(min(x_data), max(x_data), num=400)

# make predictions from each sample
y = np.zeros((len(samples[:,0]), len(x_span)))
for i in range(len(samples[:,0])):
	y[i,:] = predict_y(samples[i,:], x_span)
y_standard_deviation = np.sqrt(y.var(axis=0))

# mean estimate of theta
theta_mean = np.mean(samples, axis=0)
y_mean = predict_y(theta_mean, x_span)

# max likelihood estimate of theta
def negative_log_likelihood(theta):
	return -log_likelihood(theta)
optimizer_result = minimize(negative_log_likelihood, theta_mean, method='Nelder-Mead')
theta_max_likelihood = optimizer_result['x']
y_max_likelihood = predict_y(theta_max_likelihood, x_span)

# evaluate goodness of fit using Bayesian information criterion (BIC)
goodness_of_fit = configs['nbr_parameters'] * np.log(len(x_data)) - 2.0 * log_likelihood(theta_mean)

# save results
curve_fit_results = {} # initialize dict
curve_fit_results['mean_theta'] = theta_mean.tolist()
curve_fit_results['theta_max_likelihood'] = theta_max_likelihood.tolist()
curve_fit_results['goodness_of_fit'] = goodness_of_fit


### plot fitted curve ###
if configs['show_figs'] or configs['save_figs']:
	nbr_standard_deviations = 1 # nbr_standard_deviations = 1 gives 68% DOB interval
	fig2, ax = plt.subplots(figsize=(6, 6))
	ax.plot(x_span, y_mean, linestyle='-', color='navy', label="mean prediction")
	ax.fill_between(x_span, y_mean - nbr_standard_deviations * y_standard_deviation,
		y_mean + nbr_standard_deviations * y_standard_deviation,
		facecolor='teal',
		alpha=0.3,
		label=r'$\pm$' + str(nbr_standard_deviations) + r'$\sigma$ interval')
	ax.scatter(x_data,
		y_data,
		marker='o',
		color='black',
		label=r'data')
	ax.set_xlabel(r'Mechano-humanness score')
	ax.set_ylabel(r'Likeability')
	ax.legend()
	plt.tight_layout()


### save data on curve fit ###
# remove comments from config file
commentless_configs = {} # initialize
for tmp in configs:
	if re.match('__', tmp) is None:
		commentless_configs[str(tmp)] = configs[str(tmp)]

# create folder named with the current time to save figures and data in there
folder_name = str(datetime.datetime.now())
os.mkdir(folder_name)

# save configurations and curve fitting results
curve_fit_configs_and_results = {} # initialize
curve_fit_configs_and_results['configs'] = commentless_configs
curve_fit_configs_and_results['results'] = curve_fit_results
with open(folder_name + '/curve_fit_data.json', 'w') as curve_fit_configs_and_results_file:
	json.dump(curve_fit_configs_and_results, curve_fit_configs_and_results_file, indent=4)
	curve_fit_configs_and_results_file.close()

if configs['save_figs']:
	# save figures
	fig1.savefig(folder_name + '/' + 'mcmc_histogram_corner_plot.pdf')
	fig2.savefig(folder_name + '/' + 'curve_fit.pdf')

# show all figures
if configs['show_figs']:
	plt.show()