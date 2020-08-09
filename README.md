## What is this project?

This project is a recreation of part of the data analysis done in Mathur and Reichling's article, [Navigating a social world with robot partners: A quantitative cartography of the Uncanny Valley](https://www.sciencedirect.com/science/article/pii/S0010027715300640?via%3Dihub). Specifically, I wanted to recreate their Fig 3A. My main idea was to question the curve fit. I don't think the data is sufficient to conclude the shape of the curve, and the evidence for an uncanny valley is relatively weak.

I used the emcee python implementation of Markov Chain Monte Carlo sampling to perform a Bayesian data analysis of the data Mathur and Reichling presents in Fig 3A. I produced some of my own curve fits, placed in the folders *some piecewise linear fits*, *some polynomial fits*, and *some trigonometric sum fits*, using linear models with different basis functions. Using a goodness-of-fit metric (Bayesian information criterion) that punishes over-complicated models, I could see that the "uncanny valley" as produced by a third degree polynomial curve fit did not perform better than a zeroth or first degree polynomial curve fit where there were no valley.

The data from Mathur and Reichling's article is in *2015-06-21_expt_1B_1C_face_means.csv*, and was fetched from [here](https://osf.io/3rjnk/).


## How to use the code?

The main script is *my_own_data_analysis.py*, it runs the MCMC and produces the figures. The script also stores the curve fit results and configurations in *configs_and_results.json*.

In *config.json*, you can tweak the configurations of the curve fit.
To reproduce the curve fit corresponding to Mathur and Reichling's Fig 3A, for example, set 
```
"model" : "predict_y_from_polynomial",
"nbr_parameters" : 4,
"nbr_warmup" : 1000,
"nbr_samples" : 10000,
"nbr_walkers" : 32
```
The *model* parameter should be the name of one of the functions in *models.py*.

In *models.py*, I store the different prediction models being used in *my_own_data_analysis.py*. Here you can just add your own model if you want, it needn't even be linear like mine are.