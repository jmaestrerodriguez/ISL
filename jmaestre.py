"""
Implementations of mathematical models for learning purpose.
"""

__all__ = ['MathModel','SimpleLinearRegressor','generate_dataset','mean']
__version__ = '0.1'
__author__ = 'Cardinal Biggles'

from random import normalvariate
from random import uniform
from math import sqrt

import seaborn as sns
from matplotlib import pyplot as plt

sns.set() # Setting seaborn as default style even if use only matplotlib
sns.set(rc={'figure.figsize':(20,15)})

class MathModel:
    """
    Generic class defining a mathematical model object.
    """
    def __init__(self):
        self.dataset_parameters = dict()
        self.parameters = dict()
        self.statisticals = dict()

class SimpleLinearRegressor(MathModel):
    """
    Simple Linear Regressor object: n = 1. 
    """
        
    def fit(self, X, Y):
        """
        Fits Y to X and calculates some error metrics and statisticals.
        
        INPUT: 
        X: list of predictor values
        Y: list of response values
        
        OUTPUT:
        """
        n = len(X)
        p = 1
        
        X_mean, Y_mean = mean(X), mean(Y)
        
        slope_numerator   = sum([(x - X_mean) * (y - Y_mean) for x, y in zip(X,Y)])
        slope_denominator = sum([(x - X_mean)**2 for x in X])
        
        slope = slope_numerator / slope_denominator
        intercept = Y_mean - slope * X_mean
        
        RSS = sum([(y - (intercept + slope * x))**2 for x, y in zip(X,Y)])
        RSE = sqrt(RSS / (n - 2))
        
        slope_SE = sqrt((RSE**2) * ((1/n) + (X_mean**2 / slope_denominator)))
        intercept_SE = sqrt((RSE**2) / slope_denominator)
        
        slope_CI = (slope - 2 * slope_SE, slope + 2 * slope_SE)
        intercept_CI = (intercept - 2 * intercept_SE, intercept + 2 * intercept_SE)
        
        try:
            t_statistic = (slope - 0)/ slope_SE # number of SE that slope is far from zero.
        except:
            t_statistic = None # to prevent a crash when slope_SE = 0
            
        p_value = None
            
        # Assign class attributes if everything went OK.
        
        self.dataset_parameters['n'] = n
        self.dataset_parameters['p'] = p
        
        self.parameters['slope'] = slope
        self.parameters['intercept'] = intercept
        
        self.statisticals['RSS'] = RSS
        self.statisticals['RSE'] = RSE
        self.statisticals['slope_SE'] = slope_SE
        self.statisticals['intercept_SE'] = intercept_SE
        self.statisticals['slope_CI'] = slope_CI
        self.statisticals['intercept_CI'] = intercept_CI
        self.statisticals['t_statistic'] = t_statistic
    
    def predict(self, X):
        return [self.parameters['intercept'] + self.parameters['slope'] * x for x in X]
    
    def plot(self, X, Y):
        """
        Plots the data, the line of best fit and the residuals.
        """
        
        Y_pred = self.predict(X)
        residuals =  residuals = [y - y_pred for y, y_pred in zip(Y, Y_pred)]
        
        fig, axes = plt.subplots(2, 1)
        
        fig.suptitle('Linear regression summary plots')
        axes[0].set_title('Data and line of best fit')
        axes[1].set_title('Residuals plot')
        
        sns.scatterplot(ax = axes[0], x = X, y = Y)
        sns.lineplot(ax = axes[0], x = X, y = Y_pred)
        sns.scatterplot(ax = axes[1], x = X, y = residuals)
    

def generate_dataset(n = 30, x_min = -1, x_max = 1, slope = 1, intercept = 1, e_mean = 0, e_std = 0.05):
    """
    Generates a dataset according to the input parameters
    
    Keyword Arguments:
    n:            number of observations
    x_min, x_max: limits of the preditor's range (x_min <= x <= x_max)
    slope:        slope of the true linear model
    intercept:    intercept of the true linear model
    e_mean:       error term mean
    e_std:        error term standard deviation
    
    Output:
    X: predictor values
    Y: response values
    """
    
    X = tuple(uniform(x_min,x_max) for _ in range(n))
    Y = tuple(intercept + slope * x + normalvariate(e_mean,e_std) for x in X)
    
    return X, Y
    
def mean(L):
    """Returns the mean of the elements of a list."""
    return sum(L)/len(L)