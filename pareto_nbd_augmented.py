import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
# from statsmodels import api
from scipy import stats
from scipy.optimize import minimize
from lifetimes.fitters.pareto_nbd_fitter import ParetoNBDFitter
from lifetimes.fitters.gamma_gamma_fitter import GammaGammaFitter
from sklearn.ensemble import RandomForestRegressor
from scipy.special import gamma


customer_rfm = pd.read_csv('C:\\Users\\il.pugin\\Downloads\\customer_rfm\\customer_rfm.csv')
# print(customer_rfm.head(10))
print(customer_rfm.columns)

def gamma_gamma_likelihood_i(p, q, y, xi, mi):
    likelihood_i = gamma(p*xi+q)/gamma(p*xi)/gamma(q)*(y**q)*(mi**(p*xi-1))*(xi**(p*xi))/((y+mi*xi)**(p*xi+q))
    return  likelihood_i

def gamma_gamma_likelihood(p,q,y,df):
    col = pd.Series(df.apply(lambda row: gamma_gamma_likelihood_i(p,q,y,row[0],row[1]), axis=1))
    return col.product().iloc[0]

def nbd_likelihood_i(p, q, y, xi, mi):
    likelihood_i = gamma(p*xi+q)/gamma(p*xi)/gamma(q)*(y**q)*(mi**(p*xi-1))*(xi**(p*xi))/((y+mi*xi)**(p*xi+q))
    return  likelihood_i

def nbd_likelihood(p,q,y,df):
    col = pd.Series(df.apply(lambda row: gamma_gamma_likelihood_i(p,q,y,row[0],row[1]), axis=1))
    return col.product().iloc[0]



m_predicted = ((q-1)/(p*xi+q-1))*lam/(q-1) + ((p*xi)/(p*xi+q-1))*mi

xi_predicted = xi+(gamma(r+xi)*(a**r)*(b**s))