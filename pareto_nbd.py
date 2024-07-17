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
from sklearn.metrics import mean_squared_error, r2_score
import math
from math import sqrt

from lifetimes.plotting import plot_frequency_recency_matrix
from lifetimes.plotting import plot_probability_alive_matrix
from lifetimes.plotting import plot_period_transactions
from lifetimes.utils import calibration_and_holdout_data

customer_rfm = pd.read_csv('C:\\Users\\il.pugin\\Downloads\\customer_rfm\\customer_rfm.csv')
# print(customer_rfm.head(10))
# print(customer_rfm.columns)

# fit a model
# pareto_model = ParetoNBDFitter(penalizer_coef=0.0)
# pareto_model.fit(customer_rfm['frequency'], customer_rfm['recency'], customer_rfm['t'])


def get_model(data, penalizer_val, time):
    pareto_result = data.copy()

    pareto_model = ParetoNBDFitter(penalizer_coef=penalizer_val)
    pareto_model.fit(pareto_result["frequency"], pareto_result["recency"], pareto_result["T"])

    # calculating the predicted_purchases

    t = time

    pareto_result["predicted_purchases"] = pareto_model.conditional_expected_number_of_purchases_up_to_time(t,
                                                                                                            pareto_result[
                                                                                                                "frequency"],
                                                                                                            pareto_result[
                                                                                                                "recency"],
                                                                                                            pareto_result[
                                                                                                                "T"])

    pareto_result["Actual_Purchases"] = pareto_result["frequency"] / pareto_result["recency"] * time

    # filling the null values
    pareto_result["Actual_Purchases"].fillna(0, inplace=True)

    # calculating the error
    pareto_result["Prediction_Error"] = pareto_result["Actual_Purchases"] - pareto_result["predicted_purchases"]

    # calcuating the purchase prediction error
    pareto_mse_purchase = mean_squared_error(pareto_result["Actual_Purchases"], pareto_result["predicted_purchases"])
    pareto_r2_purchase = r2_score(pareto_result["Actual_Purchases"], pareto_result["predicted_purchases"])
    pareto_rmse_purchase = sqrt(
        mean_squared_error(pareto_result["Actual_Purchases"], pareto_result["predicted_purchases"]))
    pareto_avg_error_purchase = pareto_result["Prediction_Error"].mean()

    # printing the purchase prediction error
    print("Predicted Purchase Mean Squared Error: %s" % (pareto_mse_purchase))
    print("Predicted Purchase R2 Score: %s" % (pareto_r2_purchase))
    print("Predicted Purchase Root Mean Squared Error: %s" % (pareto_rmse_purchase))
    print("Predicted Purchase Average Purchases Error: %s" % (pareto_avg_error_purchase))

    # plotting the prediction v/s actual purchase plot
    plt.figure(figsize=(6, 4))
    plt.errorbar(pareto_result["Actual_Purchases"], pareto_result["predicted_purchases"],
                 yerr=pareto_result["Prediction_Error"], fmt='o', color='darkseagreen',
                 ecolor='grey', elinewidth=1.5, capsize=0, alpha=0.2);
    plt.title("Prediction v/s Actual")

pareto_result = customer_rfm.copy()

plot_period_transactions(pareto_model)