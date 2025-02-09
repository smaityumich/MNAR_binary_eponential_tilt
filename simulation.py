import numpy as np
from exp_grad import exponentiated_gradient, ext_x, profile_likelihood_opt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns

n, d = 200, 2


def gen_data(n = 200, d = 3, var_m = 1.):
    delta = 1
    v = np.array([-delta, 2], dtype = "float").reshape((1, -1))
    str_y = np.random.binomial(1, 0.4, size = (n, 1))
    str_x = np.random.normal(size = (n, d)) * (str_y * var_m + (1 - str_y)) * 1. + (2 * str_y - 1) * v
    str_y = str_y.reshape((-1, ))
    
    v[0, 0] = delta
    ttr_y = np.random.binomial(1, 0.6, size = (n, 1))
    ttr_x = np.random.normal(size = (n, d)) * (ttr_y  * var_m + (1 - ttr_y)) * 1. + (2 * ttr_y - 1) * v
    ttr_y = ttr_y.reshape((-1, ))
    
    return str_x, str_y, ttr_x, ttr_y

def gen_data_missp(n = 200, d = 3, var_m = 1.):
    delta = 1
    v = np.array([-delta, 2], dtype = "float").reshape((1, -1))
    str_y = np.random.binomial(1, 0.4, size = (n, 1))
    str_x = np.random.normal(size = (n, d)) * (str_y * var_m + (1 - str_y)) * 1. + (2 * str_y - 1) * v
    str_y = str_y.reshape((-1, ))
    
    v[0, 0] = delta
    ttr_y = np.random.binomial(1, 0.6, size = (n, 1))
    ttr_x = np.random.normal(size = (n, d)) * (ttr_y + (1 - ttr_y)  * var_m) * 1. + (2 * ttr_y - 1) * v
    ttr_y = ttr_y.reshape((-1, ))
    
    return str_x, str_y, ttr_x, ttr_y





np.random.seed(13847)
n_taus = 5    
taus = np.logspace(-.6, .2, num= n_taus)
ITER = 40
IPWS = np.zeros(shape = (n_taus, ITER))
IMPS = np.zeros(shape = (n_taus, ITER))
DRS = np.zeros(shape = (n_taus, ITER))

all_measures = []
    
for j in range(ITER):
    for i, tau in enumerate(taus):
        
        print("tau: ", tau, "...")
        
        str_x, str_y, ttr_x, ttr_y = gen_data(n = 200, d = d, var_m = tau)
        sts_x, sts_y, tts_x, tts_y = gen_data(n = 200, d = d, var_m = tau)
        
        lr = RandomForestClassifier().fit(sts_x, sts_y)
        str_p = lr.predict_proba(str_x)
        ttr_p = lr.predict_proba(ttr_x)
        
        beta = exponentiated_gradient(str_x, str_y, str_p, ttr_x, ttr_p, 
                                                eps=1e-3, tol=1e-3, B=10, lr=4e-3, 
                                                max_iter=5000, verbose=0, reg = 1e-3)
        
        # beta = profile_likelihood_opt(str_x, str_y, str_p, ttr_x, ttr_p, 
        #                                          tol=3e-4, lr=1e-3, max_iter=4000, verbose=0, reg = 1e-3)
        
        str_x_ext = ext_x(str_x)
        ttr_x_ext = ext_x(ttr_x)
        
        str_w = np.exp(str_x_ext @ beta)
        ttr_w = np.exp(ttr_x_ext @ beta)
        str_w_sel = str_w[np.arange(str_y.shape[0]), str_y]
        
        IPW = ((str_y * str_w_sel).mean() + str_y.mean())/2
        # print("IPW", IPW)
        
        r_1 = 0.5
        eta_r = 1/(1 + (1 - r_1) * str_w_sel/r_1)
        
        alpha = str_x_ext @ beta
        log_or_str = alpha[:, 1] - alpha[:, 0]
        m0_str = (np.exp(log_or_str) * str_p[:, 1]) / (np.exp(log_or_str) * str_p[:, 1] + str_p[:, 0])
        
        
        alpha = ttr_x_ext @ beta
        log_or_ttr = alpha[:, 1] - alpha[:, 0]
        m0_ttr = np.exp(log_or_ttr) * ttr_p[:, 1] / (np.exp(log_or_ttr) * ttr_p[:, 1] + ttr_p[:, 0])
        
        imputed = (m0_str.mean() + m0_ttr.mean())/2
        
        DR = ((str_y - m0_str) * (1 + str_w_sel)).mean()/2 + imputed
        print("true value", 0.5, "IPW", IPW, "imputed", imputed, "DR", DR, "\n")
        
        all_measures.append({
            "estimator": "IPW", 
            "sigma_1": tau.round(3), 
            "estimate": IPW,
            "parameter": "overall_mean",
            "true_value": 0.5,
            })
        all_measures.append({
            "estimator": "DR", 
            "sigma_1": tau.round(3), 
            "estimate": DR,
            "parameter": "overall_mean",
            "true_value": 0.5,
            })
        all_measures.append({
            "estimator": "OR", 
            "sigma_1": tau.round(3), 
            "estimate": imputed,
            "parameter": "overall_mean",
            "true_value": 0.5,
            })
        
        
        ## Target mean

        IPW = (str_y * str_w_sel).mean()
        # print("IPW", IPW)
        
        r_1 = 0.5
        eta_r = 1/(1 + (1 - r_1) * str_w_sel/r_1)
        
        alpha = str_x_ext @ beta
        log_or_str = alpha[:, 1] - alpha[:, 0]
        m0_str = (np.exp(log_or_str) * str_p[:, 1]) / (np.exp(log_or_str) * str_p[:, 1] + str_p[:, 0])
        
        
        alpha = ttr_x_ext @ beta
        log_or_ttr = alpha[:, 1] - alpha[:, 0]
        m0_ttr = np.exp(log_or_ttr) * ttr_p[:, 1] / (np.exp(log_or_ttr) * ttr_p[:, 1] + ttr_p[:, 0])
        
        imputed =  m0_ttr.mean()
        
        DR = ((str_y - m0_str) *  str_w_sel).mean() + imputed
        print("true value", 0.6, "IPW", IPW, "imputed", imputed, "DR", DR, "\n")
        
        all_measures.append({
            "estimator": "IPW", 
            "sigma_1": tau.round(3), 
            "estimate": IPW,
            "parameter": "target_mean",
            "true_value": 0.6,
            })
        all_measures.append({
            "estimator": "DR", 
            "sigma_1": tau.round(3), 
            "estimate": DR,
            "parameter": "target_mean",
            "true_value": 0.6,
            })
        all_measures.append({
            "estimator": "OR", 
            "sigma_1": tau.round(3), 
            "estimate": imputed,
            "parameter": "target_mean",
            "true_value": 0.6,
            })
data = pd.DataFrame(all_measures)
data.to_csv("plots/results_el_mis.csv", index_label="False")   

data = pd.read_csv("plots/results.csv")     
data = data.replace("IPW", "IW")


fig, axs = plt.subplots(1, 2, figsize = (10, 3))  

data_subset = data.loc[data["parameter"] == "overall_mean"]
data_subset = data_subset.loc[data_subset["estimator"] != "OR"]

ax = axs[0] 
sns.boxplot(data=data_subset, x="sigma_1", y="estimate", hue="estimator", ax = ax)
ax.set_xlabel(r"$\sigma_1$", fontsize="x-large")
ax.set_ylabel(r"$\hat \mu$", fontsize="x-large")
ax.axhline(.5, color="blue")
# ax.grid()


ax = axs[1]
data_subset = data.loc[data["parameter"] == "target_mean"]
data_subset = data_subset.loc[data_subset["estimator"] != "OR"]

sns.boxplot(data=data_subset, x="sigma_1", y="estimate", hue="estimator", ax = ax)
ax.set_xlabel(r"$\sigma_1$", fontsize="x-large")
ax.set_ylabel(r"$\hat \mu_0$", fontsize="x-large")
ax.axhline(.6, color="blue")
# ax.grid()
fig.savefig("plots/estimates.pdf", bbox_inches = "tight")
