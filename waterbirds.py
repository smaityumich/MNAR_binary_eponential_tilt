import numpy as np 
from waterbirds_LR import load_waterbirds_data_full
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import os
from exp_grad import exponentiated_gradient, ext_x, profile_likelihood_opt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True
mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import seaborn as sns

data_path = "/Users/smaity/projects/MNAR/codes/dataset/"

a = pd.read_csv(data_path + "metadata.csv")
a["g"] = 2 * a["y"] + a["place"]
a.groupby(["split", "g"])["y"].count()

((train_x, train_y, train_g), (test_x, test_y, test_g)), _ = load_waterbirds_data_full(data_path)

str_x, sts_x, str_y, sts_y, str_g, sts_g = train_test_split(train_x, train_y, train_g,  test_size=0.25)
ttr_x, tts_x, ttr_y, tts_y, ttr_g, tts_g = train_test_split(test_x, test_y, test_g, test_size=0.25)



meta_data = pd.read_csv("/Users/smaity/projects/MNAR/codes/dataset/metadata.csv")
meta_data["R"] = (meta_data["split"] == 0).astype("int")
count_0 = 6993
count_1 = 4795
meta_data["c"] = meta_data["R"] / count_1 + (1 - meta_data["R"])/count_0


meta_data["g"] = 2 * meta_data["y"] + meta_data["place"]
df = meta_data.groupby(["R", "g"])["c"].sum()
df = df.reset_index()  
df["c"] = df["c"].round(2)
df = df.astype({"R": str})
df = df.replace({"1": r"source; $R = 1$", "0": "target; $R = 0$"})

fig, ax = plt.subplots(1, 1, figsize = (4, 3))

w = sns.barplot(data=df, x="g", y="c", hue="R", ax = ax)
ax.set_xlabel(r"$(Y, A)$", fontsize="x-large")
ax.set_ylabel(r"group proportions", fontsize="x-large")
x_ticks = [r"$(0,0)$", r"$(0,1)$", r"$(1,0)$", r"$(1,1)$"] 
ax.set_xticklabels(x_ticks, rotation = 0)
w.bar_label(w.containers[0], fontsize=10);
w.bar_label(w.containers[1], fontsize=10);
ax.set_ylim(0, 0.8)
legend = ax.legend()
legend.texts[0].set_text("target; $R = 0$")

fig.savefig("plots/waterbirds_rep.pdf", bbox_inches="tight")


with open(os.path.join(data_path, "classifier_scores.npy"), "rb") as f:
    p_train = np.load(f)
    p_test = np.load(f)

p_train = np.exp(p_train)
p_train = p_train / p_train.sum(axis = 1, keepdims = True)
p_test = np.exp(p_test)
p_test = p_test / p_test.sum(axis = 1, keepdims = True)
    
np.random.seed(12121)

DR = []
DR_SVM = []
OR = []
IW = []
source = []
target = []
RW = []

acc_all_covariates = []
acc_IV = []

for i in range(20):
    
    print("\n\nIteration", i)
    
    np.random.seed(12121 + 200 * i)
    
    str_x, sts_x, str_y, sts_y, str_g, sts_g, str_p, sts_p = train_test_split(train_x, train_y, train_g, p_train, test_size=0.25)
    ttr_x, tts_x, ttr_y, tts_y, ttr_g, tts_g, ttr_p, tts_p = train_test_split(test_x, test_y, test_g, p_test, test_size=0.25)

    str_x = train_x
    str_y = train_y
    str_p = p_train
    str_g = train_g
    
    
    
    beta = exponentiated_gradient(str_x, str_y, str_p, ttr_x, ttr_p, 
                                            eps=1e-3, tol=3e-3, B=5, lr=4e-3, 
                                            max_iter=4000, verbose=0)


    # beta = profile_likelihood_opt(str_x, str_y, str_p, ttr_x, ttr_p, 
    #                                          tol=2e-4, lr=4e-3, max_iter=4000, verbose=1, reg = 1e-3)

    str_b = str_g - 2 * str_y
    sts_b = sts_g - 2 * sts_y
    ttr_b = ttr_g - 2 * ttr_y
    tts_b = tts_g - 2 * tts_y


    lr_b = LogisticRegression(penalty="l2", C = 0.001).fit(ttr_x, ttr_b)
    acc = lr_b.score(tts_x, tts_b)
    print("Accuracy with all covariates", acc)
    acc_all_covariates.append(acc)

    
    lr_b = LogisticRegression().fit(ext_x(ttr_x) @ beta, ttr_b)
    acc = lr_b.score(ext_x(tts_x) @ beta, tts_b)
    print("Accuracy with IV covariates", acc)
    acc_IV.append(acc)

    ## Target model
    lr_b = LogisticRegression(penalty="l2", C = 0.001).fit(ttr_x, ttr_y)
    acc = lr_b.score(tts_x, tts_y)
    print("Oracle accuracy", acc)
    target.append(acc)

    ## IW model
    str_w = np.exp(ext_x(str_x) @ beta)
    weights = str_w[np.arange(str_w.shape[0], ), str_y]

    lr_w = LogisticRegression(penalty="l2", C = 0.001).fit(str_x, str_y, sample_weight=weights)
    acc = lr_w.score(tts_x, tts_y)
    print("IW accuracy", acc)
    IW.append(acc)

    ## Source model
    b_init = np.copy(beta[:, 0])
    b_init[0] = lr_w.intercept_[0]
    b_init[1:] = lr_w.coef_


    lr_b = LogisticRegression(penalty="l2", C = 0.001).fit(str_x, str_y)
    acc = lr_b.score(tts_x, tts_y)
    print("source accuracy", acc)
    source.append(acc)
    
    ## OR model
    str_x_ext = ext_x(str_x)
    alpha = str_x_ext @ beta
    log_or_str = alpha[:, 1] - alpha[:, 0]
    m0_str = (np.exp(log_or_str) * str_p[:, 1]) / (np.exp(log_or_str) * str_p[:, 1] + str_p[:, 0])

    ttr_x_ext = ext_x(ttr_x)
    alpha = ttr_x_ext @ beta
    log_or_ttr = alpha[:, 1] - alpha[:, 0]
    m0_ttr = np.exp(log_or_ttr) * ttr_p[:, 1] / (np.exp(log_or_ttr) * ttr_p[:, 1] + ttr_p[:, 0])
    
    ttr_n = ttr_x.shape[0]
    ttr_x_all = np.concatenate((ttr_x, ttr_x), axis = 0)
    ttr_y_all = np.array([0.] * ttr_n + [1.] * ttr_n)
    ttr_w_all = np.concatenate((1 - m0_ttr, m0_ttr))
    lr_b = LogisticRegression(penalty="l2", C = 0.001).fit(ttr_x_all, ttr_y_all, ttr_w_all)
    acc = lr_b.score(tts_x, tts_y)
    print("OR accuracy", acc)
    OR.append(acc)
    
    ## DR estimates
    str_w = np.exp(ext_x(str_x) @ beta)
    weights = str_w[np.arange(str_w.shape[0], ), str_y]

    def grad_DR(theta):
        str_l = str_x_ext @ theta.reshape((-1, 1))
        ttr_l = ttr_x_ext @ theta.reshape((-1, 1))
        l_ret = (weights.reshape((-1, 1)) * str_x_ext / (1 + np.exp(-str_l))).mean(axis = 0)
        l_ret += (ttr_x_ext / (1 + np.exp(-ttr_l))).mean(axis = 0)
        l_ret -= ((m0_ttr).astype("float").reshape((-1, 1)) * ttr_x_ext).mean(axis = 0)
        l_ret -= ((str_y - (m0_str).astype("float")).reshape((-1, 1)) * str_x_ext * weights.reshape((-1, 1))).mean(axis = 0)
        return l_ret


    b_init = np.copy(beta[:, 0])
    theta = b_init
    err = 1 
    ITER = 0 
    max_iter = 2000
    thr = 1e-3
    lr = 1e-2
    c = 0.01
    while err > thr and ITER < max_iter:
        g = grad_DR(theta)
        err = np.linalg.norm(g * lr) / np.linalg.norm(theta)
        theta = (1 - lr * c) * theta - lr * g
        ITER += 1
        # if ITER %10 == 0:
        #     print("ITER", ITER, "err", err)
        
    pred = (ext_x(tts_x) @ theta.reshape((-1, 1)) > 0).astype("float").reshape((-1, ))
    acc = np.mean(tts_y == pred)
    print("DR accuracy", acc) 
    DR.append(acc)
    
    weights = np.array(df["c"])
    rw = (weights[:4]/weights[4:]).reshape((1, -1))
    str_g_onehot = np.zeros((str_g.size, str_g.max() + 1))
    str_g_onehot[np.arange(str_g.size), str_g] = 1
    weights = (str_g_onehot * rw).sum(axis = 1)
    lr_w = LogisticRegression(penalty="l2", C = 0.001).fit(str_x, str_y, sample_weight=weights)
    acc = lr_w.score(tts_x, tts_y)
    print("RW accuracy", acc)
    RW.append(acc)
    
    

DR = np.array(DR)
OR = np.array(OR)
IW = np.array(IW)
source = np.array(source)
target = np.array(target)
RW = np.array(RW)

acc_all_covariates = np.array(acc_all_covariates)
acc_IV = np.array(acc_IV)


data = pd.DataFrame({"IW": IW, "OR": OR, "DR": DR, 
                     "source": source, "target": target, "reweight": RW,
                     "acc_all_covariates": acc_all_covariates, 
                     "acc_IV": acc_IV})

data.to_csv("plots/results_waterbird_v2.csv", index_label="False")   

data = pd.read_csv("plots/results_waterbird_v2.csv")  
data = data.drop(columns = ["False",]).T   
n = data.shape[0]
data = data.iloc[list(np.arange(n-2, ))]
data = data.T

measures = []
metrics = []
for c in data.columns:
    v = list(data[c])
    metrics += v
    measures += [c] * len(v)
    
df_plot = pd.DataFrame({"measures": measures, "metrics": metrics})



fig, ax = plt.subplots(1, 1, figsize = (4, 3))

w = sns.barplot(data=df_plot, x="measures", y="metrics", ax = ax, 
                errorbar=("pi", 50), capsize=.4,
    err_kws={"color": ".3", "linewidth": 1.},
    linewidth=1.5, edgecolor="0.2", facecolor=(0, 0, 0, 0),)
ax.set_xlabel("", fontsize="x-large")
ax.set_ylabel(r"target accuracy", fontsize="x-large")
ax.set_ylim([0.75, 0.95])
ax.grid()
fig.savefig("plots/waterbirds_acc_v2.pdf", bbox_inches="tight")

data.agg(["mean", "std"])
