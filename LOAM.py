import pandas as pd
import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib import ticker as mtick

def calculate_LOAM(data):
    CI = 0.95
    z = np.abs(stats.norm.ppf((1- CI) / 2))
    up = 1 - (1 - CI) / 2
    lo = (1 - CI) / 2

    a = data.shape[0] #subjects
    b = data.shape[1] #observers
    N = a * b

    vE = a * b - a - b + 1
    vA = a - 1
    vB = b - 1

    observerMean = data.mean(0)
    subjectMean = data.mean(1)
    valueMean = np.mean(data)

    SSE = 0
    for obs in range(0, b):
        for sub in range(0, a):
            SSE += (data[sub,obs] - subjectMean[sub] - observerMean[obs] + valueMean)**2

    SSA = b * sum((subjectMean - valueMean)**2)
    SSB = a * sum((observerMean - valueMean)**2)

    MSE = SSE / vE
    MSA = SSA / vA
    MSB = SSB / vB

    sigma2E = MSE
    sigma2A = (MSA - MSE) / b
    sigma2B = (MSB - MSE) / a

    sigmaE = np.sqrt(sigma2E)
    sigmaA = np.sqrt(sigma2A)
    sigmaB = np.sqrt(sigma2B)

    LOAM = 1.96 * np.sqrt((SSB + SSE) / N)

    if sigma2A >= 0:
        ICC = sigma2A / (sigma2A + sigma2B + sigma2E)
        A = b * ICC / (a * (1 - ICC))
        B = 1 + b * ICC * (a - 1) / (a * (1 - ICC))
        v = (A * MSB + B * MSE)**2 / ((A * MSB)**2 / vB + (B * MSE)**2 / vE)
        FL = stats.f.ppf(up, a - 1, v)
        FU = stats.f.ppf(up, v, a - 1)
        low_num = a * (MSA - FL * MSE)
        low_denom = FL * (b * MSB + (a * b - a - b) * MSE) + a * MSA
        upp_num = a * (FU * MSA - MSE)
        upp_denom = b * MSB + (a * b - a - b) * MSE + a * FU * MSA
        ICC_CI = [low_num / low_denom, upp_num / upp_denom]


    lB = 1 - 1 / stats.f.ppf(up, vB, 10**9)
    hB = 1 / stats.f.ppf(lo, vB, 10**9) - 1
    lE = 1 - 1 / stats.f.ppf(up, vE, 10**9)
    hE = 1 / stats.f.ppf(lo, vE, 10**9) - 1

    H = np.sqrt(hB**2 * SSB**2 + hE**2 * SSE**2)
    L = np.sqrt(lB**2 * SSB**2 + lE**2 * SSE**2)

    LOAM_CI = [z * np.sqrt((SSB + SSE - L) / N), z * np.sqrt((SSB + SSE + H) / N)]

    rel_LOAM = LOAM / valueMean
    rel_LOAM_CI = LOAM_CI / valueMean

    sigmaA_dCI = (((z / b) * np.sqrt((1 / (2 * sigma2A)) * (((b * sigma2A + sigma2E)**2 / vA)) + (sigma2E**2 / vE))))
    sigmaA_CI = [sigmaA - sigmaA_dCI, sigmaA + sigmaA_dCI]
    sigmaB_dCI = (((z / a) * np.sqrt((1 / (2 * sigma2B)) * (((a * sigma2B + sigma2E)**2 / vB)) + (sigma2E**2 / vE))))
    sigmaB_CI = [sigmaB - sigmaB_dCI, sigmaB + sigmaB_dCI]

    sigmaE_CI = [sigmaE * np.sqrt(vE / stats.chi2.ppf(up, vE)), sigmaE * np.sqrt(vE / stats.chi2.ppf(lo, vE))]

    return valueMean, subjectMean, rel_LOAM, rel_LOAM_CI

#Reading data
data = pd.read_excel('FILE.xlsx', usecols=[])
#data = data[data.cystic != 1]
#data = data.drop(columns='cystic')
W_size = 26

vM = []
sM = []
rLM = []
rLMCI = []
print('N Patients =', len(data))
for n in range(0, len(data)-W_size):
    valueMean, subjectMean, rel_LOAM, rel_LOAM_CI = calculate_LOAM(data.values[n:n+W_size,:])
    vM.append(valueMean)
    sM.append(subjectMean)
    rLM.append(100*rel_LOAM)
    rLMCI.append(100*rel_LOAM_CI)

fig, ax = plt.subplots()
fig.set_size_inches((8,6))

plt.plot(vM, rLM, color='b', label = 'LOAM')
plt.plot(vM, np.array(rLM)*-1.0, color='b')
ax.fill_between(vM, np.array(rLMCI)[:,0], np.array(rLMCI)[:,1], alpha=0.25, color='b', label = 'LOAM CI-95%')
ax.fill_between(vM, -1*np.array(rLMCI)[:,0], -1* np.array(rLMCI)[:,1], alpha=0.25, color='b')
plt.scatter(data.mean(1), 100 * (data.values[:,0]/data.mean(1) - 1), marker='o', color='grey', label = 'Observer')
plt.scatter(data.mean(1), 100 * (data.values[:,1]/data.mean(1) - 1), marker='^', color='grey')
plt.scatter(data.mean(1), 100 * (data.values[:,2]/data.mean(1) - 1), marker='*', color='grey')
plt.scatter(data.mean(1), 100 * (data.values[:,3]/data.mean(1) - 1), marker='x', color='grey')
plt.scatter(data.mean(1), 100 * (data.values[:,4]/data.mean(1) - 1), marker='+', color='grey')
plt.xscale('log')
plt.xlabel('Volume [mm\u00b3]', fontname="Times New Roman")
plt.ylabel('Deviation from the mean [%]', fontname="Times New Roman")
plt.title('Volume Dependent Agreement Plot \n (Excluding Tumors with Peritumoral Cysts)', fontname="Times New Roman", fontweight='bold', fontsize=14)
plt.axhline(y = 0, color = 'r')
plt.axhline(y = 20, color = 'r', linestyle = ':')
plt.axhline(y = -20, color = 'r', linestyle = ':')
plt.ylim([-40, 40])
plt.xlim([20, 15000])
plt.xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000], [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000])
plt.legend(prop={'family':"Times New Roman"})
ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
ax.grid(alpha=0.4)

plt.show()
print()