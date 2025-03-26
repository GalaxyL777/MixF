#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 16:40:15 2025

@author: xiaohuiliu
"""

import numpy as np
import matplotlib.pyplot as plt
import preprocessing as prep
from scipy.constants import c

# RM of this burst is -672.199
RM = -672.199
Stokes, t, f, MJDs = np.load('FRB20201124A_track_M01_59326.23324917.npy', allow_pickle=True)
I, Q, U, V = Stokes

P = np.sqrt(Q**2+U**2+V**2)

# choose a section of noise
noise_sel = np.logical_and(t>0, t<30) # protected variables

# handle a RFI mask
RFI_sel = prep.look_for_RFI(P, t, f, noise_sel, 'FRB20201124A1_'+'59326.23324917')     # protected variables

# remove these RFI channel
t, f = t, f[~RFI_sel[::-1]]
I1, Q1, U1, V1 = I[~RFI_sel[::-1],:], Q[~RFI_sel[::-1],:], U[~RFI_sel[::-1],:], V[~RFI_sel[::-1],:]

P1 = np.sqrt(Q1**2+U1**2+V1**2)

# remove small P channels
P2 = P1 - np.mean(P1[:,noise_sel], axis=1, keepdims=True)
t,f = t, f
I2, Q2, U2, V2 = I1, Q1, U1, V1

Qerr_f = np.nanstd(Q2[:,noise_sel], axis=1)
Uerr_f = np.nanstd(U2[:,noise_sel], axis=1)
Verr_f = np.nanstd(V2[:,noise_sel], axis=1)

# mask the P < 7*noise
P2_t = np.nanmean(P2, axis=0)
P5noise_sel = P2_t < np.mean(P2_t[noise_sel]) + 7*np.std(P2_t[noise_sel])

I2[:,P5noise_sel] = np.nan
Q2[:,P5noise_sel] = np.nan
U2[:,P5noise_sel] = np.nan
V2[:,P5noise_sel] = np.nan
P2[:,P5noise_sel] = np.nan

f = f[::-1]*1e6
P_f = np.nanmean(P2, axis=1)
Q_f = np.nanmean(Q2, axis=1)
U_f = np.nanmean(U2, axis=1)
V_f = np.nanmean(V2, axis=1)


# the final f and \vec{P}(f)
f = f[~np.isnan(P_f)]
lambda_square = (c/f)**2
Q_f, U_f, V_f = Q_f[~np.isnan(P_f)], U_f[~np.isnan(P_f)], V_f[~np.isnan(P_f)]
Qerr_f, Uerr_f, Verr_f = Qerr_f[~np.isnan(P_f)], Uerr_f[~np.isnan(P_f)], Verr_f[~np.isnan(P_f)]

# filter the low P channels (0.04841591 is the noise std of the off-pulse P_f after we subtract the time noise mean for each channel)
sel_lowP = np.logical_and(~(P_f < 0.04841591*10), f>1.02e9) 

f = f[sel_lowP]
lambda_square = lambda_square[sel_lowP]
Q_f, U_f, V_f = Q_f[sel_lowP], U_f[sel_lowP], V_f[sel_lowP]
Qerr_f, Uerr_f, Verr_f  = Qerr_f[sel_lowP], Uerr_f[sel_lowP], Verr_f [sel_lowP]
P_normal = np.sqrt(Q_f**2+U_f**2+V_f**2)

# # derotate QU
# L_f = Q_f + 1j*U_f
# L_f_deRM = L_f * np.exp(-2j * RM * lambda_square)
# Q_f_deRM, U_f_deRM = np.real(L_f_deRM), np.imag(L_f_deRM)


f = f
QoP, UoP, VoP = Q_f/P_normal, U_f/P_normal, V_f/P_normal
QoP_err = np.abs(QoP)*np.sqrt( ((U_f**2+V_f**2)/(Q_f*P_normal**2))**2*Qerr_f**2 + (U_f/P_normal**2)**2*Uerr_f**2 + (V_f/P_normal**2)**2*Verr_f**2 )/np.sqrt(np.sum(~P5noise_sel))
UoP_err = np.abs(UoP)*np.sqrt( (Q_f/P_normal**2)**2*Qerr_f**2 + ((Q_f**2+V_f**2)/(U_f*P_normal**2))**2*Uerr_f**2 + (V_f/P_normal**2)**2*Verr_f**2 )/np.sqrt(np.sum(~P5noise_sel))
VoP_err = np.abs(VoP)*np.sqrt( (Q_f/P_normal**2)**2*Qerr_f**2 + (U_f/P_normal**2)**2*Uerr_f**2 + ((V_f**2+Q_f**2)/(V_f*P_normal**2))**2*Verr_f**2 )/np.sqrt(np.sum(~P5noise_sel))

#%% conversion fit

'''
         This is debated
log10n0L, log10T = 22.60668139332857, 12.325797459817247
log10B, thetaB, chi = -2.5679797137729428, 29.877676700536554, 60
beta0, chi0 =  -114.69217160520388, 3.6753016991091525
RM_back, RM_fore, chi2_min = 19.244658162065647, -50, 0
'''


from qcosmc.MCMC import MCMC_class, MCplot
from Faraday_conversion import conversions_class, QUVs_likelihood
conversion = conversions_class()
likelihood = QUVs_likelihood(f,QoP,UoP,VoP,QoP_err,UoP_err,VoP_err,conversion)

parameters_best = []

### check cold plasma scenario
def chi2(theta):
    log10n0L, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore = theta
    P=1
    log10T = 0
    # RM_back, RM_fore = 0, 0
    # chi = 0
    
    beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
    I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)
    likelihood.update(10**log10n0L, 10**log10T, 10**log10B, thetaB, chi,
                        I0, Q0, U0, V0,
                        RM_back, RM_fore)
    chi2 = likelihood.log_likelihood()/(-0.5)
    if chi2 < 562056:
        # print(chi2)
        parameters_best.append( [log10n0L, log10T, log10B, thetaB, chi, beta0*180/np.pi, chi0*180/np.pi, RM_back, RM_fore, chi2])
    return chi2

params = [[r'\log_{10}(n_{0}L/ \mathrm{cm}^{-2})', 14.237675627306903, 5, 25],
          # [r'\log_{10}(T / \mathrm{K})', 6.98776239192092, -2, 10],
          [r'\log_{10}(B / \mathrm{G})', 1.586653469285314, -1, 5],
          [r'\theta_{B} (\mathrm{deg})', 111.16853712024374, 0, 180],
          [r'\chi_{p} (\mathrm{deg})', -61.085453193685694, -180, 180],
          [r'\beta_0 (\mathrm{deg})',-155.3298953483711, -90*3, 90],
          [r'\chi_0 (\mathrm{deg})', 6.971974776760106, -45, 90],
          [r'\mathrm{RM_{b}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', -32.28403834579892, -400, 400],
          [r'\mathrm{RM_{f}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', -8.563898117477915, -400, 400]
          ]

MC = MCMC_class(params, chi2, 'Cold_1124A_926')
MC.MCMC(200000)

chains = [
          ['Cold_1124A_926', 'FRB20201124A burst 926 (Cold)']
          ]
pl = MCplot(chains)
pl.plot3D([1,2,3,4,5,6,7,8])
pl.results2
pl.results


A = np.array(parameters_best)

np.save('./chains/Cold_1124A_926_listchi2.npy',A)
A = np.load('./chains/Cold_1124A_926_listchi2.npy')

log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]


# log10n0L, log10T = 14.5, 5
# log10B, thetaB, chi = 1.5, 104.9, -70.0
# beta0, chi0 =  19.52, 7
# RM_back, RM_fore, chi2_min = -31.7, -10.73, 0

n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
P = 1
I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)   

conversion.parameters_update(n0L, T, B, thetaB, chi, I0, Q0, U0, V0, RM_back, RM_fore)
num = np.linspace(1.0, 1.5, 1000)*1e9
Qm, Um, Vm = conversion.Stokes_calculate(num)

likelihood.update(10**log10n0L, 10**log10T, 10**log10B, thetaB, chi,
                  I0, Q0, U0, V0,
                  RM_back, RM_fore)
chi2 = likelihood.log_likelihood()/(-0.5)
print(chi2, chi2_min)



fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6), gridspec_kw={'hspace': 0})

ax1.errorbar(f, QoP, yerr=QoP_err, fmt='r.', markersize=2, alpha=0.5)
ax1.plot(num, Qm, 'r-')
ax1.set_ylabel('Q/P')
ax1.set_ylim([-1,1])

ax2.errorbar(f, UoP, yerr=UoP_err, fmt='g.', markersize=2, alpha=0.5)
ax2.plot(num, Um, 'g-')
ax2.set_ylabel('U/P')
ax2.set_ylim([-1,1])

ax3.errorbar(f, VoP, yerr=VoP_err, fmt='b.', markersize=2, alpha=0.5)
ax3.plot(num, Vm, 'b-')
ax3.set_xlabel('f')
ax3.set_ylabel('V/P')
ax3.set_ylim([-1,1])

plt.xlim([1e9, 1.5e9])
plt.savefig('./Cold_FRB20201124A_burst_926.pdf')


#%%
### check the hot plasma scenario
def chi2(theta):
    log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore = theta
    P=1
    
    beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
    I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)
    likelihood.update(10**log10n0L, 10**log10T, 10**log10B, thetaB, chi,
                        I0, Q0, U0, V0,
                        RM_back, RM_fore)
    chi2 = likelihood.log_likelihood()/(-0.5)
    
    if chi2 < 556717:
        parameters_best.append( [log10n0L, log10T, log10B, thetaB, chi, beta0*180/np.pi, chi0*180/np.pi, RM_back, RM_fore, chi2])
    return chi2

params = [[r'\log_{10}(n_{0}L/ \mathrm{cm}^{-2})', 24.821043213267075, 18, 35],
          [r'\log_{10}(T / \mathrm{K})', 13.275740621103614, 10, 20],
          [r'\log_{10}(B / \mathrm{G})', -2.98936330333464, -5, -1],
          [r'\theta_{B} (\mathrm{deg})', 179.6925353933328, 0, 180],
          [r'\chi_{p} (\mathrm{deg})', -58.35871699991151, -180, 180],
          [r'\beta_0 (\mathrm{deg})', -91.6449340437642, -90*3, 90],
          [r'\chi_0 (\mathrm{deg})', 5.546019255452664, -45, 90],
          [r'\mathrm{RM_{b}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', 9.349112875089173, -1000, 1000],
          [r'\mathrm{RM_{f}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', -8.124686065911114, -1000, 1000]
          ]

MC = MCMC_class(params, chi2, 'Hot_1124A_926')
MC.MCMC(200000)

chains = [
          ['Hot_1124A_926', 'FRB20201124A burst 926 (Hot)']
          ]
pl = MCplot(chains)
pl.plot3D([1,2,3,4,5,6,7,8,9])
pl.results2
pl.results


A = np.array(parameters_best)

np.save('./chains/Hot_1124A_926_listchi2.npy',A)
A = np.load('./chains/Hot_1124A_926_listchi2.npy')

log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]


log10n0L, log10T = 27.481814167223682, 13.311131714264453
log10B, thetaB, chi = -5.618074240059897, 178.7294966249979, 0
beta0, chi0 =  -5.638584538705724, 1.9841675472624232
RM_back, RM_fore, chi2_min = 0, 8, 0

n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
P = 1
I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)   

conversion.parameters_update(n0L, T, B, thetaB, chi, I0, Q0, U0, V0, RM_back, RM_fore)
num = np.linspace(1.0, 1.5, 1000)*1e9
Qm, Um, Vm = conversion.Stokes_calculate(num)

likelihood.update(10**log10n0L, 10**log10T, 10**log10B, thetaB, chi,
                  I0, Q0, U0, V0,
                  RM_back, RM_fore)
chi2 = likelihood.log_likelihood()/(-0.5)
print(chi2, chi2_min)


fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6), gridspec_kw={'hspace': 0})

ax1.errorbar(f, QoP, yerr=QoP_err, fmt='r.', markersize=2, alpha=0.5)
ax1.plot(num, Qm, 'r-')
ax1.set_ylabel('Q/P')
ax1.set_ylim([-1,1])

ax2.errorbar(f, UoP, yerr=UoP_err, fmt='g.', markersize=2, alpha=0.5)
ax2.plot(num, Um, 'g-')
ax2.set_ylabel('U/P')
ax2.set_ylim([-1,1])

ax3.errorbar(f, VoP, yerr=VoP_err, fmt='b.', markersize=2, alpha=0.5)
ax3.plot(num, Vm, 'b-')
ax3.set_xlabel('f')
ax3.set_ylabel('V/P')
ax3.set_ylim([-1,1])

plt.xlim([1e9, 1.5e9])
plt.savefig('./Hot_FRB20201124A_burst_926.pdf')

#%%
# cold scenario
A = np.load('./chains/Cold_1124A_926_listchi2.npy')
log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]
n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
P = 1
I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)   

conversion.parameters_update(n0L, T, B, thetaB, chi, I0, Q0, U0, V0, RM_back, RM_fore)
nu_cp = np.linspace(1.0, 1.5, 1000)*1e9
Q_cp, U_cp, V_cp = conversion.Stokes_calculate(nu_cp)

# hot scenarrio
A = np.load('./chains/Hot_1124A_926_listchi2.npy')
log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]
n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
P = 1
I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)   

conversion.parameters_update(n0L, T, B, thetaB, chi, I0, Q0, U0, V0, RM_back, RM_fore)
nu_hp = np.linspace(1.0, 1.5, 1000)*1e9
Q_hp, U_hp, V_hp = conversion.Stokes_calculate(nu_hp)



import matplotlib.pyplot as plt
# f,QoP,UoP,VoP,QoP_err,UoP_err,VoP_err
fig, axs = plt.subplots(3, 1, sharex=True)
# Remove vertical space between Axes
fig.subplots_adjust(hspace=0.15)
fontsize=12
# Plot each graph, and manually set the y tick values
axs[0].errorbar(f/1e9, QoP, yerr=QoP_err, fmt='r.', markersize=2, alpha=0.5, zorder=1)
axs[0].plot(nu_cp/1e9, Q_cp,'-', label='cold', zorder=2)
axs[0].plot(nu_hp/1e9, Q_hp,'--', label='hot', zorder=3)
# axs[0].set_yticks([-0.3, -0.1, 0.1, 0.3])
axs[0].set_ylim(-1.0, 1.0)
axs[0].set_ylabel(r'$Q/I$', fontsize=fontsize)
axs[0].legend()

axs[1].errorbar(f/1e9, UoP, yerr=UoP_err, fmt='r.', markersize=2, alpha=0.5,zorder=1)
axs[1].plot(nu_cp/1e9, U_cp,'-',zorder=2)
axs[1].plot(nu_hp/1e9, U_hp,'--',zorder=3)
# axs[1].set_yticks([-0.5, 0.0, 0.5, 1.0])
axs[1].set_ylim(-1.00, 1.00)
axs[1].set_ylabel(r'$U/I$', fontsize=fontsize)

axs[2].errorbar(f/1e9, VoP, yerr=VoP_err, fmt='r.', markersize=2, alpha=0.5,zorder=1)
axs[2].plot(nu_cp/1e9, V_cp,'-',zorder=2)
axs[2].plot(nu_hp/1e9, V_hp,'--',zorder=3)
axs[2].set_ylim(-1.00, 1.00)
# axs[2].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

axs[2].set_ylabel(r'$V/I$', fontsize=fontsize)
axs[2].set_xlabel(r'$\nu ~(\mathrm{GHz})$', fontsize=fontsize)

plt.legend()
plt.savefig('./results/1124A2_ColdandHot.pdf', dpi=100, bbox_inches='tight')
plt.show()



