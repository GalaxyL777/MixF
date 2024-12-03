#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 09:34:05 2024

@author: xiaohuiliu
"""
import numpy as np

from scipy.special import kv
from scipy.constants import pi                                                 # Mathematical constants
from scipy.constants import c, e, m_e, k                                       # Physical constants: speed of light (m s^-1), elementary charge (C), electron mass (kg), Boltzmann constant (J K^-1)
SIcgs_c, SIcgs_e, SIcgs_me, SIcgs_k = 1e2, 1/3.335640952e-10, 1e3, 1e7
c, e, m_e, k = c*SIcgs_c, e*SIcgs_e, m_e*SIcgs_me, k*SIcgs_k # Gauss units (cm s^-1, esu, g, 1)

#%%
class GFR_class(object):
    def __init__(self):
        # the default input angle is in unit of degree
        
        # basic parameters
        self.RM = 40
        self.psi0 = 10 /180*np.pi
        
        self.theta = 10 /180*np.pi
        self.phi = 20 /180*np.pi
        
        self.GRM = 1000        
        self.Psi0 = 20 /180*np.pi
        self.alpha = 2.3
        
        self.chi = 0 /180*np.pi
        
        # important variable
        self.Qn, self.Un, self.Vn = 1, 1, 1
    
    def parameters_update(self, RM, psi0, theta, phi, GRM, Psi0, alpha, chi):
        self.RM = RM
        self.psi0 = psi0/180*np.pi
        self.theta = theta/180*np.pi
        self.phi = phi/180*np.pi
        self.GRM = GRM
        self.Psi0 = Psi0/180*np.pi
        self.alpha = alpha
        self.chi = chi/180*np.pi
    
    def Stokes_calculate(self,nu):
        wavelength = 299792458.0 / nu # nu in unit of Hz
        # wavelength0 = (299792458.0/1.35e9+299792458.0/1.525e9)/2  # the maximum frequency of bandwidth
        wavelength0 = 299792458.0/1.375e9
        # wavelength0 = 299792458.0/1.5e9
        psi = self.psi0 + self.RM*(np.power(wavelength,2)-np.power(wavelength0,2))
        Psi = self.Psi0 + self.GRM*(np.power(wavelength,self.alpha)-np.power(wavelength0,self.alpha))
        
        R11 = np.cos(2*psi)*np.cos(self.theta)*np.cos(self.phi) - np.sin(2*psi)*np.sin(self.phi)
        R12 = -np.cos(2*psi)*np.cos(self.theta)*np.sin(self.phi) - np.sin(2*psi)*np.cos(self.phi)
        R13 = np.cos(2*psi)*np.sin(self.theta)
        
        R21 = np.sin(2*psi)*np.cos(self.theta)*np.cos(self.phi) + np.cos(2*psi)*np.sin(self.phi)
        R22 = -np.sin(2*psi)*np.cos(self.theta)*np.sin(self.phi) + np.cos(2*psi)*np.cos(self.phi)
        R23 = np.sin(2*psi)*np.sin(self.theta)
        
        R31 = -np.sin(self.theta)*np.cos(self.phi)
        R32 = np.sin(self.theta)*np.sin(self.phi)
        R33 = np.cos(self.theta)
        
        Q = R11*np.cos(2*Psi)*np.cos(2*self.chi) + R12*np.sin(2*Psi)*np.cos(2*self.chi) + R13*np.sin(2*self.chi)
        U = R21*np.cos(2*Psi)*np.cos(2*self.chi) + R22*np.sin(2*Psi)*np.cos(2*self.chi) + R23*np.sin(2*self.chi)
        V = R31*np.cos(2*Psi)*np.cos(2*self.chi) + R32*np.sin(2*Psi)*np.cos(2*self.chi) + R33*np.sin(2*self.chi)
        self.Q, self.U, self.V = Q, U, V
        return Q, U, V

class PEPWs_class(object):
    '''
    This is the approximate solution of the transfer of polarization of FRB as a
    strong incoming wave propagating in a homogeneous magnetized plasma. This
    is the case of the thermal plasma (section 2.1) when zeta^*_s >> zeta_s.
    '''
    def __init__(self, electrons_model='Thermal'):
        self.electrons_model = electrons_model                                 # only support 'Thermal' and 'Power-law'
        
        # basic parameters
        self.n0L = 1e4                                                         # number density (cm^(-3)) * typical size of medium (cm)
        self.T = 100                                                           # temperature of medium (K)
        self.B = 1                                                             # magnetic field (G)
        self.theta_B = 70/180*np.pi                                            # angle between the magnetic field B and wave vector
        self.I0, self.Q0, self.U0, self.V0 = 1, 1, 1, 1                        # the initial Stokes angles of the incoming wave (indeed no units)
        
        # important variables
        self.chistao = None
        self.I, self.Q, self.U, self.V = None, None, None, None
        self.eta_I, self.eta_Q, self.eta_V = None, None, None
        self.rho_Q, self.rho_V = None, None
        self.Matrix = None
    
    def parameters_update(self, n0L, T, B, theta_B, I0, Q0, U0, V0):
        self.n0L = n0L
        self.T = T
        self.B = B
        self.theta_B = theta_B/180*np.pi # the input theta_B is in units of degree
        self.I0, self.Q0, self.U0, self.V0 = I0, Q0, U0, V0

    def Stokes_calculate(self, nu):
        # calculate the cyclotron frequency
        omega_B = e*self.B/(m_e*c)
        nu_B = omega_B/2/pi                                                    # Hz
        rho = m_e*c**2/(k*self.T)
        X = 10**(3/2) * 2**(1/4) / rho * (nu_B/nu*np.sin(self.theta_B))**(1/2)
        # print(X)
        
        # check whether this combination of parameters is physical
        # rho_over_eta_Q = np.power(k*self.T,3/2) * np.sqrt(m_e) * nu * np.power(e,-4) * np.power(self.n0L)
        # if np.min(X) > 10:
        #     print('Danger')
        
        # defination of E_s and R_s (E_s = eta_s/n0**2 and R_s = rho_s/n0)
        # E_I = 8/3/np.sqrt(2*pi) * np.power(e,6)/np.power(k*self.T*m_e,3/2)/c/np.power(nu,2) * np.log( np.power(2*k*self.T, 3/2)/(4.2*pi*np.power(e,2)*np.power(m_e,1/2)*nu) )
        # E_Q = 3/8/pi**2 * np.power(omega_B*np.sin(self.theta_B)/nu,2) * E_I
        # E_V = omega_B*np.cos(self.theta_B)/pi/nu * E_I
        
        R_V = -np.power(omega_B/nu,2) * e/self.B * np.cos(self.theta_B)/pi * kv_ratio(0,2,rho) * Function_g(X)
        R_Q = -np.power(omega_B/nu,3) * e/self.B * np.power(np.sin(self.theta_B),2)/(4*pi*pi) * (kv_ratio(1,2,rho)+6/rho) * Function_f(X)
        
        # the dot and cross products of q, v, k
        qdotv = R_Q*R_V/(np.power(R_Q,2)+np.power(R_V,2))
        qdotk = R_Q/np.sqrt(np.power(R_Q,2)+np.power(R_V,2))
        vdotk = R_V/np.sqrt(np.power(R_Q,2)+np.power(R_V,2))
        q2 = np.power(qdotk,2)
        v2 = np.power(vdotk,2)
        
        # the simplified solution
        chistao = np.sqrt(np.power(R_Q,2)+np.power(R_V,2)) * self.n0L
        M11 = 0.5*(1+q2+v2) + 0.5*(1-q2-v2)*np.cos(chistao)
        M12 = 0
        M13 = 0
        M14 = 0
        
        M21 = 0
        M22 = 0.5*(1+q2-v2) + 0.5*(1-q2+v2)*np.cos(chistao)
        M23 = - vdotk*np.sin(chistao)
        M24 = qdotv - qdotv*np.cos(chistao)
        
        M31 = 0
        M32 = vdotk*np.sin(chistao)
        M33 = 0.5*(1-q2-v2) + 0.5*(1+q2+v2)*np.cos(chistao)
        M34 = - qdotk*np.sin(chistao)
        
        M41 = 0
        M42 = qdotv - qdotv*np.cos(chistao)
        M43 = qdotk*np.sin(chistao)
        M44 = 0.5*(1-q2+v2) + 0.5*(1+q2-v2)*np.cos(chistao)
        
            # update
        self.I = (M11*self.I0 + M12*self.Q0 + M13*self.U0 + M14*self.V0)
        self.Q = (M21*self.I0 + M22*self.Q0 + M23*self.U0 + M24*self.V0)
        self.U = (M31*self.I0 + M32*self.Q0 + M33*self.U0 + M34*self.V0)
        self.V = (M41*self.I0 + M42*self.Q0 + M43*self.U0 + M44*self.V0)
        return self.Q, self.U, self.V

    def call_Stokes(self):
        '''
        This function can return the Stokes parameters for easy checking.
        Returns
        -------
        arrays
            Stokes parameters.
        '''
        return self.I, self.Q, self.U, self.V
    
    def call_Matrix(self):
        ''' Not available now
        The Matrix.
        Stokes = exp(-tao)*Matrix*Stokes_0
        Returns
        -------
        M : list
            list of arrays
        '''
        return self.Matrix

class QUVs_Likelihood(object):
    def __init__(self, nu, Q, U, V, Qerr, Uerr, Verr, Models):
        """
        A very simple QUV likelihood based on the Gaussian noise.
        In this method, both Qerr, Uerr, Verr are known quantities.

        Parameters
        ----------
        data: array_like
            The data to analyse
        """
        
        self.nu = nu
        self.Q, self.Qerr = Q, Qerr
        self.U, self.Uerr = U, Uerr
        self.V, self.Verr = V, Verr
        
        self.Models = Models
        
        self.parameters = {}
    
    def update(self, parameters):
        self.parameters['log10n0L'] = parameters['log10n0L']
        self.parameters['log10T'] = parameters['log10T']
        self.parameters['log10B'] = parameters['log10B']
        self.parameters['thetaB'] = parameters['thetaB']
        self.parameters['I0'] = parameters['I0']
        self.parameters['P'] = parameters['P']
        self.parameters['beta0'] = parameters['beta0']/180*np.pi
        self.parameters['chi0'] = parameters['chi0']/180*np.pi
    
    def log_likelihood(self):
        # get the parameters
        n0L = 10**self.parameters['log10n0L']
        T = 10**self.parameters['log10T']
        B = 10**self.parameters['log10B']
        thetaB = self.parameters['thetaB']
        I0 = self.parameters['I0']
        P = self.parameters['P']
        beta0 = self.parameters['beta0']
        chi0 = self.parameters['chi0']
        
        self.Models.parameters_update(n0L, T, B, thetaB, I0, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0))
        Qm, Um, Vm = self.Models.Stokes_calculate(self.nu)
        
        if np.sum(np.isinf(Qm)):
            return -np.inf
        
        # chi2 of Q, U, V
        chi2Q = np.sum( np.power((self.Q-Qm)/self.Qerr,2) )
        chi2U = np.sum( np.power((self.U-Um)/self.Uerr,2) )
        chi2V = np.sum( np.power((self.V-Vm)/self.Verr,2) )
        return -0.5 * (chi2Q+chi2U+chi2V)

#------------------------------ basic functions ------------------------------#
def Function_f(X):
    '''
    Here X is 10^(3/2) 2^(1/4) rho^(-1) (omega_B/omega sin(theta_B))^(1/2)
    '''
    return 2.011*np.exp(-X**1.035/4.7) - np.cos(X/2)*np.exp(-X**1.2/2.73) - 0.011*np.exp(-X/47.2)

def Function_g(X):
    return 1 - 0.11*np.log(1+0.035*X)

def kv_ratio(n1,n2,x):
    # n1 = 0 or 1, does not support other numbers, n2 must be 2
    # when x > 300, abs(1-kv0/kv2) < 0.01 and abs(1-kv1/kv2) < 0.01, so we just set it to be 1 for simplicity
    if x > 300:
        y = 1
    else:
        y = kv(n1,x)/kv(n2,x)
    return y

def log_prior(theta, params):
    for i in range(len(theta)):
        if theta[i] <= params[i][2] or theta[i] >= params[i][3]:
            return -np.inf
    return 0

#%% simulation
import emcee
from getdist import plots, MCSamples
# generate a mock GFR data using the best-fit parameters from FRB 20180301A (arxiv: 2405.11515)
GFR = GFR_class()
RM, psi0, theta, phi, GRM, Psi0, alpha, chi = 27.7, -87.3, 104.2, 76.3, 4351.7, 0, 2.3, -0.1 
GFR.parameters_update(RM, psi0, theta, phi, GRM, Psi0, alpha, chi)

nu1 = np.linspace(1.370, 1.425, 50)*1e9
Q1,U1,V1 = GFR.Stokes_calculate(nu1)
Q1err,U1err,V1err = Q1*0+0.05,Q1*0+0.05,Q1*0+0.05 # add a 0.05 error for each data point

# initialize the IPEPWs and likelihood
IPs = PEPWs_class()
likelihood = QUVs_Likelihood(nu1,Q1,U1,V1,Q1err,U1err,V1err,IPs)


### the cold plasma scenario
# emcee
params = [[r'\log_{10}(n_{0}L / 1 \mathrm{cm}^{-2})', 12.714782341124213, 0, 15],
          [r'\log_{10}(T / 1 \mathrm{K})', 7.156614900042346, 1, 10],
          [r'\log_{10}(B / 1 \mathrm{G})', 3.11242638877485, 2, 5],
          [r'\theta_{B} (deg)', 72.62437743496272, 0, 180],
          ['P', 0.9988259112722111, 0,1],
          [r'\beta_0 (deg)', 134.50993566193597, 0, 180],
          [r'\chi_0 (deg)', 2.3791183136886493, -45, 45],
          ]

def log_probability(theta):
    lp = log_prior(theta, params)
    
    log10n0L, log10T, log10B, thetaB, P, beta0, chi0 = theta
    likelihood.update({'log10n0L':log10n0L, 'log10T':log10T, 'log10B':log10B, 'thetaB':thetaB,
                       'I0': 1, 'P':P, 'beta0':beta0, 'chi0':chi0})
    ll = likelihood.log_likelihood()
    return lp + ll

pos = np.array([param[1] for param in params]) + 1e-4*np.random.randn(32, len(params))
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 100000, progress=True)

# getdist
samples = MCSamples(samples=sampler.chain[:, 100:, :].reshape((-1, len(params))), names=[param[0] for param in params], labels=[param[0] for param in params])

g = plots.get_subplot_plotter()
g.triangle_plot(samples, filled=True)

### the hot plasma scenario
params = [[r'\log_{10}(n_{0}L)',21.940784429414933, 20, 23],
          [r'\log_{10}(T)', 11.73896501100829, 10, 13],
          [r'\log_{10}(B)',-2.476278011242103, -5, -1],
          [r'\theta_{B}',65.7711004245088, 50, 80],
          ['P', 0.9973851508392868, 0.9,1],
          [r'\beta_0', 56.71139513661876, -90, 90],
          [r'\chi_0',28.89873359547977, 0, 90],
          ]

def log_probability(theta):
    lp = log_prior(theta, params)
    
    log10n0L, log10T, log10B, thetaB, P, beta0, chi0 = theta
    likelihood.update({'log10n0L':log10n0L, 'log10T':log10T, 'log10B':log10B, 'thetaB':thetaB,
                       'I0': 1, 'P':P, 'beta0':beta0, 'chi0':chi0})
    ll = likelihood.log_likelihood()
    return lp + ll

pos = np.array([param[1] for param in params]) + 1e-4*np.random.randn(32, len(params))
nwalkers, ndim = pos.shape
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability)
sampler.run_mcmc(pos, 100000, progress=True)

# getdist
from getdist import plots, MCSamples
samples = MCSamples(samples=sampler.chain[:, 100:, :].reshape((-1, len(params))), names=[param[0] for param in params], labels=[param[0] for param in params])

g = plots.get_subplot_plotter()
g.triangle_plot(samples, filled=True)

#%%
import matplotlib.pyplot as plt

nu_GFR = np.linspace(1.370, 1.425, 50)*1e9
Q_GFR,U_GFR,V_GFR = GFR.Stokes_calculate(nu_GFR)

# cold
n0L, T, B, theta_B = 10**12.712596249571535, 10**7.148993793827589, 10**3.1137075160415386, 72.54382124774659
P, beta0, chi0 = 0.9990242267738322, 134.61817187103765, 1.960448266625026
I0, Q0, U0, V0 = 1, P*np.cos(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*chi0/180*np.pi)
IPs.parameters_update(n0L, T, B, theta_B, I0, Q0, U0, V0)
nu_cp = np.linspace(1.2, 1.6, 1000)*1e9
Q_cp, U_cp, V_cp = IPs.Stokes_calculate(nu_cp)

# hot
n0L, T, B, theta_B = 10**21.940784429414933, 10**11.73896501100829, 10**(-2.476278011242103), 65.7711004245088
P, beta0, chi0 = 0.9973851508392868, 56.71139513661876, 28.89873359547977
I0, Q0, U0, V0 = 1, P*np.cos(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*chi0/180*np.pi)
IPs.parameters_update(n0L, T, B, theta_B, I0, Q0, U0, V0)
nu_hp = np.linspace(1.2, 1.6, 1000)*1e9
Q_hp, U_hp, V_hp = IPs.Stokes_calculate(nu_hp)

fig, axs = plt.subplots(3, 1, sharex=True)
# Remove vertical space between Axes
fig.subplots_adjust(hspace=0)
fontsize=12
# Plot each graph, and manually set the y tick values
axs[0].errorbar(nu_GFR/1e9, Q_GFR, yerr=0.05, fmt='r.')
axs[0].plot(nu_cp/1e9, Q_cp)
axs[0].plot(nu_hp/1e9, Q_hp)
axs[0].set_yticks([-0.3, -0.1, 0.1, 0.3])
# axs[0].set_ylim(-1.05, 1.05)
axs[0].set_ylabel(r'$Q/I$', fontsize=fontsize)

axs[1].errorbar(nu_GFR/1e9, U_GFR, yerr=0.05, fmt='r.')
axs[1].plot(nu_cp/1e9, U_cp)
axs[1].plot(nu_hp/1e9, U_hp)
axs[1].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
# axs[1].set_ylim(-1.05, 1.05)
axs[1].set_ylabel(r'$U/I$', fontsize=fontsize)

axs[2].errorbar(nu_GFR/1e9, V_GFR, yerr=0.05, fmt='r.')
axs[2].plot(nu_cp/1e9, V_cp)
axs[2].plot(nu_hp/1e9, V_hp)
axs[2].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
axs[2].set_ylabel(r'$V/I$', fontsize=fontsize)
axs[2].set_xlabel(r'$\nu ~(\mathrm{GHz})$', fontsize=fontsize)
plt.show()

# plt.savefig('./GFR_ColdandHot1.2-1.6.pdf', dpi=100, bbox_inches='tight')







