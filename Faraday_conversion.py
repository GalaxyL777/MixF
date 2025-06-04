#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 10:48:07 2025

@author: xiaohuiliu

Investigate the propagation effects of the polarized wave

"""

import bilby
import numpy as np

from scipy.special import kv, erf
from scipy.constants import pi                                                 # Mathematical constants
from scipy.constants import c, e, m_e, k, parsec                               # Physical constants: speed of light (m s^-1), elementary charge (C), electron mass (kg), Boltzmann constant (J K^-1)
speed_of_light_movers = c
SIcgs_c, SIcgs_e, SIcgs_me, SIcgs_k = 1e2, 1/3.335640952e-10, 1e3, 1e7
c, e, m_e, k = c*SIcgs_c, e*SIcgs_e, m_e*SIcgs_me, k*SIcgs_k # Gauss units (cm s^-1, esu, g, 1)

#%% main module

class GFR_class(object):
    '''
    The Generalized Faraday Rotation (GFR) is a phenomenological model, which
    depicts the motion of spectrum in the Poincare sphere.
    '''
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



def Rp_from_RMCM(f, RM, CM, chi_p):
    l = speed_of_light_movers / f # m = */Hz
    Rv = 2 * RM * l**2 * 1       # 1 represents kv_ratio(0,2,rho) * Function_g(X)
    Rq = - CM * l**3 * 1         # 1 represents (kv_ratio(1,2,rho)+6/rho) * Function_f(X)
    # rotate R
    RQ, RU, RV = Rq*np.cos(2*chi_p), -Rq*np.sin(2*chi_p), Rv
    return [RQ, RU, RV]

def Rp_from_Bn0L(f, B, n0L, theta_B, T, chi_p):
    # calculate the cyclotron frequency
    omega_B = e*B/(m_e*c)
    nu_B = omega_B/2/pi                                                    # Hz
    rho = m_e*c**2/(k*T)
    X = 10**(3/2) * 2**(1/4) / rho * (nu_B/f*np.sin(theta_B))**(1/2)
    # RV = rho_V * L, RQ = rho_Q * L
    Rv = np.power(omega_B/f,2) * e/B * n0L * np.cos(theta_B)/pi * kv_ratio(0,2,rho) * Function_g(X)
    Rq = -np.power(omega_B/f,3) * e/B *n0L * np.power(np.sin(theta_B),2)/(4*pi*pi) * (kv_ratio(1,2,rho)+6/rho) * Function_f(X)
    #rotate R
    RQ, RU, RV = Rq*np.cos(2*chi_p), -Rq*np.sin(2*chi_p), Rv
    return [RQ, RU, RV]

def erp_from_Bn0L(f, B, n0, L, theta_B, T, chi_p):
    # calculate the cyclotron frequency
    omega_B = e*B/(m_e*c)
    nu_B = omega_B/2/pi                                                    # Hz
    
    rho = m_e*c**2/(k*T)
    X = 10**(3/2) * 2**(1/4) / rho * (nu_B/f*np.sin(theta_B))**(1/2)
    
    # Faraday rotation and conversion coefficients in the etaU = rhoU = 0 system
    eta_I = 8/3/np.sqrt(2*pi) * np.power(e,6)*np.power(n0,2)/np.power(k*T*m_e,3/2)/c/np.power(f,2) * np.log( np.power(2*k*T, 3/2)/(4.2*pi*np.power(e,2)*np.power(m_e,1/2)*f) )
    eta_Q = 3/8/pi**2 * np.power(omega_B*np.sin(theta_B)/f,2) * eta_I
    eta_V = -omega_B*np.cos(theta_B)/pi/f * eta_I
    
    rho_V = np.power(omega_B/f,2) * e/B *n0 * np.cos(theta_B)/pi * kv_ratio(0,2,rho) * Function_g(X)
    rho_Q = -np.power(omega_B/f,3) * e/B *n0 * np.power(np.sin(theta_B),2)/(4*pi*pi) * (kv_ratio(1,2,rho)+6/rho) * Function_f(X)
    
        #rotate M to M'
    eta_Ip, eta_Qp, eta_Up, eta_Vp = eta_I, eta_Q*np.cos(2*chi_p), -eta_Q*np.sin(2*chi_p), eta_V
    rho_Qp, rho_Up, rho_Vp = rho_Q*np.cos(2*chi_p), -rho_Q*np.sin(2*chi_p), rho_V
    
    return [eta_Ip, eta_Qp, eta_Up, eta_Vp], [rho_Qp, rho_Up, rho_Vp]




def solution_Matrix_noabsorption(Rp, P_input):
    RQ, RU, RV = Rp
    Qi, Ui, Vi = P_input
    
    zetas2 = np.power(RQ,2) + np.power(RU,2) + np.power(RV,2)
    kappas = np.sqrt(zetas2)
    k2_ks2 = 0 + np.power(kappas,2)
    
    q2, u2, v2 = (0+np.power(RQ,2))/k2_ks2, (0+np.power(RU,2))/k2_ks2, (0+np.power(RV,2))/k2_ks2
    
    qdotu = (0 + RQ*RU)/k2_ks2
    udotv = (0 + RU*RV)/k2_ks2
    vdotq = (0 + RV*RQ)/k2_ks2
    
    qdotk = (0 + RQ*kappas)/k2_ks2
    udotk = (0 + RU*kappas)/k2_ks2
    vdotk = (0 + RV*kappas)/k2_ks2
    
    udotq = qdotu
    vdotu = udotv
    qdotv = vdotq
    kdotq = qdotk
    kdotu = udotk
    kdotv = vdotk
    
    # kappas * L
    ksL = np.sqrt(RQ**2+RU**2+RV**2)
    
    # calculate the elements of matrix M
    '''
        | 1  0  0  0  |
    M = | 0           | 
        | 0           |
        | 0           |
    '''
    M22 = 0.5*(1+q2-u2-v2)*1                                              +0.5*(1-q2+u2+v2)*np.cos(ksL)
    M23 = qdotu*1                          +0                             -qdotu*np.cos(ksL)               -vdotk*np.sin(ksL)
    M24 = qdotv*1                          -0                             -qdotv*np.cos(ksL)               +udotk*np.sin(ksL)
    
    M32 = qdotu*1                          -0                             -qdotu*np.cos(ksL)               +vdotk*np.sin(ksL)
    M33 = 0.5*(1-q2+u2-v2)*1                                              +0.5*(1+q2-u2+v2)*np.cos(ksL)
    M34 = udotv*1                          +0                             -udotv*np.cos(ksL)               -qdotk*np.sin(ksL)
    
    M42 = qdotv*1                          +0                             -qdotv*np.cos(ksL)               -udotk*np.sin(ksL)
    M43 = udotv*1                          -0                             -udotv*np.cos(ksL)               +qdotk*np.sin(ksL)
    M44 = 0.5*(1-q2-u2+v2)*1                                              +0.5*(1+q2+u2-v2)*np.cos(ksL)
    
    Qo = (M22*Qi + M23*Ui + M24*Vi)
    Uo = (M32*Qi + M33*Ui + M34*Vi)
    Vo = (M42*Qi + M43*Ui + M44*Vi)
    
    return [Qo, Uo, Vo] # P_output

def solution_Matrix_withabsorption(eta_p, rho_p, L, IQUV_input): # a full version
    eta_Ip, eta_Qp, eta_Up, eta_Vp = eta_p
    rho_Qp, rho_Up, rho_Vp = rho_p
    Ii, Qi, Ui, Vi = IQUV_input
    
    tao = eta_Ip*L
    
    # preparations for solutions
    zeta2 = np.power(eta_Qp,2) + np.power(eta_Up,2) + np.power(eta_Vp,2)
    zetas2 = np.power(rho_Qp,2) + np.power(rho_Up,2) + np.power(rho_Vp,2)
        
    kappa = 1/np.sqrt(2) * np.sqrt( np.sqrt( np.power(zeta2-zetas2,2)+4*zeta2*zetas2 ) + zeta2-zetas2 )
    kappas = 1/np.sqrt(2) * np.sqrt( np.sqrt( np.power(zeta2-zetas2,2)+4*zeta2*zetas2 ) - zeta2+zetas2 )
        
    k2_ks2 = np.power(kappa,2) + np.power(kappas,2)
    
    q2, u2, v2 = (np.power(eta_Qp,2)+np.power(rho_Qp,2))/k2_ks2, (np.power(eta_Up,2)+np.power(rho_Up,2))/k2_ks2, (np.power(eta_Vp,2)+np.power(rho_Vp,2))/k2_ks2
    
    # icrossj = (eta_i*rho_j - rho_i*eta_j)/k2_ks2
    qcrossu = (eta_Qp*rho_Up - rho_Qp*eta_Up)/k2_ks2
    ucrossv = (eta_Up*rho_Vp - rho_Up*eta_Vp)/k2_ks2
    vcrossq = (eta_Vp*rho_Qp - rho_Vp*eta_Qp)/k2_ks2
    
    qcrossk = (eta_Qp*kappas - rho_Qp*kappa)/k2_ks2
    ucrossk = (eta_Up*kappas - rho_Up*kappa)/k2_ks2
    vcrossk = (eta_Vp*kappas - rho_Vp*kappa)/k2_ks2
    
    ucrossq = - qcrossu
    vcrossu = - ucrossv
    qcrossv = - vcrossq
    kcrossq = - qcrossk
    kcrossu = - ucrossk
    kcrossv = - vcrossk
    
    qdotu = (eta_Qp*eta_Up + rho_Qp*rho_Up)/k2_ks2
    udotv = (eta_Up*eta_Vp + rho_Up*rho_Vp)/k2_ks2
    vdotq = (eta_Vp*eta_Qp + rho_Vp*rho_Qp)/k2_ks2
    
    qdotk = (eta_Qp*kappa + rho_Qp*kappas)/k2_ks2
    udotk = (eta_Up*kappa + rho_Up*kappas)/k2_ks2
    vdotk = (eta_Vp*kappa + rho_Vp*kappas)/k2_ks2
    
    udotq = qdotu
    vdotu = udotv
    qdotv = vdotq
    kdotq = qdotk
    kdotu = udotk
    kdotv = vdotk
    
    # calculate the elements of matrix M
    M11 =  0.5*(1+q2+u2+v2)*np.cosh(kappa*L)                    +0.5*(1-q2-u2-v2)*np.cos(kappas*L)
    M12 = -ucrossv*np.cosh(kappa*L)   -qdotk*np.sinh(kappa*L)   -vcrossu*np.cos(kappas*L)   -qcrossk*np.sin(kappas*L)
    M13 = -vcrossq*np.cosh(kappa*L)   -udotk*np.sinh(kappa*L)   -qcrossv*np.cos(kappas*L)   -ucrossk*np.sin(kappas*L)
    M14 = -qcrossu*np.cosh(kappa*L)   -vdotk*np.sinh(kappa*L)   -ucrossq*np.cos(kappas*L)   -vcrossk*np.sin(kappas*L)
    
    M21 =  ucrossv*np.cosh(kappa*L)   -qdotk*np.sinh(kappa*L)   +vcrossu*np.cos(kappas*L)   -qcrossk*np.sin(kappas*L)
    M22 = 0.5*(1+q2-u2-v2)*np.cosh(kappa*L)                     +0.5*(1-q2+u2+v2)*np.cos(kappas*L)
    M23 = qdotu*np.cosh(kappa*L)      +vcrossk*np.sinh(kappa*L) -qdotu*np.cos(kappas*L)     -vdotk*np.sin(kappas*L)
    M24 = qdotv*np.cosh(kappa*L)      -ucrossk*np.sinh(kappa*L) -qdotv*np.cos(kappas*L)     +udotk*np.sin(kappas*L)
    
    M31 = vcrossq*np.cosh(kappa*L)    -udotk*np.sinh(kappa*L)   +qcrossv*np.cos(kappas*L)   -ucrossk*np.sin(kappas*L)
    M32 = qdotu*np.cosh(kappa*L)      -vcrossk*np.sinh(kappa*L) -qdotu*np.cos(kappas*L)     +vdotk*np.sin(kappas*L)
    M33 = 0.5*(1-q2+u2-v2)*np.cosh(kappa*L)                     +0.5*(1+q2-u2+v2)*np.cos(kappas*L)
    M34 = udotv*np.cosh(kappa*L)      +qcrossk*np.sinh(kappa*L) -udotv*np.cos(kappas*L)     -qdotk*np.sin(kappas*L)
    
    M41 = qcrossu*np.cosh(kappa*L)    -vdotk*np.sinh(kappa*L)   +ucrossq*np.sin(kappas*L)   -vcrossk*np.sin(kappas*L)
    M42 = qdotv*np.cosh(kappa*L)      +ucrossk*np.sinh(kappa*L) -qdotv*np.cos(kappa*L)      -udotk*np.sin(kappas*L)
    M43 = udotv*np.cosh(kappa*L)      -qcrossk*np.sinh(kappa*L) -udotv*np.cos(kappas*L)     +qdotk*np.sin(kappas*L)
    M44 = 0.5*(1-q2-u2+v2)*np.cosh(kappa*L)                     +0.5*(1+q2+u2-v2)*np.cos(kappas*L)
    
    Io = (M11*Ii + M12*Qi + M13*Ui + M14*Vi)*np.exp(-tao)
    Qo = (M21*Ii + M22*Qi + M23*Ui + M24*Vi)*np.exp(-tao)
    Uo = (M31*Ii + M32*Qi + M33*Ui + M34*Vi)*np.exp(-tao)
    Vo = (M41*Ii + M42*Qi + M43*Ui + M44*Vi)*np.exp(-tao)
    return Io, Qo, Uo, Vo # IQUV_output



def Faraday_rotation(L, RM, f):
    Q, U = L
    
    RMobs_times_2_times_lambda2 = 2*RM*(speed_of_light_movers/f)**2
    Qp = Q*np.cos(RMobs_times_2_times_lambda2) - U*np.sin(RMobs_times_2_times_lambda2)
    Up = Q*np.sin(RMobs_times_2_times_lambda2) + U*np.cos(RMobs_times_2_times_lambda2)
    return [Qp, Up]

 

class Cold_class(object):
    '''
    This is the simple version of the analytical solution of the transfer of 
    polarization of FRB as a strong incoming wave propagating in a
    magnetized plasma.
    
    In this solution, we neglect the absorption terms and set all cross terms
    to be zero and simplifies reality to three plasma layers
    (FRB --> background layer --> mixing layer --> foreground layer).
    
    T (K) only matters in the hot scenario, so T = 0 K is employed in the cold scenario.
    '''
    def __init__(self):
        self.RM = 0 # rad m^-2
        self.CM = 0 # rad m^-3
        self.I0, self.Q0, self.U0, self.V0 = 1, 1, 1, 1                        # the initial Stokes angles of the incoming wave (indeed no units)
        self.chi_p = 0                                                         # the angle (rad) between the rho_U=eta_U = 0 frame and the observer frame
        self.RM_b, self.RM_f = 0, 0
        
    def parameters_update(self, RM, CM, I0, Q0, U0, V0, chi_p, RM_b, RM_f):
        # the input RM and CM are in units of rad m^-2 and rad m^-3
        self.RM = RM
        self.CM = CM
        
        self.I0, self.Q0, self.U0, self.V0 = I0, Q0, U0, V0                    # intrinsic Stokes parameters in the coordinate of U=0
        self.chi_p = chi_p                                                     # rad
        
        self.RM_b = RM_b
        self.RM_f = RM_f
    
    def Stokes_calculate(self, f): # f is in units of Hz
        # intrinsic Stokes parameters in the coordinate of observer
        P_1 = [self.Q0*np.cos(2*self.chi_p)+self.U0*np.sin(2*self.chi_p), -self.Q0*np.sin(2*self.chi_p)+self.U0*np.cos(2*self.chi_p), self.V0]
        
        # the background Faraday rotation medium
        L_input = Faraday_rotation([P_1[0], P_1[1]], self.RM_b, f)
        P_input = [L_input[0], L_input[1], P_1[2]]
        
        # mixing medium
        Rp = Rp_from_RMCM(f, self.RM, self.CM, self.chi_p)
        P_output = solution_Matrix_noabsorption(Rp, P_input)
        
        # the foreground Faraday rotation medium
        L_observed = Faraday_rotation([P_output[0], P_output[1]], self.RM_f, f)
        P_observed = [L_observed[0], L_observed[1], P_output[2]]
        
        return P_observed



class Hot_class(object):
    '''
    In principle, this class can be also used to a cold plasma scenario.
    '''
    def __init__(self):
        # basic parameters
        self.B = 1                                                             # magnetic field (G)
        self.n0L = 1 * 10**4                                                   # number density (cm^(-3)) * typical size of medium (cm)
        self.theta_B = 70/180*np.pi                                            # angle (rad) between the magnetic field B and wave vector
        self.T = 100                                                           # temperature of medium (K)

        self.chi_p = 0                                                         # the angle (rad) between the rho_U=eta_U = 0 frame and the observer frame
                
        self.I0, self.Q0, self.U0, self.V0 = 1, 1, 1, 1                        # the initial Stokes angles of the incoming wave (indeed no units)
        
        self.RM_b = 0 # 2*RM*lambda**2 = rho_V * L
        self.RM_f = 0
    
    def parameters_update(self, B, n0L, theta_B, T, chi_p, I0, Q0, U0, V0, RM_b, RM_f):
        
        # print('parameters:', B, n0L, theta_B, T, chi_p, I0, Q0, U0, V0, RM_b, RM_f)
        # print('')
        
        self.B = B
        self.n0L = n0L
        self.theta_B = theta_B                                                 # the input theta_B is in units of rad
        self.T = T
        self.chi_p = chi_p                                                     # the input chi_p is in units of rad
        self.I0, self.Q0, self.U0, self.V0 = I0, Q0, U0, V0
        self.RM_b = RM_b                                                 # rad m^-2
        self.RM_f = RM_f                                                 # rad m^-2

    def Stokes_calculate(self, f):  # f is in units of Hz
        # intrinsic Stokes parameters in the coordinate of observer
        P_1 = [self.Q0*np.cos(2*self.chi_p)+self.U0*np.sin(2*self.chi_p), -self.Q0*np.sin(2*self.chi_p)+self.U0*np.cos(2*self.chi_p), self.V0]
        
        # the background Faraday rotation medium
        L_input = Faraday_rotation([P_1[0], P_1[1]], self.RM_b, f)
        P_input = [L_input[0], L_input[1], P_1[2]]
        
        # mixing medium
        Rp = Rp_from_Bn0L(f, self.B, self.n0L, self.theta_B, self.T, self.chi_p)
        P_output = solution_Matrix_noabsorption(Rp, P_input)
        
        # the foreground Faraday rotation medium
        L_observed = Faraday_rotation([P_output[0], P_output[1]], self.RM_f, f)
        P_observed = [L_observed[0], L_observed[1], P_output[2]]
        
        # print('P_1:', P_1)
        # print('P_input', P_input)
        # print('P_output', P_output)
        # print('P_observed:', P_observed)
        
        # import matplotlib.pyplot as plt
        
        # plt.figure()
        # plt.plot(f, P_input[0], label = 'Q')
        # plt.plot(f, P_input[1], label = 'U')
        # plt.legend()
        # plt.title('P_input (RM foreground)')
        # plt.show()
        
        # plt.figure()
        # plt.plot(f, P_output[0], label='Q')
        # plt.plot(f, P_output[1], label='U')
        # plt.plot(f, P_output[2], label='V')
        # plt.title('P_input (mixing)')
        # plt.show()
        
        # plt.figure()
        # plt.plot(f, P_observed[0], label='Q')
        # plt.plot(f, P_observed[1], label='U')
        # plt.plot(f, P_observed[2], label='V')
        # plt.title('P_input (observed)')
        # plt.show()
        return P_observed
    

class Absorption_class(object):
    '''
    This is the full version of our model.
    '''
    def __init__(self):
        # basic parameters
        self.B = 1                                                             # magnetic field (G)
        self.n0 = 1                                                            # number density (cm^(-3))
        self.L = 10**4                                                         # typical size of medium (cm)
        self.theta_B = 70/180*np.pi                                            # angle (rad) between the magnetic field B and wave vector
        self.T = 100                                                           # temperature of medium (K)

        self.chi_p = 0                                                         # the angle (rad) between the rho_U=eta_U = 0 frame and the observer frame
        
        self.I0, self.Q0, self.U0, self.V0 = 1, 1, 1, 1                        # the initial Stokes angles of the incoming wave (indeed no units)

        self.RM_b = 0            # 2*RM*lambda**2 = rho_V * L
        self.RM_f = 0
        
    def parameters_update(self, B, n0, L, theta_B, T, chi_p, I0, Q0, U0, V0, RM_b, RM_f):
        self.B = B
        self.n0 = n0
        self.L = L
        self.theta_B = theta_B                                                 # the input theta_B is in units of rad
        self.T = T
        
        self.chi_p = chi_p                                                     # the input chi is in units of rad
        
        self.I0, self.Q0, self.U0, self.V0 = I0, Q0, U0, V0
       
        self.RM_b = RM_b                                                # rad m^-2
        self.RM_f = RM_f
    
    def Stokes_calculate(self, f): # f is in units of Hz
        # intrinsic Stokes parameters in the coordinate of observer
        IQUV_1 = [self.I0, self.Q0*np.cos(2*self.chi_p)+self.U0*np.sin(2*self.chi_p), -self.Q0*np.sin(2*self.chi_p)+self.U0*np.cos(2*self.chi_p), self.V0]
        
        # the background Faraday rotation medium
        L_input = Faraday_rotation([IQUV_1[1], IQUV_1[2]], self.RM_b, f)
        IQUV_input = [IQUV_1[0], L_input[0], L_input[1], IQUV_1[3]]
        
        # mixing medium
        eta_p, rho_p = erp_from_Bn0L(f, self.B, self.n0, self.L, self.theta_B, self.T, self.chi_p)
        IQUV_output = solution_Matrix_withabsorption(eta_p, rho_p, self.L, IQUV_input)
        
        # the foreground Faraday rotation medium
        L_observed = Faraday_rotation([IQUV_output[1], IQUV_output[2]], self.RM_f, f)
        IQUV_observed = [IQUV_output[0], L_observed[0], L_observed[1], IQUV_output[3]]
        
        return IQUV_observed



class QUVcold_likelihood(bilby.Likelihood):
    def __init__(self, nu, Q, U, V, Q_err, U_err, V_err):
        '''
        A simple likelihood to fit the Cold model with QUV.
        '''
        super().__init__(parameters={'RM': None, 'CM': None, 'chi_p':None,
                                      'beta_0': None, 'chi_0': None,
                                      'RM_b': None, 'RM_f': None})
        # n0L, T, B, theta_B, chi, I0, Q0, U0, V0, RM
        self.nu = nu
        self.Q = Q
        self.U = U
        self.V = V
        self.Q_err = Q_err
        self.U_err = U_err
        self.V_err = V_err
        
        self.Model = Cold_class()
    
    def update(self, RM, CM, chi_p, beta_0, chi_0, RM_b, RM_f):
        self.parameters['RM'] = RM
        self.parameters['CM']  = CM
        
        self.parameters['chi_p']  = chi_p                                      # degree
        self.parameters['beta_0']  = beta_0
        self.parameters['chi_0']  = chi_0
        
        self.parameters['RM_b'] = RM_b
        self.parameters['RM_f'] = RM_f
    
    def log_likelihood(self):
        # all degrees are transformed to rad in likelihood.log_likelihood function
        # get the parameters
        RM = self.parameters['RM']
        CM = self.parameters['CM']
        
        chi_p = self.parameters['chi_p']/180*np.pi                             # rad
        beta_0 = self.parameters['beta_0']/180*np.pi
        chi_0 = self.parameters['chi_0']/180*np.pi
        
        RM_b = self.parameters['RM_b']
        RM_f = self.parameters['RM_f']
        
        I0 = 1
        Q0, U0, V0 = np.cos(2*beta_0)*np.cos(2*chi_0), np.sin(2*beta_0)*np.cos(2*chi_0), np.sin(2*chi_0)

        self.Model.parameters_update(RM, CM, I0, Q0, U0, V0, chi_p, RM_b, RM_f)
        Qm, Um, Vm = self.Model.Stokes_calculate(self.nu)
            
        # log_likelihood
        ll_Q = -0.5*np.sum( (self.Q-Qm)**2/self.Q_err**2 )
        ll_U = -0.5*np.sum( (self.U-Um)**2/self.U_err**2 )
        ll_V = -0.5*np.sum( (self.V-Vm)**2/self.V_err**2 )
        log_likelihood = ll_Q + ll_U + ll_V
        return log_likelihood

class QUVhot_likelihood(bilby.Likelihood):
    def __init__(self, nu, Q, U, V, Q_err, U_err, V_err):
        '''
        A simple likelihood to fit the Hot model with QUV.
        '''
        super().__init__(parameters={'B': None, 'n0L': None, 'theta_B':None,
                                      'T': None, 'chi_p': None,
                                      'beta_0': None, 'chi_0': None,
                                      'RM_b': None, 'RM_f': None})
        # B, n0L, theta_B, T, chi_p, I0, Q0, U0, V0, RM_back, RM_fore
        self.nu = nu
        self.Q = Q
        self.U = U
        self.V = V
        self.Q_err = Q_err
        self.U_err = U_err
        self.V_err = V_err
        
        self.Model = Hot_class()
    
    def update(self, B, n0L, theta_B, T, chi_p, beta_0, chi_0, RM_b, RM_f):
        self.parameters['B'] = B
        self.parameters['n0L']  = n0L
        self.parameters['theta_B']  = theta_B
        self.parameters['T']  = T
        
        self.parameters['chi_p']  = chi_p
    
        self.parameters['beta_0']  = beta_0
        self.parameters['chi_0']  = chi_0
        
        self.parameters['RM_b'] = RM_b
        self.parameters['RM_f'] = RM_f
    
    def log_likelihood(self):
        # get the parameters
        B = self.parameters['B']
        n0L = self.parameters['n0L']
        theta_B = self.parameters['theta_B']/180*np.pi
        T = self.parameters['T']
        
        chi_p = self.parameters['chi_p']/180*np.pi
        
        beta_0 = self.parameters['beta_0']/180*np.pi
        chi_0 = self.parameters['chi_0']/180*np.pi
        
        RM_b = self.parameters['RM_b']
        RM_f = self.parameters['RM_f']
        
        I0 = 1
        Q0, U0, V0 = np.cos(2*beta_0)*np.cos(2*chi_0), np.sin(2*beta_0)*np.cos(2*chi_0), np.sin(2*chi_0)

        self.Model.parameters_update(B, n0L, theta_B, T, chi_p, I0, Q0, U0, V0, RM_b, RM_f)
        Qm, Um, Vm = self.Model.Stokes_calculate(self.nu)
        
        # log_likelihood
        ll_Q = -0.5*np.sum( (self.Q-Qm)**2/self.Q_err**2 )
        ll_U = -0.5*np.sum( (self.U-Um)**2/self.U_err**2 )
        ll_V = -0.5*np.sum( (self.V-Vm)**2/self.V_err**2 )
        log_likelihood = ll_Q + ll_U + ll_V
        return log_likelihood

class QUVabsorption_likelihood(bilby.Likelihood):
    def __init__(self, nu, Q, U, V, Q_err, U_err, V_err):
        '''
        A likelihood to fit the Absorption model with QUV.
        '''
        super().__init__(parameters={'B': None, 'n0': None, 'L': None, 'theta_B':None, 'T': None,
                                     'chi_p': None, 'beta_0': None, 'chi_0': None,
                                     'RM_b': None, 'RM_f': None})
        # B, n0L, theta_B, T, chi_p, I0, Q0, U0, V0, RM_back, RM_fore
        self.nu = nu
        self.Q = Q
        self.U = U
        self.V = V
        self.Q_err = Q_err
        self.U_err = U_err
        self.V_err = V_err
        
        self.Model = Absorption_class()
    
    def update(self, B, n0, L, theta_B, T, chi_p, beta_0, chi_0, RM_b, RM_f):
        self.parameters['B'] = B
        self.parameters['n0']  = n0
        self.parameters['L']  = L
        self.parameters['theta_B']  = theta_B
        self.parameters['T']  = T
        
        self.parameters['chi_p']  = chi_p
        self.parameters['beta_0']  = beta_0
        self.parameters['chi_0']  = chi_0
        
        self.parameters['RM_b'] = RM_b
        self.parameters['RM_f'] = RM_f
    
    def log_likelihood(self):
        # get the parameters
        B = self.parameters['B']
        n0 = self.parameters['n0']
        L = self.parameters['L']
        theta_B = self.parameters['theta_B']/180*np.pi
        T = self.parameters['T']
        
        chi_p = self.parameters['chi_p']/180*np.pi
        
        beta_0 = self.parameters['beta_0']/180*np.pi
        chi_0 = self.parameters['chi_0']/180*np.pi
        
        RM_b = self.parameters['RM_b']
        RM_f = self.parameters['RM_f']
        
        I0 = 1
        Q0, U0, V0 = np.cos(2*beta_0)*np.cos(2*chi_0), np.sin(2*beta_0)*np.cos(2*chi_0), np.sin(2*chi_0)

        self.Model.parameters_update(B, n0, L, theta_B, T, chi_p, I0, Q0, U0, V0, RM_b, RM_f)
        Im, Qm, Um, Vm = self.Model.Stokes_calculate(self.nu)
        
        # log_likelihood
        ll_Q = -0.5*np.sum( (self.Q-Qm)**2/self.Q_err**2 )
        ll_U = -0.5*np.sum( (self.U-Um)**2/self.U_err**2 )
        ll_V = -0.5*np.sum( (self.V-Vm)**2/self.V_err**2 )
        log_likelihood = ll_Q + ll_U + ll_V
        return log_likelihood


#%% basic functions
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

#%% a simple fitting example of the mock data of FRB 20180301A using emcee
# import emcee

# # use GFR model to generate a mock data of FRB 20180301A
# GFR = GFR_class()
# RM, psi0, theta, phi, GRM, Psi0, alpha, chi = 27.7, -87.3, 104.2, 76.3, 4351.7, 0, 2.3, -0.1 
# GFR.parameters_update(RM, psi0, theta, phi, GRM, Psi0, alpha, chi)

# nu = np.linspace(1.370, 1.425, 50)*1e9
# Q,U,V = GFR.Stokes_calculate(nu)
# Q_err,U_err,V_err = Q*0+0.1,Q*0+0.1,Q*0+0.1

# # construct the Cold QUV likelihood
# likelihood = QUVhot_likelihood(nu, Q, U, V, Q_err, U_err, V_err)

# # construct the prior and likelihood function by imitating the tutorial of emcee
# def log_prior(theta):
#     log10n0L, log10B, thetaB, chi, beta0, chi0, RM_b, RM_f = theta
#     # set priors
#     if 5<log10n0L<25 and -1<log10B<10 and 0<thetaB<180 and -180<chi<180 and -90<beta0<90 and -90<chi0<90 and -200<RM_b<200 and -200<RM_f<200:
#         return 0.0
#     return -np.inf

# def log_likelihood(theta):
#     log10n0L, log10B, thetaB, chi, beta0, chi0, RM_b, RM_f = theta
    
#     lp = log_prior(theta)
#     if not np.isfinite(lp):
#         return -np.inf
    
#     log10T = 0 # T = 1 K for cold scenario
#     likelihood.update(10**log10B, 10**log10n0L, thetaB, 10**log10T, chi,
#                       beta0, chi0,
#                       RM_b, RM_f)
#     ll = likelihood.log_likelihood()
#     # print(ll*(-0.5), lp)
#     return lp + ll

# pos = [12.615911853191527, 3.16674240634451, 109.68500699641106, 84.53922144526928, -39.40913431115206, -4.890144138799174, -1.6045660430449313, 32.16482377117103] + np.random.randn(32,8)*1e-4
# nwalkers, ndim = pos.shape

# sampler = emcee.EnsembleSampler(
#     nwalkers, ndim, log_likelihood
# )
# sampler.run_mcmc(pos, 500000, progress=True)

# # tau = sampler.get_autocorr_time()
# # print(tau)

# samples = sampler.get_chain(discard=1000, thin=15, flat=True)
# print(samples.shape)


# # show the conner results
# import corner

# fig = corner.corner(
#     samples
# )





#%% a simple example of the mock data of FRB 20180301A
# # GFR test
# GFR = GFR_class()
# RM, psi0, theta, phi, GRM, Psi0, alpha, chi = 27.7, -87.3, 104.2, 76.3, 4351.7, 0, 2.3, -0.1 
# GFR.parameters_update(RM, psi0, theta, phi, GRM, Psi0, alpha, chi)

# nu1 = np.linspace(1.370, 1.425, 50)*1e9
# Q1,U1,V1 = GFR.Stokes_calculate(nu1)
# Q1err,U1err,V1err = Q1*0+0.05,Q1*0+0.05,Q1*0+0.05

# from qcosmc.MCMC import MCMC_class, MCplot

# likelihood = QUVhot_likelihood(nu1,Q1,U1,V1,Q1err,U1err,V1err)

# ### check cold plasma scenario
# parameters_best = []
# def chi2(theta):
#     log10n0L, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore = theta
#     log10T = 0
#     likelihood.update(10**log10B, 10**log10n0L, thetaB, 10**log10T, chi,
#                         beta0, chi0,
#                         RM_back, RM_fore)
#     chi2 = likelihood.log_likelihood()/(-0.5)
#     # print(chi2)
    
#     if chi2 < 15:
#         parameters_best.append( [log10n0L, log10T, log10B, thetaB, chi, beta0*180/np.pi, chi0*180/np.pi, RM_back, RM_fore, chi2])
#     return chi2

# params = [[r'\log_{10}(n_{0}L/ \mathrm{cm}^{-2})', 12.615911853191527, 5, 25],
#           [r'\log_{10}(B / \mathrm{G})', 3.16674240634451, -1, 10],
#           [r'\theta_{B} (\mathrm{deg})', 109.68500699641106, 0, 180],
#           [r'\chi_{p} (\mathrm{deg})', 84.53922144526928, -180, 180],
#           [r'\beta_0 (\mathrm{deg})', -39.40913431115206, -90, 90],
#           [r'\chi_0 (\mathrm{deg})', -4.890144138799174, -45, 90],
#           [r'\mathrm{RM_{b}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', -1.6045660430449313, -1000, 1000],
#           [r'\mathrm{RM_{f}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', 32.16482377117103, -1000, 1000]
#           ]


# MC = MCMC_class(params, chi2, 'Cold_180301A')
# # MC.MCMC(100000)

# chains = [
#           ['Cold_180301A', 'FRB 20180301A (cold)']
#           ]
# pl = MCplot(chains)
# pl.plot3D([1,2,3,4,5,6,7,8])
# pl.results2
# pl.results

# A = np.array(parameters_best)
# np.save('./chains/Cold_180301A_listchi2.npy', A)

# ### check hot plasma scenario
# parameters_best = []
# def chi2(theta):
#     log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore = theta
#     likelihood.update(10**log10B, 10**log10n0L, thetaB, 10**log10T, chi,
#                       beta0, chi0,
#                       RM_back, RM_fore)
#     chi2 = likelihood.log_likelihood()/(-0.5)
#     print(chi2)
#     if chi2 < 27.21:
#         parameters_best.append( [log10n0L, log10T, log10B, thetaB, chi, beta0*180/np.pi, chi0*180/np.pi, RM_back, RM_fore, chi2])
#     return chi2

# params = [[r'\log_{10}(n_{0}L/ \mathrm{cm}^{-2})', 22.79276766773475, 18, 30],
#           [r'\log_{10}(T / \mathrm{K})', 12.043468338027928, 8, 18],
#           [r'\log_{10}(B / \mathrm{G})', -2.906970787412113, -5, -1],
#           [r'\theta_{B} (\mathrm{deg})', 130.78483889392834, 0, 180],
#           [r'\chi_{p} (\mathrm{deg})', 69.11720697460022, -180, 180],
#           [r'\beta_0 (\mathrm{deg})', 11.0218349517697, -90, 90*3],
#           [r'\chi_0 (\mathrm{deg})', -36.215141988916315, -90, 45],
#           [r'\mathrm{RM_{b}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', 0.0030798962397969327, -100, 100],
#           [r'\mathrm{RM_{f}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', 26.32701863495556, -100, 100]
#           ]

# MC = MCMC_class(params, chi2, 'Hot_180301A')
# MC.MCMC(100000)

# chains = [
#           ['Hot_180301A', 'FRB 20180301A (hot)']
#           ]
# pl = MCplot(chains)
# pl.plot3D([1,2,3,4,5,6,7,8,9])
# pl.results2
# pl.results

# A = np.array(parameters_best)
# np.save('./chains/Hot_180301A_listchi2.npy', A)


# Hc = Hot_class()

# ### GFR cold and hot plasma
# nu_GFR = np.linspace(1.370, 1.425, 50)*1e9
# Q_GFR,U_GFR,V_GFR = GFR.Stokes_calculate(nu_GFR)

# # cold
# A = np.load('./chains/Cold_180301A_listchi2.npy')
# log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_b, RM_f, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]
# print('Cold:', chi2_min)
# n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
# P = 1
# I0, Q0, U0, V0 = 1, P*np.cos(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*chi0/180*np.pi)

# Hc.parameters_update(B, n0L, thetaB, T, chi, I0, Q0, U0, V0, RM_b, RM_f)
# nu_cp = np.linspace(1.3, 1.5, 1000)*1e9
# Q_cp, U_cp, V_cp = Hc.Stokes_calculate(nu_cp)


# # hot
# A = np.load('./chains/Hot_180301A_listchi2.npy')
# log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]
# print('Hot:', chi2_min)
# n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
# P = 1
# I0, Q0, U0, V0 = 1, P*np.cos(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*chi0/180*np.pi)

# Hc.parameters_update(B, n0L, thetaB, T, chi, I0, Q0, U0, V0, RM_b, RM_f)
# nu_hp = np.linspace(1.3, 1.5, 1000)*1e9
# Q_hp, U_hp, V_hp = Hc.Stokes_calculate(nu_hp)


# import matplotlib.pyplot as plt

# fig, axs = plt.subplots(3, 1, sharex=True)
# # Remove vertical space between Axes
# fig.subplots_adjust(hspace=0.15)
# fontsize=12
# # Plot each graph, and manually set the y tick values
# axs[0].errorbar(nu_GFR/1e9, Q_GFR, yerr=0.05, fmt='r.',zorder=1)
# axs[0].plot(nu_cp/1e9, Q_cp,'-', label='cold',zorder=2)
# axs[0].plot(nu_hp/1e9, Q_hp,'--', label='hot',zorder=3)
# # axs[0].set_yticks([-0.3, -0.1, 0.1, 0.3])
# axs[0].set_ylim(-0.6, 0.6)
# axs[0].set_ylabel(r'$Q/I$', fontsize=fontsize)
# axs[0].legend()

# axs[1].errorbar(nu_GFR/1e9, U_GFR, yerr=0.05, fmt='r.',zorder=1)
# axs[1].plot(nu_cp/1e9, U_cp,'-',zorder=2)
# axs[1].plot(nu_hp/1e9, U_hp,'--',zorder=3)
# # axs[1].set_yticks([-0.5, 0.0, 0.5, 1.0])
# axs[1].set_ylim(-1.00, 1.00)
# axs[1].set_ylabel(r'$U/I$', fontsize=fontsize)

# axs[2].errorbar(nu_GFR/1e9, V_GFR, yerr=0.05, fmt='r.',zorder=1)
# axs[2].plot(nu_cp/1e9, V_cp,'-',zorder=2)
# axs[2].plot(nu_hp/1e9, V_hp,'--',zorder=3)
# axs[2].set_ylim(-1.00, 1.00)
# # axs[2].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

# axs[2].set_ylabel(r'$V/I$', fontsize=fontsize)
# axs[2].set_xlabel(r'$\nu ~(\mathrm{GHz})$', fontsize=fontsize)

# # plt.legend()
# plt.savefig('./results/GFR_ColdandHot1.3-1.5.pdf', dpi=100, bbox_inches='tight')
# plt.show()

#%% a simple example of the mock data of FRB 20201124A
# import numpy as np
# import matplotlib.pyplot as plt
# import preprocessing as prep
# from scipy.constants import c

# # RM of this burst is -672.199
# RM = -672.199
# Stokes, t, f, MJDs = np.load('FRB20201124A_track_M01_59326.23324917.npy', allow_pickle=True)
# I, Q, U, V = Stokes

# P = np.sqrt(Q**2+U**2+V**2)

# # choose a section of noise
# noise_sel = np.logical_and(t>0, t<30) # protected variables

# # handle a RFI mask
# RFI_sel = prep.look_for_RFI(P, t, f, noise_sel, 'FRB20201124A1_'+'59326.23324917')     # protected variables

# # remove these RFI channel
# t, f = t, f[~RFI_sel[::-1]]
# I1, Q1, U1, V1 = I[~RFI_sel[::-1],:], Q[~RFI_sel[::-1],:], U[~RFI_sel[::-1],:], V[~RFI_sel[::-1],:]

# P1 = np.sqrt(Q1**2+U1**2+V1**2)

# # remove small P channels
# P2 = P1 - np.mean(P1[:,noise_sel], axis=1, keepdims=True)
# t,f = t, f
# I2, Q2, U2, V2 = I1, Q1, U1, V1

# Qerr_f = np.nanstd(Q2[:,noise_sel], axis=1)
# Uerr_f = np.nanstd(U2[:,noise_sel], axis=1)
# Verr_f = np.nanstd(V2[:,noise_sel], axis=1)

# # mask the P < 7*noise
# P2_t = np.nanmean(P2, axis=0)
# P5noise_sel = P2_t < np.mean(P2_t[noise_sel]) + 7*np.std(P2_t[noise_sel])

# I2[:,P5noise_sel] = np.nan
# Q2[:,P5noise_sel] = np.nan
# U2[:,P5noise_sel] = np.nan
# V2[:,P5noise_sel] = np.nan
# P2[:,P5noise_sel] = np.nan

# f = f[::-1]*1e6
# P_f = np.nanmean(P2, axis=1)
# Q_f = np.nanmean(Q2, axis=1)
# U_f = np.nanmean(U2, axis=1)
# V_f = np.nanmean(V2, axis=1)


# # the final f and \vec{P}(f)
# f = f[~np.isnan(P_f)]
# lambda_square = (c/f)**2
# Q_f, U_f, V_f = Q_f[~np.isnan(P_f)], U_f[~np.isnan(P_f)], V_f[~np.isnan(P_f)]
# Qerr_f, Uerr_f, Verr_f = Qerr_f[~np.isnan(P_f)], Uerr_f[~np.isnan(P_f)], Verr_f[~np.isnan(P_f)]

# # filter the low P channels (0.04841591 is the noise std of the off-pulse P_f after we subtract the time noise mean for each channel)
# sel_lowP = np.logical_and(~(P_f < 0.04841591*10), f>1.02e9) 

# f = f[sel_lowP]
# lambda_square = lambda_square[sel_lowP]
# Q_f, U_f, V_f = Q_f[sel_lowP], U_f[sel_lowP], V_f[sel_lowP]
# Qerr_f, Uerr_f, Verr_f  = Qerr_f[sel_lowP], Uerr_f[sel_lowP], Verr_f [sel_lowP]
# P_normal = np.sqrt(Q_f**2+U_f**2+V_f**2)

# # # derotate QU
# # L_f = Q_f + 1j*U_f
# # L_f_deRM = L_f * np.exp(-2j * RM * lambda_square)
# # Q_f_deRM, U_f_deRM = np.real(L_f_deRM), np.imag(L_f_deRM)


# f = f
# QoP, UoP, VoP = Q_f/P_normal, U_f/P_normal, V_f/P_normal
# QoP_err = np.abs(QoP)*np.sqrt( ((U_f**2+V_f**2)/(Q_f*P_normal**2))**2*Qerr_f**2 + (U_f/P_normal**2)**2*Uerr_f**2 + (V_f/P_normal**2)**2*Verr_f**2 )/np.sqrt(np.sum(~P5noise_sel))
# UoP_err = np.abs(UoP)*np.sqrt( (Q_f/P_normal**2)**2*Qerr_f**2 + ((Q_f**2+V_f**2)/(U_f*P_normal**2))**2*Uerr_f**2 + (V_f/P_normal**2)**2*Verr_f**2 )/np.sqrt(np.sum(~P5noise_sel))
# VoP_err = np.abs(VoP)*np.sqrt( (Q_f/P_normal**2)**2*Qerr_f**2 + (U_f/P_normal**2)**2*Uerr_f**2 + ((V_f**2+Q_f**2)/(V_f*P_normal**2))**2*Verr_f**2 )/np.sqrt(np.sum(~P5noise_sel))



# from qcosmc.MCMC import MCMC_class, MCplot
# from Faraday_conversion import QUVhot_likelihood

# likelihood = QUVhot_likelihood(f,QoP,UoP,VoP,QoP_err,UoP_err,VoP_err)
# Hotclass = Hot_class()

# parameters_best = []

# ### check cold plasma scenario
# def chi2(theta):
#     log10n0L, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore = theta
#     log10T = 0
#     likelihood.update(10**log10B, 10**log10n0L, thetaB, 10**log10T, chi,
#                           beta0, chi0,
#                           RM_back, RM_fore)
#     chi2 = likelihood.log_likelihood()/(-0.5)
#     if chi2 < 562056:
#         # print(chi2)
#         parameters_best.append( [log10n0L, log10T, log10B, thetaB, chi, beta0*180/np.pi, chi0*180/np.pi, RM_back, RM_fore, chi2])
#     return chi2

# params = [[r'\log_{10}(n_{0}L/ \mathrm{cm}^{-2})', 14.237675627306903, 5, 25],
#           # [r'\log_{10}(T / \mathrm{K})', 6.98776239192092, -2, 10],
#           [r'\log_{10}(B / \mathrm{G})', 1.586653469285314, -1, 5],
#           [r'\theta_{B} (\mathrm{deg})', 111.16853712024374, 0, 180],
#           [r'\chi_{p} (\mathrm{deg})', -61.085453193685694, -180, 180],
#           [r'\beta_0 (\mathrm{deg})',-155.3298953483711, -90*3, 90],
#           [r'\chi_0 (\mathrm{deg})', 6.971974776760106, -45, 90],
#           [r'\mathrm{RM_{b}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', -32.28403834579892, -400, 400],
#           [r'\mathrm{RM_{f}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', -8.563898117477915, -400, 400]
#           ]

# MC = MCMC_class(params, chi2, 'Cold_1124A_926')
# MC.MCMC(200000)

# chains = [
#           ['Cold_1124A_926', 'FRB20201124A burst 926 (Cold)']
#           ]
# pl = MCplot(chains)
# pl.plot3D([1,2,3,4,5,6,7,8])
# pl.results2
# pl.results


# A = np.array(parameters_best)

# np.save('./chains/Cold_1124A_926_listchi2.npy',A)
# A = np.load('./chains/Cold_1124A_926_listchi2.npy')

# log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]


# # log10n0L, log10T = 14.5, 5
# # log10B, thetaB, chi = 1.5, 104.9, -70.0
# # beta0, chi0 =  19.52, 7
# # RM_back, RM_fore, chi2_min = -31.7, -10.73, 0

# n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
# beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
# P = 1
# I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)   

# Hotclass.parameters_update(B, n0L, thetaB, T, chi, I0, Q0, U0, V0, RM_back, RM_fore)
# num = np.linspace(1.0, 1.5, 1000)*1e9
# Qm, Um, Vm = Hotclass.Stokes_calculate(num)

# likelihood.update(10**log10n0L, 10**log10T, 10**log10B, thetaB, chi,
#                   I0, Q0, U0, V0,
#                   RM_back, RM_fore)
# chi2 = likelihood.log_likelihood()/(-0.5)
# print(chi2, chi2_min)



# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6), gridspec_kw={'hspace': 0})

# ax1.errorbar(f, QoP, yerr=QoP_err, fmt='r.', markersize=2, alpha=0.5)
# ax1.plot(num, Qm, 'r-')
# ax1.set_ylabel('Q/P')
# ax1.set_ylim([-1,1])

# ax2.errorbar(f, UoP, yerr=UoP_err, fmt='g.', markersize=2, alpha=0.5)
# ax2.plot(num, Um, 'g-')
# ax2.set_ylabel('U/P')
# ax2.set_ylim([-1,1])

# ax3.errorbar(f, VoP, yerr=VoP_err, fmt='b.', markersize=2, alpha=0.5)
# ax3.plot(num, Vm, 'b-')
# ax3.set_xlabel('f')
# ax3.set_ylabel('V/P')
# ax3.set_ylim([-1,1])

# plt.xlim([1e9, 1.5e9])
# plt.savefig('./Cold_FRB20201124A_burst_926.pdf')


# ### check the hot plasma scenario
# def chi2(theta):
#     log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore = theta
    
#     likelihood.update(10**log10B, 10**log10n0L, thetaB, 10**log10T, chi,
#                           beta0, chi0,
#                           RM_back, RM_fore)
#     chi2 = likelihood.log_likelihood()/(-0.5)
    
#     if chi2 < 556717:
#         parameters_best.append( [log10n0L, log10T, log10B, thetaB, chi, beta0*180/np.pi, chi0*180/np.pi, RM_back, RM_fore, chi2])
#     return chi2

# params = [[r'\log_{10}(n_{0}L/ \mathrm{cm}^{-2})', 24.821043213267075, 18, 35],
#           [r'\log_{10}(T / \mathrm{K})', 13.275740621103614, 10, 20],
#           [r'\log_{10}(B / \mathrm{G})', -2.98936330333464, -5, -1],
#           [r'\theta_{B} (\mathrm{deg})', 179.6925353933328, 0, 180],
#           [r'\chi_{p} (\mathrm{deg})', -58.35871699991151, -180, 180],
#           [r'\beta_0 (\mathrm{deg})', -91.6449340437642, -90*3, 90],
#           [r'\chi_0 (\mathrm{deg})', 5.546019255452664, -45, 90],
#           [r'\mathrm{RM_{b}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', 9.349112875089173, -1000, 1000],
#           [r'\mathrm{RM_{f}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', -8.124686065911114, -1000, 1000]
#           ]

# MC = MCMC_class(params, chi2, 'Hot_1124A_926')
# MC.MCMC(200000)

# chains = [
#           ['Hot_1124A_926', 'FRB20201124A burst 926 (Hot)']
#           ]
# pl = MCplot(chains)
# pl.plot3D([1,2,3,4,5,6,7,8,9])
# pl.results2
# pl.results


# A = np.array(parameters_best)

# np.save('./chains/Hot_1124A_926_listchi2.npy',A)
# A = np.load('./chains/Hot_1124A_926_listchi2.npy')

# log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]


# log10n0L, log10T = 27.481814167223682, 13.311131714264453
# log10B, thetaB, chi = -5.618074240059897, 178.7294966249979, 0
# beta0, chi0 =  -5.638584538705724, 1.9841675472624232
# RM_back, RM_fore, chi2_min = 0, 8, 0

# n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
# beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
# P = 1
# I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)   

# Hotclass.parameters_update(B, n0L, thetaB, T, chi, I0, Q0, U0, V0, RM_back, RM_fore)
# num = np.linspace(1.0, 1.5, 1000)*1e9
# Qm, Um, Vm = Hotclass.Stokes_calculate(num)

# likelihood.update(10**log10n0L, 10**log10T, 10**log10B, thetaB, chi,
#                   I0, Q0, U0, V0,
#                   RM_back, RM_fore)
# chi2 = likelihood.log_likelihood()/(-0.5)
# print(chi2, chi2_min)


# fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8, 6), gridspec_kw={'hspace': 0})

# ax1.errorbar(f, QoP, yerr=QoP_err, fmt='r.', markersize=2, alpha=0.5)
# ax1.plot(num, Qm, 'r-')
# ax1.set_ylabel('Q/P')
# ax1.set_ylim([-1,1])

# ax2.errorbar(f, UoP, yerr=UoP_err, fmt='g.', markersize=2, alpha=0.5)
# ax2.plot(num, Um, 'g-')
# ax2.set_ylabel('U/P')
# ax2.set_ylim([-1,1])

# ax3.errorbar(f, VoP, yerr=VoP_err, fmt='b.', markersize=2, alpha=0.5)
# ax3.plot(num, Vm, 'b-')
# ax3.set_xlabel('f')
# ax3.set_ylabel('V/P')
# ax3.set_ylim([-1,1])

# plt.xlim([1e9, 1.5e9])
# plt.savefig('./Hot_FRB20201124A_burst_926.pdf')


# ### plot fitting results
# # cold scenario
# A = np.load('./chains/Cold_1124A_926_listchi2.npy')
# log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]
# n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
# beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
# P = 1
# I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)   

# Hotclass.parameters_update(B, n0L, thetaB, T, chi, I0, Q0, U0, V0, RM_back, RM_fore)
# nu_cp = np.linspace(1.0, 1.5, 1000)*1e9
# Q_cp, U_cp, V_cp = Hotclass.Stokes_calculate(nu_cp)

# # hot scenarrio
# A = np.load('./chains/Hot_1124A_926_listchi2.npy')
# log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]
# n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
# beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
# P = 1
# I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)   

# Hotclass.parameters_update(B, n0L, thetaB, T, chi, I0, Q0, U0, V0, RM_back, RM_fore)
# nu_hp = np.linspace(1.0, 1.5, 1000)*1e9
# Q_hp, U_hp, V_hp = Hotclass.Stokes_calculate(nu_hp)



# import matplotlib.pyplot as plt
# # f,QoP,UoP,VoP,QoP_err,UoP_err,VoP_err
# fig, axs = plt.subplots(3, 1, sharex=True)
# # Remove vertical space between Axes
# fig.subplots_adjust(hspace=0.15)
# fontsize=12
# # Plot each graph, and manually set the y tick values
# axs[0].errorbar(f/1e9, QoP, yerr=QoP_err, fmt='r.', markersize=2, alpha=0.5, zorder=1)
# axs[0].plot(nu_cp/1e9, Q_cp,'-', label='cold', zorder=2)
# axs[0].plot(nu_hp/1e9, Q_hp,'--', label='hot', zorder=3)
# # axs[0].set_yticks([-0.3, -0.1, 0.1, 0.3])
# axs[0].set_ylim(-1.0, 1.0)
# axs[0].set_ylabel(r'$Q/I$', fontsize=fontsize)
# axs[0].legend()

# axs[1].errorbar(f/1e9, UoP, yerr=UoP_err, fmt='r.', markersize=2, alpha=0.5,zorder=1)
# axs[1].plot(nu_cp/1e9, U_cp,'-',zorder=2)
# axs[1].plot(nu_hp/1e9, U_hp,'--',zorder=3)
# # axs[1].set_yticks([-0.5, 0.0, 0.5, 1.0])
# axs[1].set_ylim(-1.00, 1.00)
# axs[1].set_ylabel(r'$U/I$', fontsize=fontsize)

# axs[2].errorbar(f/1e9, VoP, yerr=VoP_err, fmt='r.', markersize=2, alpha=0.5,zorder=1)
# axs[2].plot(nu_cp/1e9, V_cp,'-',zorder=2)
# axs[2].plot(nu_hp/1e9, V_hp,'--',zorder=3)
# axs[2].set_ylim(-1.00, 1.00)
# # axs[2].set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])

# axs[2].set_ylabel(r'$V/I$', fontsize=fontsize)
# axs[2].set_xlabel(r'$\nu ~(\mathrm{GHz})$', fontsize=fontsize)

# plt.legend()
# plt.savefig('./results/1124A2_ColdandHot.pdf', dpi=100, bbox_inches='tight')
# plt.show()















































