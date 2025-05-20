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


class Cold_class(object):
    '''
    This is a simple version of the analytical solution of the transfer of 
    polarization of FRB as a strong incoming wave propagating in a cold
    magnetized plasma.
    
    In this solution, we neglect the absorption terms and set all cross terms
    to be zero and simplifies reality to three plasma mediums
    (FRB --> background screen --> conversion screen (cold) --> foreground screen).
    In this scenario, B, n0L, and thetaB can be combined into RM0 and CM0, so the
    degeneracy between them is avioded. 
    (Note thta this is only valid for cold scenario due to the existence of f(X) and g(X))
    
    T only matters in the hot scenario, so T = 0 is employed in the cold scenario.
    '''
    def __init__(self):
        self.RM = 0 # rad m^-2
        self.CM = 0 # rad m^-3
        self.I0, self.Q0, self.U0, self.V0 = 1, 1, 1, 1                        # the initial Stokes angles of the incoming wave (indeed no units)
        self.chi_p = 0                                                         # the angle between the rho_U=eta_U = 0 frame and the observer frame
        self.T = 1 # K
    def parameters_update(self, RM, CM, I0, Q0, U0, V0, chi_p, RM_b, RM_f):
        # the input RM and CM are in units of rad m^-2 and rad m^-3
        self.RM = RM
        self.CM = CM

        
        self.I0, self.Q0, self.U0, self.V0 = I0, Q0, U0, V0
        self.chi_p = chi_p
        
        self.RM_b = RM_b
        self.RM_f = RM_f
        
    def Stokes_calculate(self, nu):
        l = speed_of_light_movers / nu # m = */Hz
        
        rho = m_e*c**2/(k*self.T)
        
        # Faraday rotation and conversion coefficients in the etaU = rhoU = 0 system
        # We reparameterize RV = rho_V * L = 2 * RM * lambda^2 * kv_ratio(0,2,rho) * Function_g(X)
        # RQ = rho_Q * L = CM * lambda^3 * (kv_ratio(1,2,rho)+6/rho) * Function_f(X)
        # Function_g(X) = Function_f(X) = 1
        
        RV = 2 * self.RM * l**2 * kv_ratio(0,2,rho) * 1
        RQ = - self.CM * l**3 * (kv_ratio(1,2,rho)+6/rho) * 1
        
            #rotate M to M'
        # eta_Ip, eta_Qp, eta_Up, eta_Vp = 0, 0, 0, 0
        rho_Qp, rho_Up, rho_Vp = RQ*np.cos(2*self.chi_p), -RQ*np.sin(2*self.chi_p), RV
        
        
        # preparations for solutions
        # zeta2 = 0
        zetas2 = np.power(rho_Qp,2) + np.power(rho_Up,2) + np.power(rho_Vp,2)
        
        # kappa = 0
        kappas = np.sqrt(zetas2)
        
        k2_ks2 = 0 + np.power(kappas,2)
        
        q2, u2, v2 = (0+np.power(rho_Qp,2))/k2_ks2, (0+np.power(rho_Up,2))/k2_ks2, (0+np.power(rho_Vp,2))/k2_ks2
        
        # ### icrossj = (eta_i*rho_j - rho_i*eta_j)/k2_ks2
        # qcrossu = 0
        # ucrossv = 0
        # vcrossq = 0
        
        # qcrossk = 0
        # ucrossk = 0
        # vcrossk = 0
        
        # ucrossq = - qcrossu
        # vcrossu = - ucrossv
        # qcrossv = - vcrossq
        # kcrossq = - qcrossk
        # kcrossu = - ucrossk
        # kcrossv = - vcrossk
        
        qdotu = (0 + rho_Qp*rho_Up)/k2_ks2
        udotv = (0 + rho_Up*rho_Vp)/k2_ks2
        vdotq = (0 + rho_Vp*rho_Qp)/k2_ks2
        
        qdotk = (0 + rho_Qp*kappas)/k2_ks2
        udotk = (0 + rho_Up*kappas)/k2_ks2
        vdotk = (0 + rho_Vp*kappas)/k2_ks2
        
        udotq = qdotu
        vdotu = udotv
        qdotv = vdotq
        kdotq = qdotk
        kdotu = udotk
        kdotv = vdotk
        
        # kappas * L
        ksL = np.sqrt(RQ**2+RV**2)

        # calculate the elements of matrix M
        '''
              | 1  0  0  0  |
        M =   | 0           | 
              | 0           |
              | 0           |
        
        '''
        # M11 = 1*1                                                             +0
        # M12 = 0                                -0                             -0                               -0
        # M13 = 0                                -0                             -0                               -0
        # M14 = 0                                -0                             -0                               -0
        
        M21 = 0                                -0                             +0                               -0
        M22 = 0.5*(1+q2-u2-v2)*1                                              +0.5*(1-q2+u2+v2)*np.cos(ksL)
        M23 = qdotu*1                          +0                             -qdotu*np.cos(ksL)               -vdotk*np.sin(ksL)
        M24 = qdotv*1                          -0                             -qdotv*np.cos(ksL)               +udotk*np.sin(ksL)
        
        M31 = 0                                -0                             +0                               -0
        M32 = qdotu*1                          -0                             -qdotu*np.cos(ksL)               +vdotk*np.sin(ksL)
        M33 = 0.5*(1-q2+u2-v2)*1                                              +0.5*(1+q2-u2+v2)*np.cos(ksL)
        M34 = udotv*1                          +0                             -udotv*np.cos(ksL)               -qdotk*np.sin(ksL)
        
        M41 = 0                                -0                             +0                               -0
        M42 = qdotv*1                          +0                             -qdotv*np.cos(ksL)               -udotk*np.sin(ksL)
        M43 = udotv*1                          -0                             -udotv*np.cos(ksL)               +qdotk*np.sin(ksL)
        M44 = 0.5*(1-q2-u2+v2)*1                                              +0.5*(1+q2+u2-v2)*np.cos(ksL)
        
        I0, Q0, U0, V0 = self.I0, self.Q0*np.cos(2*self.chi_p)+self.U0*np.sin(2*self.chi_p), -self.Q0*np.sin(2*self.chi_p)+self.U0*np.cos(2*self.chi_p), self.V0
        
        # background
        RMobs_times_2_times_lambda2 = 2*self.RM_b*(speed_of_light_movers/nu)**2
        Ib = I0
        Qb = Q0*np.cos(RMobs_times_2_times_lambda2) - U0*np.sin(RMobs_times_2_times_lambda2)
        Ub = Q0*np.sin(RMobs_times_2_times_lambda2) + U0*np.cos(RMobs_times_2_times_lambda2)
        Vb = V0
        
        # conversion
        Q1 = (M21*Ib + M22*Qb + M23*Ub + M24*Vb)
        U1 = (M31*Ib + M32*Qb + M33*Ub + M34*Vb)
        V1 = (M41*Ib + M42*Qb + M43*Ub + M44*Vb)
        
        # foreground
        RMobs_times_2_times_lambda2 = 2*self.RM_f*(speed_of_light_movers/nu)**2
        Qf = Q1*np.cos(RMobs_times_2_times_lambda2) - U1*np.sin(RMobs_times_2_times_lambda2)
        Uf = Q1*np.sin(RMobs_times_2_times_lambda2) + U1*np.cos(RMobs_times_2_times_lambda2)
        Vf = V1
        return Qf, Uf, Vf

class QUV_cold_likelihood(bilby.Likelihood):
    def __init__(self, nu, Q, U, V, Q_err, U_err, V_err, Model):
        '''
        A simple likelihood to fit Cold_class.
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
        
        self.Model = Model
        
    def update(self, RM, CM, chi_p, beta_0, chi_0, RM_back, RM_fore):
        self.parameters['RM'] = RM
        self.parameters['CM']  = CM
        self.parameters['chi_p']  = chi_p
        self.parameters['beta_0']  = beta_0
        self.parameters['chi_0']  = chi_0
        self.parameters['RM_b'] = RM_back
        self.parameters['RM_f'] = RM_fore
    
    def log_likelihood(self):
        # get the parameters
        RM = self.parameters['RM']
        CM = self.parameters['CM']
        
        chi_p = self.parameters['chi_p']/180*np.pi
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
        # print(ll_Q/(-0.5), ll_U/(-0.5), ll_V/(-0.5))
        return log_likelihood



class conversion_class(object):
    '''
    This is the full version of the analytical solution of the transfer of 
    polarization of FRB as a strong incoming wave propagating in a homogeneous
    magnetized plasma. This is the case of the thermal plasma (section 2.1).
    '''
    def __init__(self):
        # basic parameters
        self.B = 1                                                             # magnetic field (G)
        self.L = 10**4                                                         # typical size of medium (cm)
        self.n0 = 1                                                            # number density (cm^(-3))
        self.T = 100                                                           # temperature of medium (K)
        self.theta_B = 70/180*np.pi                                            # angle between the magnetic field B and wave vector
        self.I0, self.Q0, self.U0, self.V0 = 1, 1, 1, 1                        # the initial Stokes angles of the incoming wave (indeed no units)
        self.chi = 0                                                           # the angle between the rho_U=eta_U = 0 frame and the observer frame
    
    def parameters_update(self, n0, L, T, B, theta_B, chi, I0, Q0, U0, V0):
        self.n0 = n0
        self.L = L
        self.T = T
        self.B = B
        self.theta_B = theta_B/180*np.pi    # the input theta_B is in units of degree
        self.chi = chi/180*np.pi    # the input chi is in units of degree
        self.I0, self.Q0, self.U0, self.V0 = I0, Q0, U0, V0

    def Stokes_calculate(self, nu):
        # calculate the cyclotron frequency
        omega_B = e*self.B/(m_e*c)
        nu_B = omega_B/2/pi                                                    # Hz
        
        rho = m_e*c**2/(k*self.T)
        X = 10**(3/2) * 2**(1/4) / rho * (nu_B/nu*np.sin(self.theta_B))**(1/2)
        
        # Faraday rotation and conversion coefficients in the etaU = rhoU = 0 system
        eta_I = 8/3/np.sqrt(2*pi) * np.power(e,6)*np.power(self.n0,2)/np.power(k*self.T*m_e,3/2)/c/np.power(nu,2) * np.log( np.power(2*k*self.T, 3/2)/(4.2*pi*np.power(e,2)*np.power(m_e,1/2)*nu) )
        eta_Q = 3/8/pi**2 * np.power(omega_B*np.sin(self.theta_B)/nu,2) * eta_I
        eta_V = -omega_B*np.cos(self.theta_B)/pi/nu * eta_I
        
        rho_V = np.power(omega_B/nu,2) * e/self.B *self.n0 * np.cos(self.theta_B)/pi * kv_ratio(0,2,rho) * Function_g(X)
        rho_Q = -np.power(omega_B/nu,3) * e/self.B *self.n0 * np.power(np.sin(self.theta_B),2)/(4*pi*pi) * (kv_ratio(1,2,rho)+6/rho) * Function_f(X)
        
            #rotate M to M'
        eta_Ip, eta_Qp, eta_Up, eta_Vp = eta_I, eta_Q*np.cos(2*self.chi), -eta_Q*np.sin(2*self.chi2), eta_V
        rho_Qp, rho_Up, rho_Vp = rho_Q*np.cos(2*self.chi), -rho_Q*np.sin(2*self.chi), rho_V
        
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
        M11 =  0.5*(1+q2+u2+v2)*np.cosh(kappa*self.L)                         +0.5*(1-q2-u2-v2)*np.cos(kappas*self.L)
        M12 = -ucrossv*np.cosh(kappa*self.L)   -qdotk*np.sinh(kappa*self.L)   -vcrossu*np.cos(kappas*self.L)   -qcrossk*np.sin(kappas*self.L)
        M13 = -vcrossq*np.cosh(kappa*self.L)   -udotk*np.sinh(kappa*self.L)   -qcrossv*np.cos(kappas*self.L)   -ucrossk*np.sin(kappas*self.L)
        M14 = -qcrossu*np.cosh(kappa*self.L)   -vdotk*np.sinh(kappa*self.L)   -ucrossq*np.cos(kappas*self.L)   -vcrossk*np.sin(kappas*self.L)
        
        M21 =  ucrossv*np.cosh(kappa*self.L)   -qdotk*np.sinh(kappa*self.L)   +vcrossu*np.cos(kappas*self.L)   -qcrossk*np.sin(kappas*self.L)
        M22 = 0.5*(1+q2-u2-v2)*np.cosh(kappa*self.L)                          +0.5*(1-q2+u2+v2)*np.cos(kappas*self.L)
        M23 = qdotu*np.cosh(kappa*self.L)      +vcrossk*np.sinh(kappa*self.L) -qdotu*np.cos(kappas*self.L)     -vdotk*np.sin(kappas*self.L)
        M24 = qdotv*np.cosh(kappa*self.L)      -ucrossk*np.sinh(kappa*self.L) -qdotv*np.cos(kappas*self.L)     +udotk*np.sin(kappas*self.L)
        
        M31 = vcrossq*np.cosh(kappa*self.L)    -udotk*np.sinh(kappa*self.L)   +qcrossv*np.cos(kappas*self.L)   -ucrossk*np.sin(kappas*self.L)
        M32 = qdotu*np.cosh(kappa*self.L)      -vcrossk*np.sinh(kappa*self.L) -qdotu*np.cos(kappas*self.L)     +vdotk*np.sin(kappas*self.L)
        M33 = 0.5*(1-q2+u2-v2)*np.cosh(kappa*self.L)                          +0.5*(1+q2-u2+v2)*np.cos(kappas*self.L)
        M34 = udotv*np.cosh(kappa*self.L)      +qcrossk*np.sinh(kappa*self.L) -udotv*np.cos(kappas*self.L)     -qdotk*np.sin(kappas*self.L)
        
        M41 = qcrossu*np.cosh(kappa*self.L)    -vdotk*np.sinh(kappa*self.L)   +ucrossq*np.sin(kappas*self.L)   -vcrossk*np.sin(kappas*self.L)
        M42 = qdotv*np.cosh(kappa*self.L)      +ucrossk*np.sinh(kappa*self.L) -qdotv*np.cos(kappa*self.L)      -udotk*np.sin(kappas*self.L)
        M43 = udotv*np.cosh(kappa*self.L)      -qcrossk*np.sinh(kappa*self.L) -udotv*np.cos(kappas*self.L)     +qdotk*np.sin(kappas*self.L)
        M44 = 0.5*(1-q2-u2+v2)*np.cosh(kappa*self.L)                          +0.5*(1+q2+u2-v2)*np.cos(kappas*self.L)
        
        # self.Matrix = [M11,M12,M13,M14,M21,M22,M23,M24,M31,M32,M33,M34,M41,M42,M43,M44]
        
        I0, Q0, U0, V0 = self.I0, self.Q0*np.cos(2*self.chi)+self.U0*np.sin(2*self.chi), -self.Q0*np.sin(2*self.chi)+self.U0*np.cos(2*self.chi), self.V0
            # update
        tao = eta_Ip*self.L
        self.I = (M11*I0 + M12*Q0 + M13*U0 + M14*V0)*np.exp(-tao)
        self.Q = (M21*I0 + M22*Q0 + M23*U0 + M24*V0)*np.exp(-tao)
        self.U = (M31*I0 + M32*Q0 + M33*U0 + M34*V0)*np.exp(-tao)
        self.V = (M41*I0 + M42*Q0 + M43*U0 + M44*V0)*np.exp(-tao)
        
        return self.Q, self.U, self.V

class conversions_class(object):
    '''
    This is the simple version of the analytical solution of the transfer of 
    polarization of FRB as a strong incoming wave propagating in a
    magnetized plasma. This is the case of the thermal plasma (section 2.1).
    
    In this solution, we neglect the absorption terms and set all cross terms
    to be zero and simplifies reality to three plasma mediums
    (FRB --> background screen --> conversion screen --> foreground screen).
    '''
    def __init__(self):
        # basic parameters
        self.B = 1                                                             # magnetic field (G)
        self.n0L = 1 * 10**4                                                   # number density (cm^(-3)) * typical size of medium (cm)
        self.T = 100                                                           # temperature of medium (K)
        self.theta_B = 70/180*np.pi                                            # angle between the magnetic field B and wave vector
        self.I0, self.Q0, self.U0, self.V0 = 1, 1, 1, 1                        # the initial Stokes angles of the incoming wave (indeed no units)
        self.chi = 0                                                           # the angle between the rho_U=eta_U = 0 frame and the observer frame
        
        self.RM = 0 # 2*RM*lambda**2 = rho_V * L
    def parameters_update(self, n0L, T, B, theta_B, chi, I0, Q0, U0, V0, RM_back, RM_fore):
        self.n0L = n0L
        self.T = T
        self.B = B
        self.theta_B = theta_B/180*np.pi                                       # the input theta_B is in units of degree
        self.chi = chi/180*np.pi                                               # the input chi is in units of degree
        self.I0, self.Q0, self.U0, self.V0 = I0, Q0, U0, V0
        self.RM_back = RM_back                                                 # rad m^-2
        self.RM_fore = RM_fore                                                 # rad m^-2

    def Stokes_calculate(self, nu):
        
        # calculate the cyclotron frequency
        omega_B = e*self.B/(m_e*c)
        nu_B = omega_B/2/pi                                                    # Hz
        
        rho = m_e*c**2/(k*self.T)
        X = 10**(3/2) * 2**(1/4) / rho * (nu_B/nu*np.sin(self.theta_B))**(1/2)
        
        # Faraday rotation and conversion coefficients in the etaU = rhoU = 0 system
        # eta_I = 8/3/np.sqrt(2*pi) * np.power(e,6)*np.power(self.n0,2)/np.power(k*self.T*m_e,3/2)/c/np.power(nu,2) * np.log( np.power(2*k*self.T, 3/2)/(4.2*pi*np.power(e,2)*np.power(m_e,1/2)*nu) )
        # eta_Q = 3/8/pi**2 * np.power(omega_B*np.sin(self.theta_B)/nu,2) * eta_I
        # eta_V = omega_B*np.cos(self.theta_B)/pi/nu * eta_I
        
        # RV = rho_V * L, RQ = rho_Q * L
        RV = np.power(omega_B/nu,2) * e/self.B *self.n0L * np.cos(self.theta_B)/pi * kv_ratio(0,2,rho) * Function_g(X)
        RQ = -np.power(omega_B/nu,3) * e/self.B *self.n0L * np.power(np.sin(self.theta_B),2)/(4*pi*pi) * (kv_ratio(1,2,rho)+6/rho) * Function_f(X)
        
            #rotate M to M'
        # eta_Ip, eta_Qp, eta_Up, eta_Vp = 0, 0, 0, 0
        rho_Qp, rho_Up, rho_Vp = RQ*np.cos(2*self.chi), -RQ*np.sin(2*self.chi), RV
        
        # preparations for solutions
        # zeta2 = 0
        zetas2 = np.power(rho_Qp,2) + np.power(rho_Up,2) + np.power(rho_Vp,2)
        
        # kappa = 0
        kappas = np.sqrt(zetas2)
        
        k2_ks2 = 0 + np.power(kappas,2)
        
        q2, u2, v2 = (0+np.power(rho_Qp,2))/k2_ks2, (0+np.power(rho_Up,2))/k2_ks2, (0+np.power(rho_Vp,2))/k2_ks2
        
        # ### icrossj = (eta_i*rho_j - rho_i*eta_j)/k2_ks2
        # qcrossu = 0
        # ucrossv = 0
        # vcrossq = 0
        
        # qcrossk = 0
        # ucrossk = 0
        # vcrossk = 0
        
        # ucrossq = - qcrossu
        # vcrossu = - ucrossv
        # qcrossv = - vcrossq
        # kcrossq = - qcrossk
        # kcrossu = - ucrossk
        # kcrossv = - vcrossk
        
        qdotu = (0 + rho_Qp*rho_Up)/k2_ks2
        udotv = (0 + rho_Up*rho_Vp)/k2_ks2
        vdotq = (0 + rho_Vp*rho_Qp)/k2_ks2
        
        qdotk = (0 + rho_Qp*kappas)/k2_ks2
        udotk = (0 + rho_Up*kappas)/k2_ks2
        vdotk = (0 + rho_Vp*kappas)/k2_ks2
        
        udotq = qdotu
        vdotu = udotv
        qdotv = vdotq
        kdotq = qdotk
        kdotu = udotk
        kdotv = vdotk
        
        # kappas * L
        ksL = np.sqrt(RQ**2+RV**2)

        # calculate the elements of matrix M
        '''
             | 1  0  0  0  |
        M =  | 0           | 
             | 0           |
             | 0           |
        
        '''
        # M11 = 1*1                                                             +0
        # M12 = 0                                -0                             -0                               -0
        # M13 = 0                                -0                             -0                               -0
        # M14 = 0                                -0                             -0                               -0
        
        M21 = 0                                -0                             +0                               -0
        M22 = 0.5*(1+q2-u2-v2)*1                                              +0.5*(1-q2+u2+v2)*np.cos(ksL)
        M23 = qdotu*1                          +0                             -qdotu*np.cos(ksL)               -vdotk*np.sin(ksL)
        M24 = qdotv*1                          -0                             -qdotv*np.cos(ksL)               +udotk*np.sin(ksL)
        
        M31 = 0                                -0                             +0                               -0
        M32 = qdotu*1                          -0                             -qdotu*np.cos(ksL)               +vdotk*np.sin(ksL)
        M33 = 0.5*(1-q2+u2-v2)*1                                              +0.5*(1+q2-u2+v2)*np.cos(ksL)
        M34 = udotv*1                          +0                             -udotv*np.cos(ksL)               -qdotk*np.sin(ksL)
        
        M41 = 0                                -0                             +0                               -0
        M42 = qdotv*1                          +0                             -qdotv*np.cos(ksL)               -udotk*np.sin(ksL)
        M43 = udotv*1                          -0                             -udotv*np.cos(ksL)               +qdotk*np.sin(ksL)
        M44 = 0.5*(1-q2-u2+v2)*1                                              +0.5*(1+q2+u2-v2)*np.cos(ksL)
        

        I0, Q0, U0, V0 = self.I0, self.Q0*np.cos(2*self.chi)+self.U0*np.sin(2*self.chi), -self.Q0*np.sin(2*self.chi)+self.U0*np.cos(2*self.chi), self.V0
        
        # background
        RMobs_times_2_times_lambda2 = 2*self.RM_back*(speed_of_light_movers/nu)**2
        Ib = I0
        Qb = Q0*np.cos(RMobs_times_2_times_lambda2) - U0*np.sin(RMobs_times_2_times_lambda2)
        Ub = Q0*np.sin(RMobs_times_2_times_lambda2) + U0*np.cos(RMobs_times_2_times_lambda2)
        Vb = V0
        
        # conversion
        Q1 = (M21*Ib + M22*Qb + M23*Ub + M24*Vb)
        U1 = (M31*Ib + M32*Qb + M33*Ub + M34*Vb)
        V1 = (M41*Ib + M42*Qb + M43*Ub + M44*Vb)
        
        # foreground
        RMobs_times_2_times_lambda2 = 2*self.RM_fore*(speed_of_light_movers/nu)**2
        Qf = Q1*np.cos(RMobs_times_2_times_lambda2) - U1*np.sin(RMobs_times_2_times_lambda2)
        Uf = Q1*np.sin(RMobs_times_2_times_lambda2) + U1*np.cos(RMobs_times_2_times_lambda2)
        Vf = V1
        return Qf, Uf, Vf

class QUVs_likelihood(bilby.Likelihood):
    def __init__(self, nu, Q, U, V, Q_err, U_err, V_err, Model):
        '''
        A simple likelihood to fit the propagation model with QUV.
        '''
        super().__init__(parameters={'n0L': None, 'T': None, 'B': None, 'thetaB': None, 'chi':None,
                                      'I0': None,'Q0':None, 'U0':None, 'V0':None,
                                      'RM_back': None, 'RM_fore': None})
        # n0L, T, B, theta_B, chi, I0, Q0, U0, V0, RM
        self.nu = nu
        self.Q = Q
        self.U = U
        self.V = V
        self.Q_err = Q_err
        self.U_err = U_err
        self.V_err = V_err
        
        self.Model = Model
        
    def update(self, n0L, T, B, thetaB, chi, I0, Q0, U0, V0, RM_back, RM_fore):
        self.parameters['n0L'] = n0L
        self.parameters['T']  = T
        self.parameters['B']  = B
        self.parameters['thetaB']  = thetaB
        self.parameters['chi']  = chi
        self.parameters['I0'] , self.parameters['Q0'] , self.parameters['U0'] , self.parameters['V0']  = I0, Q0, U0, V0
        self.parameters['RM_back'] = RM_back
        self.parameters['RM_fore'] = RM_fore
    
    def log_likelihood(self):
        # get the parameters
        n0L = self.parameters['n0L']
        T = self.parameters['T']
        B = self.parameters['B']
        thetaB = self.parameters['thetaB']
        chi = self.parameters['chi']
        
        I0, Q0, U0, V0 = self.parameters['I0'], self.parameters['Q0'], self.parameters['U0'], self.parameters['V0']
        RM_back, RM_fore = self.parameters['RM_back'], self.parameters['RM_fore']
        
        self.Model.parameters_update(n0L, T, B, thetaB, chi, I0, Q0, U0, V0, RM_back, RM_fore)
        Qm, Um, Vm = self.Model.Stokes_calculate(self.nu)
            
        # log_likelihood
        
        ll_Q = -0.5*np.sum( (self.Q-Qm)**2/self.Q_err**2 )
        ll_U = -0.5*np.sum( (self.U-Um)**2/self.U_err**2 )
        ll_V = -0.5*np.sum( (self.V-Vm)**2/self.V_err**2 )
        log_likelihood = ll_Q + ll_U + ll_V
        # print(ll_Q/(-0.5), ll_U/(-0.5), ll_V/(-0.5))
        return log_likelihood

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








#%% test module

# # GFR test
# GFR = GFR_class()
# RM, psi0, theta, phi, GRM, Psi0, alpha, chi = 27.7, -87.3, 104.2, 76.3, 4351.7, 0, 2.3, -0.1 
# GFR.parameters_update(RM, psi0, theta, phi, GRM, Psi0, alpha, chi)

# nu1 = np.linspace(1.370, 1.425, 50)*1e9
# Q1,U1,V1 = GFR.Stokes_calculate(nu1)
# Q1err,U1err,V1err = Q1*0+0.05,Q1*0+0.05,Q1*0+0.05


# from qcosmc.MCMC import MCMC_class, MCplot
# conversion = conversions_class()
# likelihood = QUVs_likelihood(nu1,Q1,U1,V1,Q1err,U1err,V1err,conversion)

# ### check cold plasma scenario
# parameters_best = []
# def chi2(theta):
#     log10n0L, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore = theta
#     P = 1
#     log10T = 0
#     beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
#     I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)
#     likelihood.update(10**log10n0L, 10**log10T, 10**log10B, thetaB, chi,
#                         I0, Q0, U0, V0,
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
# MC.MCMC(1000000)

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
#     P=1
#     beta0, chi0 = beta0/180*np.pi, chi0/180*np.pi
#     I0, Q0, U0, V0 = 1, P*np.cos(2*beta0)*np.cos(2*chi0), P*np.sin(2*beta0)*np.cos(2*chi0), P*np.sin(2*chi0)
#     likelihood.update(10**log10n0L, 10**log10T, 10**log10B, thetaB, chi,
#                         I0, Q0, U0, V0,
#                         RM_back, RM_fore)
#     chi2 = likelihood.log_likelihood()/(-0.5)
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
#           [r'\mathrm{RM_{b}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', 0.0030798962397969327, -1000, 1000],
#           [r'\mathrm{RM_{f}} ~ \left(\operatorname{rad} \mathrm{~m}^{-2}\right)', 26.32701863495556, -1000, 1000]
#           ]

# MC = MCMC_class(params, chi2, 'Hot_180301A')
# MC.MCMC(1000000)

# chains = [
#           ['Hot_180301A', 'FRB 20180301A (hot)']
#           ]
# pl = MCplot(chains)
# pl.plot3D([1,2,3,4,5,6,7,8,9])
# pl.results2
# pl.results

# A = np.array(parameters_best)
# np.save('./chains/Hot_180301A_listchi2.npy', A)




# ### GFR cold and hot plasma
# nu_GFR = np.linspace(1.370, 1.425, 50)*1e9
# Q_GFR,U_GFR,V_GFR = GFR.Stokes_calculate(nu_GFR)

# # cold
# A = np.load('./chains/Cold_180301A_listchi2.npy')
# log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]
# print('Cold:', chi2_min)
# n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
# P = 1
# I0, Q0, U0, V0 = 1, P*np.cos(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*chi0/180*np.pi)
# conversion.parameters_update(n0L, T, B, thetaB, chi, I0, Q0, U0, V0, RM_back, RM_fore)
# nu_cp = np.linspace(1.3, 1.5, 1000)*1e9
# Q_cp, U_cp, V_cp = conversion.Stokes_calculate(nu_cp)

# # hot
# A = np.load('./chains/Hot_180301A_listchi2.npy')
# log10n0L, log10T, log10B, thetaB, chi, beta0, chi0, RM_back, RM_fore, chi2_min = A[A[:,-1] == np.min(A[:,-1]), :][0]
# print('Hot:', chi2_min)
# n0L, T, B = 10**log10n0L, 10**log10T, 10**log10B
# P = 1
# I0, Q0, U0, V0 = 1, P*np.cos(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*beta0/180*np.pi)*np.cos(2*chi0/180*np.pi), P*np.sin(2*chi0/180*np.pi)
# conversion.parameters_update(n0L, T, B, thetaB, chi, I0, Q0, U0, V0, RM_back, RM_fore)
# nu_hp = np.linspace(1.3, 1.5, 1000)*1e9
# Q_hp, U_hp, V_hp = conversion.Stokes_calculate(nu_hp)


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










































