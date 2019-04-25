
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp



def RPCA(M):
    L = cp.Variable()
    S = cp.Variable()
    lam = cp.Parameter()
    lam.value = (1/((max(M.shape))**(1/2)))
    constraints = [L + S == M]
    obj = cp.Minimize(cp.norm(L, "nuc") + (lam*cp.norm(S, 1)))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, use_indirect=False)
    print("optimal solutions", L.value, S.value)
    
    


X = np.random.rand(40, 40)
RPCA(X)

    


def threshold_op(X, tau):
    return (np.sign(X)*np.maximum(np.abs(X)-tau, np.zeros(X.shape)))


def singvalthreshold(X, tau):
    U, D, V = LA.svd(X, full_matrices=False)
    return np.dot(U, np.dot(np.diag(threshold_op(D, tau))), V)

def RPCA2(X, tol=None, max_iter=1000):
    S = np.zeros(X.shape)
    Y = np.zeros(X.shape)
    L = np.zeros(X.shape)
    lmbda = 1/(np.sqrt(np.max(X.shape)))
    Sk = S
    Yk = Y
    Lk = L
    mu = (np.prod(X.shape)/(4*LA.norm(X, ord=2)))
    mu_inv = (1/mu)
    
    while ((LA.norm(X-L-S, ord="fro")/LA.norm(X, ord="fro")) > 10**-7):
        Lk = singvalthreshold(X - Sk + mu_inv*Yk, mu_inv)
        Sk = singvalthreshold(X - Lk + mu_inv*Yk, lmbda*mu_inv)
        Yk = Yk + mu(X - Lk - Sk)
        
    return Lk, Sk
    
    