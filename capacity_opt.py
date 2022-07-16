import numpy as np
import scipy.stats as sps

def skellam(b1, b2, B, a, d, M):
    if (b2 == 0):
        p = 0
        for n in range(-M, 1 - b1):
            p += sps.skellam.pmf(n, a, d)
        return p
    if (b2 == B):
        p = 0
        for n in range(B - b1, M + 1):
            p += sps.skellam.pmf(n, a, d)
        return p
    return sps.skellam.pmf(b2 - b1, a, d)

def one_step_matrix(B, a, d):
    P = np.zeros((B + 1, B + 1))
    for i in range(B + 1):
        for j in range(B + 1):
            P[i, j] = skellam(i, j, B, a, d, 100)
    return P

def K_step_matrix(B, a, d, tau, K):
    onestep = one_step_matrix(B, a[0], d[0])
    P = np.eye(B + 1)
    steps = 0
    for k in range(K):
        if (np.floor(k / tau) > steps):
            steps += 1
            onestep = one_step_matrix(B, a[steps], d[steps])
        P = np.matmul(P, onestep)
    return P

def survival(B, a, d, tau, K, b0):
    tot_time = K * tau
    onestep = one_step_matrix(B, a[0], d[0])
    P = np.eye(B + 1)
    downtime = 0
    steps = 0
    for k in range(K):
        if (np.floor(k * tau) > steps):
            steps += 1
            onestep = one_step_matrix(B, a[steps], d[steps])
        P = np.matmul(P, onestep)
        downtime += tau * (P[b0, 0] + P[b0, B])
    return (tot_time - downtime) / tot_time

def best_survival(B, a, d, tau, K):
    p = 0
    for b0 in range(B + 1):
        pb = survival(B, a, d, tau, K, b0)
        if (p < pb):
            p = pb
    return p

def min_size(a, d, tau, K, p_thr):
    for B in range(1,100):
        p = best_survival(B, a, d, tau, K)
        print(B,p)
        if (p > p_thr):
            return B
        

if __name__ == '__main__':
    a = [0, 5, 4, 4]
    d = [0, 5, 4, 4]
    tau = 1 / 6
    K = 18
    
    min_size(a,d,tau,K,0.9)
