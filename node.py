import math
from turtle import pu
import numpy as np

class Node:
    def __init__(self, tree, i: int, j: int):
        self.tree = tree
        self.i = i
        self.j = j
        self.next_up = None
        self.next_mid = None
        self.next_down = None
        self.under = None
        self.proba_up = None
        self.proba_mid = None
        self.proba_down = None
        self.forward = None

    def solve_trinomial_probs(Si_prime, nxt_mid, r, sigma, dt, D_next, alpha):
        """
        Calcule (p_up, p_mid, p_down) en imposant :
        1) somme = 1
        2) E[S_{t+dt}|S_t] = fwd = Si' * e^{r dt} - D_next
        3) Var[S_{t+dt}|S_t] = Si'^2 * e^{2 r dt} * (e^{sigma^2 dt} - 1)
        Les prix cibles sont (Sup, Smid, Sdown) = (nxt_mid*alpha, nxt_mid, nxt_mid/alpha)
        """
        fwd = Si_prime * math.exp(r*dt) - D_next
        var = (Si_prime**2) * math.exp(2.0*r*dt) * (math.exp(sigma*sigma*dt) - 1.0)

        Sup, Smid, Sdown = nxt_mid*alpha, nxt_mid, nxt_mid/alpha

        A = np.array([
            [1.0,     1.0,       1.0],
            [Sup,     Smid,      Sdown],
            [Sup*Sup, Smid*Smid, Sdown*Sdown]
        ], dtype=float)
        b = np.array([1.0, fwd, var + fwd*fwd], dtype=float)

        pu, pm, pd = np.linalg.solve(A, b)

        # clips légers pour stabilité numérique
        eps = 1e-14
        pu, pm, pd = [
            max(0.0, min(1.0, x)) if -eps <= x <= 1.0 + eps else x
            for x in (pu, pm, pd)
        ]
        s = pu + pm + pd
        if abs(s - 1.0) > 1e-12:
            pu, pm, pd = pu/s, pm/s, pd/s
        return pu, pm, pd


    




    
    
