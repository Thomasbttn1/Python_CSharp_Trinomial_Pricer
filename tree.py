# tree.py
from node import Node
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from matplotlib.collections import LineCollection

class Tree:
    def __init__(self, market, nb_steps: int, delta_t: float):
        self.market = market
        self.nb_steps = nb_steps
        self.delta_t = delta_t
        self.alpha = math.exp(market.vol * math.sqrt(3 * delta_t))
        self.root = Node(self, 0, 0)  # racine
        # probas par pas i -> i+1 : {i: (p_down, p_mid, p_up)}
        self.step_probs = {}

    # ---------- utilitaires ----------
    def pick_next_mid_closest_to_forward(self, nodes_next_col, fwd):
        """Renvoie le Node (colonne i+1) dont le prix est le plus proche du forward attendu."""
        nodes = nodes_next_col.values() if isinstance(nodes_next_col, dict) else nodes_next_col
        return min(nodes, key=lambda n: abs(n.under - fwd))

    def _div_step_index(self, option):
        """Mappe option.div_date (en années) vers l'indice de pas i_div (1..N), sinon None."""
        if option is None or not getattr(option, "div", 0) or option.div_date is None:
            return None, 0.0
        t_div = option.get_div_time_in_years()
        if t_div is None:
            return None, 0.0
        i_div = int(round(t_div / self.delta_t))
        if 1 <= i_div <= self.nb_steps:
            return i_div, float(option.div)
        return None, 0.0

    def trinomial_prob_no_div(self):
        """Probabilités fermées quand D=0 sur le pas (K-T)."""
        a = self.alpha
        v = math.exp((self.market.vol ** 2) * self.delta_t) - 1.0
        denom = (1.0 - a) * ((a**-2) - 1.0)
        p_down = v / denom
        p_up   = p_down / a
        p_mid  = 1.0 - p_up - p_down
        return p_down, p_mid, p_up

    def _solve_trinomial_probs(self, Si_prime, nxt_mid, D_next):
        """
        Résout (p_up, p_mid, p_down) en imposant :
          somme = 1
          E[S_{t+dt}|S_t]  = fwd = Si' * e^{r dt} - D_next
          Var[S_{t+dt}|S_t]= Si'^2 e^{2 r dt} (e^{σ² dt} - 1)
        avec (Sup, Smid, Sdown) = (nxt_mid*a, nxt_mid, nxt_mid/a).
        """
        r, sigma, dt, a = self.market.rate, self.market.vol, self.delta_t, self.alpha
        fwd = Si_prime * math.exp(r*dt) - D_next
        var = (Si_prime**2) * math.exp(2.0*r*dt) * (math.exp(sigma*sigma*dt) - 1.0)

        Sup, Smid, Sdown = nxt_mid*a, nxt_mid, nxt_mid/a
        A = np.array([
            [1.0,     1.0,       1.0],
            [Sup,     Smid,      Sdown],
            [Sup*Sup, Smid*Smid, Sdown*Sdown]
        ], dtype=float)
        b = np.array([1.0, fwd, var + fwd*fwd], dtype=float)

        pu, pm, pd = np.linalg.solve(A, b)

        # clips légers
        eps = 1e-14
        pu, pm, pd = [
            max(0.0, min(1.0, x)) if -eps <= x <= 1.0 + eps else x
            for x in (pu, pm, pd)
        ]
        print(pu, pm, pd)
        s = pu + pm + pd
        if abs(s - 1.0) > 1e-12:
            pu, pm, pd = pu/s, pm/s, pd/s
        # on retourne (p_down, p_mid, p_up) pour cohérence du code
        return pd, pm, pu

    # ---------- construction de l'arbre ----------
    def build_bottom_first(self, option):
        """
        Arbre trinomial recombinant, centré à chaque pas sur le forward local :
            f_i = S_mid * e^{r dt} - D_i
        (Le dividende est pris dans le forward, pas en baissant S_mid directement.)
        """
        N, dt, a = self.nb_steps, self.delta_t, self.alpha
        S0, r = self.market.underlying, self.market.rate
        i_div, D = self._div_step_index(option)

        # racine
        self.levels = {0: {0: self.root}}
        self.root.under = S0
        self.root.forward = S0

        current_mid = S0  # centre de la colonne précédente

        for i in range(1, N + 1):
            self.levels[i] = {}

            # dividende payé sur (i-1 -> i) ?
            D_next = D if (i_div is not None and i == i_div) else 0.0

            # forward local net du dividende
            fwd_mid = current_mid * math.exp(r * dt) - D_next

            # construire la colonne i autour du forward théorique
            for j in range(-i, i + 1):
                self.levels[i][j] = Node(self, i, j)
                self.levels[i][j].under = fwd_mid * (a ** j)

            # choisir le vrai "mid" (noeud le plus proche du forward)
            nxt_mid_node = self.pick_next_mid_closest_to_forward(self.levels[i], fwd_mid)
            nxt_mid_S = nxt_mid_node.under
            nxt_mid_node.forward = fwd_mid

            # calcul des probas du pas i-1 -> i
            if D_next == 0.0:
                p_d, p_m, p_u = self.trinomial_prob_no_div()
            else:
                p_d, p_m, p_u = self._solve_trinomial_probs(current_mid, nxt_mid_S, D_next)
            self.step_probs[i-1] = (p_d, p_m, p_u)

            # préparer pour le pas suivant (mid réel)
            current_mid = nxt_mid_S

        return self

    # ---------- pricing ----------
    def price_european(self, option):
        """Backward induction avec probas par pas."""
        N, dt = self.nb_steps, self.delta_t
        DF = math.exp(-self.market.rate * dt)

        values = {i: {j: 0.0 for j in range(-i, i + 1)} for i in range(N + 1)}
        for j, nd in self.levels[N].items():
            values[N][j] = option.payoff(nd.under)

        for i in range(N - 1, -1, -1):
            p_d, p_m, p_u = self.step_probs[i]
            for j in range(-i, i + 1):
                v_up   = values[i + 1][j + 1]
                v_mid  = values[i + 1][j]
                v_down = values[i + 1][j - 1]
                values[i][j] = DF * (p_u*v_up + p_m*v_mid + p_d*v_down)

        return values[0][0]

    def price_european_recursive(self, option):
        """Prix européen par récursion (prend en compte les probas par pas)."""
        if not (self.levels and len(self.levels) == (self.nb_steps + 1)):
            raise ValueError("Construis d'abord l'arbre (build_bottom_first).")
        DF = math.exp(-self.market.rate * self.delta_t)
        N = self.nb_steps

        @lru_cache(maxsize=None)
        def V(i: int, j: int) -> float:
            if i == N:
                return option.payoff(self.levels[i][j].under)
            p_d, p_m, p_u = self.step_probs[i]
            return DF * (p_u*V(i+1, j+1) + p_m*V(i+1, j) + p_d*V(i+1, j-1))

        return V(0, 0)

    def price_american(self, option):
        """Backward induction avec exercice anticipé + probas par pas."""
        N, dt = self.nb_steps, self.delta_t
        DF = math.exp(-self.market.rate * dt)

        values = {i: {j: 0.0 for j in range(-i, i + 1)} for i in range(N + 1)}
        for j, nd in self.levels[N].items():
            values[N][j] = option.payoff(nd.under)

        for i in range(N - 1, -1, -1):
            p_d, p_m, p_u = self.step_probs[i]
            for j, nd in self.levels[i].items():
                hold = DF * (p_u*values[i+1][j+1] + p_m*values[i+1][j] + p_d*values[i+1][j-1])
                exercise = option.payoff(nd.under)
                values[i][j] = max(exercise, hold)

        return values[0][0]

    # ---------- greeks (no-bump) ----------
    def local_greeks_no_bump(self, option):
        """
        Greeks "classiques" (sans bump) à la racine via le 1er étage.
        Delta/Gamma par différences spatiales. Theta par différence temporelle sur un pas:
            theta ≈ (V_{i=1, j=0} - V_{i=0, j=0}) / (-dt)   (annualisé).
        """
        N, dt = self.nb_steps, self.delta_t
        r = self.market.rate
        DF = math.exp(-r * dt)

        # backward pour remplir values[i][j]
        values = {i: {j: 0.0 for j in range(-i, i + 1)} for i in range(N + 1)}
        for j, nd in self.levels[N].items():
            values[N][j] = option.payoff(nd.under)
        for i in range(N - 1, -1, -1):
            p_d, p_m, p_u = self.step_probs[i]
            for j in range(-i, i + 1):
                v_up   = values[i + 1][j + 1]
                v_mid  = values[i + 1][j]
                v_down = values[i + 1][j - 1]
                values[i][j] = DF * (p_u*v_up + p_m*v_mid + p_d*v_down)

        V0 = values[0][0]
        S_up   = self.levels[1][1].under
        S_mid  = self.levels[1][0].under
        S_down = self.levels[1][-1].under

        V_up   = values[1][1]
        V_mid  = values[1][0]
        V_down = values[1][-1]

        # Delta / Gamma (maille non uniforme ok)
        denom = (S_up - S_down)
        if abs(denom) < 1e-15:
            raise ZeroDivisionError("Maille du 1er étage dégénérée.")
        delta = (V_up - V_down) / denom

        h_up = (S_up - S_mid)
        h_dn = (S_mid - S_down)
        if abs(h_up) < 1e-15 or abs(h_dn) < 1e-15:
            raise ZeroDivisionError("Maille du 1er étage trop fine.")
        gamma = 2.0 * ( (V_up - V_mid)/h_up - (V_mid - V_down)/h_dn ) / (h_up + h_dn)

        # Theta par différence temporelle (stable dans un arbre)
        # V_next = valeur au mid de la colonne suivante (i=1, j=0)
        theta = (V_mid - V0) / (-dt)

        return {"price": V0, "delta": delta, "gamma": gamma, "theta": theta}

    # ---------- plots ----------
    def plot(self, annotate=True, figsize=(9,6), dpi=110, save_path=None):
        """Plot adaptatif: FULL (N<=20) sinon FAST (sous-échantillonné + LineCollection)."""
        if not hasattr(self, "levels"):
            raise ValueError("Il faut d'abord construire l'arbre.")

        N = self.nb_steps
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Mode adaptatif
        use_fast = N > 20
        annotate = annotate and (not use_fast)

        # 1) Arêtes
        segments = []
        if not use_fast:
            for i in range(N):
                col_i = self.levels[i]
                col_ip1 = self.levels[i+1]
                for j, nd in col_i.items():
                    y = nd.under
                    if (j-1) in col_ip1: segments.append([(i, y), (i+1, col_ip1[j-1].under)])
                    if (j  ) in col_ip1: segments.append([(i, y), (i+1, col_ip1[j  ].under)])
                    if (j+1) in col_ip1: segments.append([(i, y), (i+1, col_ip1[j+1].under)])
        else:
            col_stride = max(1, N // 40)  # ~40 colonnes
            for i in range(0, N, col_stride):
                col_i = self.levels[i]
                ip1 = min(i+1, N); col_ip1 = self.levels[ip1]
                row_stride = max(1, (i+1)//10)
                for j in range(-i, i+1, row_stride):
                    nd = col_i.get(j)
                    if nd is None: continue
                    y = nd.under
                    for dj in (-1, 0, +1):
                        j2 = j + dj
                        nd2 = col_ip1.get(j2)
                        if nd2 is None: continue
                        segments.append([(i, y), (ip1, nd2.under)])

        lc = LineCollection(segments, colors="gray", linewidths=0.6, alpha=0.7,
                            antialiased=False, rasterized=True)
        ax.add_collection(lc)

        # 2) Noeuds
        xs, ys = [], []
        if not use_fast:
            for i, col in self.levels.items():
                for j, nd in col.items():
                    xs.append(i); ys.append(nd.under)
        else:
            for i, col in self.levels.items():
                if i % max(1, N//40): continue
                row_stride = max(1, (i+1)//10)
                for j in range(-i, i+1, row_stride):
                    nd = col.get(j)
                    if nd is None: continue
                    xs.append(i); ys.append(nd.under)
        ax.scatter(xs, ys, s=8 if not use_fast else 4, color="C0", zorder=3, rasterized=True)

        # 3) Annotations (petit N)
        if annotate:
            for i, col in self.levels.items():
                for j, nd in col.items():
                    if nd.under is None: continue
                    ax.annotate(f"{nd.under:.2f}", (i, nd.under),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=8)

        # 4) Marque ex-div (si les probas changent)
        i_div = None; p_before = None; p_after = None
        if hasattr(self, "step_probs") and self.step_probs:
            for i in range(1, N):
                prev = self.step_probs.get(i-1); cur = self.step_probs.get(i)
                if prev and cur and any(abs(a-b) > 1e-9 for a,b in zip(prev, cur)):
                    i_div = i; p_before = prev; p_after = cur; break
        if i_div is not None:
            ax.axvline(i_div, color="red", linestyle="--", alpha=0.6, linewidth=1.0)
            fig.text(0.66, 0.95,
                     (f"Div à i={i_div}\n"
                      f"Avant: pu={p_before[2]:.4f}, pm={p_before[1]:.4f}, pd={p_before[0]:.4f}\n"
                      f"Après: pu={p_after[2]:.4f}, pm={p_after[1]:.4f}, pd={p_after[0]:.4f}"),
                     fontsize=9, color="darkred", ha="left", va="top",
                     bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

        ax.set_title(f"Arbre trinomial (N={N}) — mode {'FAST' if use_fast else 'FULL'}")
        ax.set_xlabel("Étape (i)")
        ax.set_ylabel("Sous-jacent S(i,j)")
        ax.grid(True, linestyle=":", alpha=0.4)

        ax.set_xlim(-0.2, N + 0.2)
        ymin = min(v.under for col in self.levels.values() for v in col.values())
        ymax = max(v.under for col in self.levels.values() for v in col.values())
        marg = 0.04*(ymax - ymin) if ymax>ymin else 1.0
        ax.set_ylim(ymin - marg, ymax + marg)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor="white")
        plt.show()