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
        T = self.nb_steps * self.delta_t
        if not (0.0 < t_div <= T + 1e-12):
            return None, 0.0

        # Mapping par intervalles: ((i-1)*dt, i*dt]  -> i
        # => i_div = ceil(t_div / dt), avec une micro tolérance pour les bords
        i_div = int(math.ceil((t_div - 1e-12) / self.delta_t))
        i_div = max(1, min(self.nb_steps, i_div))  # clamp de sécurité
        return i_div, float(option.div)

    def trinomial_prob_no_div(self):
        """Probabilités fermées quand D=0 sur le pas (K-T)."""
        a = self.alpha
        v = math.exp((self.market.vol ** 2) * self.delta_t) - 1.0
        denom = (1.0 - a) * ((a**-2) - 1.0)
        p_down = v / denom
        p_up   = p_down / a
        p_mid  = 1.0 - p_up - p_down
        return p_down, p_mid, p_up

    def _solve_trinomial_probs(self, E_target: float, V_next: float, nxt_mid_S: float, alpha: float):
        """
        Résout (p_up, p_mid, p_down) en imposant :
          somme = 1
          E[S_{t+dt}|S_t]  = fwd = Si' * e^{r dt} - D_next
          Var[S_{t+dt}|S_t]= Si'^2 e^{2 r dt} (e^{σ² dt} - 1)
        avec (Sup, Smid, Sdown) = (nxt_mid*a, nxt_mid, nxt_mid/a).
        """
        S_mid = float(nxt_mid_S)
        a     = float(alpha)
        inv_a = 1.0 / a

        num = (V_next + E_target * E_target) / (S_mid * S_mid) - 1.0 - (a + 1.0) * ((E_target / S_mid) - 1.0)
        den = (1.0 - a) * ((inv_a * inv_a) - 1.0)
        p_down = num / den

        # Équation d'espérance sur S_{i+1}/S_mid :
        # (a - 1)*p_up + (a^-1 - 1)*p_down = E/S_mid - 1
        rhs   = (E_target / S_mid) - 1.0
        p_up  = (rhs - (inv_a - 1.0) * p_down) / (a - 1.0)
        p_mid = 1.0 - p_up - p_down

        # Petit clip défensif pour les bords numériques
        """
        eps = 1e-12
        p_down = 0.0 if p_down < -eps else (1.0 if p_down > 1.0 + eps else max(0.0, min(1.0, p_down)))
        p_up   = 0.0 if p_up   < -eps else (1.0 if p_up   > 1.0 + eps else max(0.0, min(1.0, p_up)))
        p_mid  = 1.0 - p_up - p_down  # renormalise après clip doux
        """

        return p_down, p_mid, p_up

    # ---------- construction de l'arbre ----------
    def build_bottom_first(self, option):
        """
        Arbre trinomial recombinant, centré à chaque pas sur le forward local :
            f_i = S_mid * e^{r dt} - D_i
        (Le dividende est pris dans le forward, pas en baissant S_mid directement.)
        """
        N, dt, a = self.nb_steps, self.delta_t, self.alpha
        S0, r = self.market.underlying, self.market.rate
        sigma = self.market.vol

        i_div, D = self._div_step_index(option)

        # racine
        self.levels = {0: {0: self.root}}
        self.root.under = S0
        self.root.forward = S0

        # containers probas
        self.step_probs = {}
        self.step_probs_by_node = {}

        current_mid = S0  # centre de la colonne précédente

        for i in range(1, N + 1):
            self.levels[i] = {}

            # dividende payé sur (i-1 -> i) ?
            D_next = D if (i_div is not None and i == i_div) else 0.0

            # forward local net du dividende
            fwd_mid = current_mid * math.exp(r * dt) - D_next

            # construire la colonne i autour du forward théorique (géométrique via alpha)
            for j in range(-i, i + 1):
                nd = Node(self, i, j)
                nd.under = fwd_mid * (a ** j)
                self.levels[i][j] = nd

            # choisir le vrai "mid" (noeud le plus proche du forward)
            nxt_mid_node = self.pick_next_mid_closest_to_forward(self.levels[i], fwd_mid)
            nxt_mid_S = nxt_mid_node.under
            nxt_mid_node.forward = fwd_mid

            # calcul des probas du pas (i-1 -> i)
            if D_next == 0.0:
                # cas sans dividende : probas identiques pour tous les nœuds
                p_d, p_m, p_u = self.trinomial_prob_no_div()
                self.step_probs[i - 1] = (p_d, p_m, p_u)
            else:
                # cas ex-div : probas PAR NŒUD de la colonne précédente
                col_prev = self.levels[i - 1]
                col_curr = self.levels[i]
                self.step_probs_by_node[i - 1] = {}
                representative = None  # pour compatibilité du plot

                for j_prev, nd_prev in col_prev.items():
                    S_prev = nd_prev.under

                    # Espérance et variance conditionnelles depuis ce nœud
                    E_j = S_prev * math.exp(r * dt) - D_next
                    V_j = (S_prev ** 2) * math.exp(2.0 * r * dt) * (math.exp((sigma ** 2) * dt) - 1.0)

                    # Le "mid" enfant de (i-1, j_prev) est (i, j_prev) dans une structure recombinante
                    mid_child = col_curr.get(j_prev)
                    if mid_child is None:
                        continue

                    S_mid_child = mid_child.under

                    # Formules fermées : appel à la méthode interne existante
                    p_d, p_m, p_u = self._solve_trinomial_probs(E_j, V_j, S_mid_child, a)

                    # Stockage par nœud
                    self.step_probs_by_node[i - 1][j_prev] = (p_d, p_m, p_u)

                    # Triplette représentative pour le plot (priorité j=0)
                    if representative is None or j_prev == 0:
                        representative = (j_prev, (p_d, p_m, p_u))

                # Pour compatibilité avec le plot (ligne rouge / cartouche)
                if representative is not None:
                    self.step_probs[i - 1] = representative[1]
                else:
                    self.step_probs[i - 1] = (0.0, 1.0, 0.0)  # fallback si rien

            # préparer pour le pas suivant (mid réel)
            current_mid = nxt_mid_S

        return self

    # ---------- pricing ----------
    def _get_probs(self, i, j):
        """
        Retourne (pd, pm, pu) pour le step i -> i+1.
        - Priorité aux probas par nœud (ex-div): self.step_probs_by_node[i][j]
        - Sinon fallback aux probas du step:     self.step_probs[i]
        """
        by_node = getattr(self, "step_probs_by_node", {}).get(i)
        if by_node is not None and j in by_node:
            return by_node[j]
        return self.step_probs[i]

    def price_european(self, option):
        """Backward induction, en tenant compte des probas par nœud au pas ex-div."""
        N, dt = self.nb_steps, self.delta_t
        DF = math.exp(-self.market.rate * dt)

        values = {i: {j: 0.0 for j in range(-i, i + 1)} for i in range(N + 1)}
        for j, nd in self.levels[N].items():
            values[N][j] = option.payoff(nd.under)

        for i in range(N - 1, -1, -1):
            for j in range(-i, i + 1):
                pd, pm, pu = self._get_probs(i, j)
                v_up   = values[i + 1][j + 1]
                v_mid  = values[i + 1][j]
                v_down = values[i + 1][j - 1]
                values[i][j] = DF * (pu * v_up + pm * v_mid + pd * v_down)

        return values[0][0]

    def price_european_recursive(self, option):
        """Version récursive (lru_cache), avec probas par nœud au pas ex-div."""

        if not (self.levels and len(self.levels) == (self.nb_steps + 1)):
            raise ValueError("Construis d'abord l'arbre (build_bottom_first).")

        N, dt = self.nb_steps, self.delta_t
        DF = math.exp(-self.market.rate * dt)

        @lru_cache(maxsize=None)
        def V(i: int, j: int) -> float:
            if i == N:
                return option.payoff(self.levels[i][j].under)
            pd, pm, pu = self._get_probs(i, j)
            return DF * (pu * V(i + 1, j + 1) + pm * V(i + 1, j) + pd * V(i + 1, j - 1))

        return V(0, 0)

    def price_american(self, option):
        """Backward induction avec exercice anticipé + probas par nœud au pas ex-div."""
        import math
        N, dt = self.nb_steps, self.delta_t
        DF = math.exp(-self.market.rate * dt)

        values = {i: {j: 0.0 for j in range(-i, i + 1)} for i in range(N + 1)}
        for j, nd in self.levels[N].items():
            values[N][j] = option.payoff(nd.under)

        for i in range(N - 1, -1, -1):
            for j, nd in self.levels[i].items():
                pd, pm, pu = self._get_probs(i, j)
                hold = DF * (
                    pu * values[i + 1][j + 1] +
                    pm * values[i + 1][j] +
                    pd * values[i + 1][j - 1]
                )
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
    def plot(self, annotate=True, figsize=(9, 6), dpi=110, save_path=None,
             show_edge_probs=True, edge_prob_digits=3):
        """
        Trace l'arbre trinomial.
        - En FULL (N<=20) : affiche les valeurs nodales et les proba sur CHAQUE arête.
          * Au pas ex-div : lit self.step_probs_by_node[i][j] (proba par nœud source).
          * Sinon : fallback self.step_probs[i] (triplette globale du step).
        - Les proba sont placées juste AU-DESSUS du trait correspondant.
          Couleurs: down=red, mid=orange, up=green.
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection

        if not hasattr(self, "levels"):
            raise ValueError("Il faut d'abord construire l'arbre.")

        N = self.nb_steps
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        # Mode adaptatif
        use_fast = N > 20
        annotate_nodes = annotate and (not use_fast)
        show_edge_probs = bool(show_edge_probs and (not use_fast))

        # 1) Arêtes (+ proba sur arêtes en FULL)
        segments = []
        if not use_fast:
            for i in range(N):  # edges from column i -> i+1
                col_i = self.levels[i]
                col_ip1 = self.levels[i + 1]

                # proba: par nœud prioritaire (ex-div), sinon globale (step)
                by_node = getattr(self, "step_probs_by_node", {}).get(i)
                by_step = getattr(self, "step_probs", {}).get(i)

                for j, nd in col_i.items():
                    y = nd.under
                    for dj, idx in ((-1, 0), (0, 1), (+1, 2)):  # down, mid, up
                        j2 = j + dj
                        nd2 = col_ip1.get(j2)
                        if nd2 is None:
                            continue

                        # segment
                        segments.append([(i, y), (i + 1, nd2.under)])

                        # proba juste AU-DESSUS du trait
                        if show_edge_probs:
                            p = None
                            if by_node is not None and j in by_node:
                                p = by_node[j][idx]
                            elif by_step is not None:
                                p = by_step[idx]

                            if p is not None:
                                x_mid = i + 0.5
                                y_mid = 0.5 * (y + nd2.under)

                                # petit décalage horizontal pour distinguer down/mid/up
                                dx_map = {-1: -8, 0: 0, 1: +8}
                                dy = +8  # au-dessus du trait (en pixels écran)
                                color_map = {-1: "red", 0: "orange", 1: "green"}
                                base_color = color_map[dj]
                                color = "red" if (p < 0 or p > 1) else base_color

                                ax.annotate(f"{p:.{edge_prob_digits}f}",
                                            (x_mid, y_mid),
                                            textcoords="offset points",
                                            xytext=(dx_map[dj], dy),
                                            ha="center", va="bottom",
                                            fontsize=8, color=color,
                                            bbox=dict(boxstyle="round,pad=0.2",
                                                      facecolor="white", alpha=0.9,
                                                      linewidth=0))
        else:
            # FAST : sous-échantillonnage, pas d’annotations d’arêtes
            col_stride = max(1, N // 40)
            for i in range(0, N, col_stride):
                col_i = self.levels[i]
                ip1 = min(i + 1, N)
                col_ip1 = self.levels[ip1]
                row_stride = max(1, (i + 1) // 10)
                for j in range(-i, i + 1, row_stride):
                    nd = col_i.get(j)
                    if nd is None:
                        continue
                    y = nd.under
                    for dj in (-1, 0, +1):
                        j2 = j + dj
                        nd2 = col_ip1.get(j2)
                        if nd2 is None:
                            continue
                        segments.append([(i, y), (ip1, nd2.under)])

        # Ajout des arêtes
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
                if i % max(1, N // 40): continue
                row_stride = max(1, (i + 1) // 10)
                for j in range(-i, i + 1, row_stride):
                    nd = col.get(j)
                    if nd is None: continue
                    xs.append(i); ys.append(nd.under)
        ax.scatter(xs, ys, s=8 if not use_fast else 4, color="C0", zorder=3, rasterized=True)

        # 3) Annotations des valeurs nodales
        if annotate_nodes:
            for i, col in self.levels.items():
                for j, nd in col.items():
                    if nd.under is None: continue
                    ax.annotate(f"{nd.under:.2f}", (i, nd.under),
                                textcoords="offset points", xytext=(0, 6),
                                ha="center", fontsize=8)

        # 4) Ligne rouge verticale (repère ex-div si proba changent d’un step au suivant)
        i_div = None
        if hasattr(self, "step_probs") and self.step_probs:
            for i in range(1, N):
                prev = self.step_probs.get(i - 1)
                cur = self.step_probs.get(i)
                if prev and cur and any(abs(a - b) > 1e-9 for a, b in zip(prev, cur)):
                    i_div = i; break
        if i_div is not None:
            ax.axvline(i_div, color="red", linestyle="--", alpha=0.6, linewidth=1.0)

        # Mise en forme
        ax.set_title(f"Arbre trinomial (N={N}) — mode {'FAST' if use_fast else 'FULL'}")
        ax.set_xlabel("Étape (i)")
        ax.set_ylabel("Sous-jacent S(i,j)")
        ax.grid(True, linestyle=":", alpha=0.4)

        ax.set_xlim(-0.2, N + 0.2)
        ymin = min(v.under for col in self.levels.values() for v in col.values())
        ymax = max(v.under for col in self.levels.values() for v in col.values())
        marg = 0.04 * (ymax - ymin) if ymax > ymin else 1.0
        ax.set_ylim(ymin - marg, ymax + marg)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=dpi, facecolor="white")
        plt.show()