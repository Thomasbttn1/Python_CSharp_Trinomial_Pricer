# tree.py
from node import Node
import math
import numpy as np
import matplotlib.pyplot as plt
from functools import lru_cache
from matplotlib.collections import LineCollection

class Tree:
    """
    Arbre trinomial recombinant sans listes/dictionnaires :
    - Seuls des nœuds "chaînés" sont utilisés.
    - On garde la racine et on parcourt via pointeurs.
    - Les probas "no-div" sont globales (triplette scalaire).
    - Les probas "ex-div" sont stockées sur chaque nœud source (pd/pm/pu).
    """

    def __init__(self, market, nb_steps: int, delta_t: float):
        self.market = market
        self.nb_steps = nb_steps
        self.delta_t = delta_t

        self.alpha = math.exp(market.vol * math.sqrt(3.0 * delta_t))
        self.root = Node(self, 0, 0)
        self.root.under = market.underlying
        self.root.forward = market.underlying  # au départ

        # Probas "no-div" globales (remplies à la 1ère demande)
        self._pd_global = None
        self._pm_global = None
        self._pu_global = None

        # Pointeur (optionnel) vers la tête de la colonne finale (N)
        self._head_last = None

    # ---------- utilitaires ----------
    def trinomial_prob_no_div(self):
        """
        Probabilités fermées quand D = 0 sur un step.
        Renvoie (p_down, p_mid, p_up) et met en cache dans l'objet.
        """
        if self._pd_global is None:
            a = self.alpha
            v = math.exp((self.market.vol ** 2) * self.delta_t) - 1.0
            denom = (1.0 - a) * ((a**-2) - 1.0)
            p_down = v / denom
            p_up   = p_down / a
            p_mid  = 1.0 - p_up - p_down
            self._pd_global, self._pm_global, self._pu_global = p_down, p_mid, p_up
        return self._pd_global, self._pm_global, self._pu_global

    def _solve_trinomial_probs(self, E_target: float, V_next: float, nxt_mid_S: float, alpha: float):
        """
        Résout (p_down, p_mid, p_up) à partir d'E[S] et Var[S] conditionnels
        avec (Sup, Smid, Sdown) = (nxt_mid*a, nxt_mid, nxt_mid/a).
        """
        S_mid = float(nxt_mid_S)
        a     = float(alpha)
        inv_a = 1.0 / a

        num = (V_next + E_target * E_target) / (S_mid * S_mid) - 1.0 - (a + 1.0) * ((E_target / S_mid) - 1.0)
        den = (1.0 - a) * ((inv_a * inv_a) - 1.0)
        p_down = num / den

        rhs   = (E_target / S_mid) - 1.0
        p_up  = (rhs - (inv_a - 1.0) * p_down) / (a - 1.0)
        p_mid = 1.0 - p_up - p_down
        return p_down, p_mid, p_up

    def _div_step_index(self, option):
        """
        Mappe la date de dividende en indice de pas i_div (1..N).
        Retourne (i_div, D) ; i_div peut être None si pas de dividende dans [0,T].
        """
        if option is None or not getattr(option, "div", 0) or option.div_date is None:
            return None, 0.0
        t_div = option.get_div_time_in_years()
        T = self.nb_steps * self.delta_t
        if not (0.0 < t_div <= T + 1e-12):
            return None, 0.0
        i_div = int(math.ceil((t_div - 1e-12) / self.delta_t))
        i_div = max(1, min(self.nb_steps, i_div))
        return i_div, float(option.div)

    def _level_head(self, i: int):
        """
        Retourne la tête (gauche) de la colonne i en descendant 'down' i fois
        depuis la racine. Hypothèse : structure recombinante correctement liée.
        """
        nd = self.root
        for _ in range(i):
            nd = nd.down
        return nd

    # ---------- construction de l'arbre ----------
    def build_bottom_first(self, option):
        """
        Construction colonne par colonne sans containers :
        - Chaque colonne i est une liste chaînée (left/right).
        - Recombinaison par pointeurs down/mid/up depuis la colonne i-1 vers i.
        - Probas ex-div stockées sur les nœuds sources au step concerné.
        """
        N, dt, a = self.nb_steps, self.delta_t, self.alpha
        S0, r, sigma = self.market.underlying, self.market.rate, self.market.vol

        i_div, D = self._div_step_index(option)

        # force le calcul une fois des probas "no-div"
        self.trinomial_prob_no_div()

        current_mid = S0
        prev_head = None  # tête de la colonne i-1

        exp_rdt    = math.exp(r * dt)
        exp_2rdt   = math.exp(2.0 * r * dt)
        var_factor = math.exp((sigma ** 2) * dt) - 1.0

        for i in range(1, N + 1):
            # dividende payé sur (i-1 -> i) ?
            D_next = D if (i_div is not None and i == i_div) else 0.0

            # forward local net du dividende autour duquel on centre la colonne i
            fwd_mid = current_mid * exp_rdt - D_next

            # 1) Construire la colonne i (chaînée gauche->droite) et trouver le "mid" réel
            # j va de -i à +i ; S(i,j) = fwd_mid * a^j
            head_i = Node(self, i, -i)
            head_i.under = fwd_mid * (a ** (-i))

            cur = head_i
            nxt_mid_node = head_i
            min_dev = abs(head_i.under - fwd_mid)

            for j in range(-i + 1, i + 1):
                nd = Node(self, i, j)
                nd.under = fwd_mid * (a ** j)
                # chainage horizontal
                cur.right = nd
                nd.left = cur
                cur = nd

                # repérer le nœud le plus proche du forward
                dev = abs(nd.under - fwd_mid)
                if dev < min_dev:
                    min_dev = dev
                    nxt_mid_node = nd

            nxt_mid_node.forward = fwd_mid
            nxt_mid_S = nxt_mid_node.under

            # 2) Recombinaison : lier enfants depuis la colonne i-1 vers i
            if i == 1:
                # les enfants de la racine
                parent = self.root            # j=0
                child  = head_i               # j=-1 = (j_parent - 1)
                parent.down = child           # j-1
                parent.mid  = child.right     # j
                parent.up   = child.right.right  # j+1
            else:
                # on positionne le parent le plus à gauche (j=-(i-1))
                parent = prev_head
                # et le "curseur enfant" au j_parent-1 = -i
                child = head_i
                while parent is not None:
                    parent.down = child
                    parent.mid  = child.right
                    parent.up   = child.right.right
                    # avance : parent j -> j+1, child j-1 -> j
                    parent = parent.right
                    child  = child.right

            # 3) Probabilités au step (i-1 -> i)
            if D_next == 0.0:
                # step "no-div": rien à stocker localement (on utilisera la triplette globale)
                pass
            else:
                # ex-div : calculer E, V, et proba PAR nœUD SOURCE de la colonne i-1
                # tête de la colonne i-1 = prev_head (sauf i=1: parent unique = root)
                if i == 1:
                    # nœud source unique : root
                    sources = [self.root]
                else:
                    sources = list(prev_head.iter_right())

                for nd_prev in sources:
                    S_prev = nd_prev.under
                    E_j = S_prev * exp_rdt - D_next
                    V_j = (S_prev ** 2) * exp_2rdt * var_factor

                    mid_child = nd_prev.mid  # enfant "mid" (i, j_prev)
                    p_d, p_m, p_u = self._solve_trinomial_probs(E_j, V_j, mid_child.under, a)

                    # stocker sur le nœud source
                    nd_prev.pd, nd_prev.pm, nd_prev.pu = p_d, p_m, p_u

            # 4) préparer l'itération suivante
            prev_head = head_i
            current_mid = nxt_mid_S

        self._head_last = prev_head
        return self

    # ---------- pricing ----------
    def _probas_for_node(self, nd):
        """
        Renvoie (pd, pm, pu) applicables au step sortant du nœud nd.
        - Si nd a des probas locales (ex-div), on les utilise.
        - Sinon, on prend la triplette globale "no-div".
        """
        if nd.pd is not None and nd.pm is not None and nd.pu is not None:
            return nd.pd, nd.pm, nd.pu
        return self._pd_global, self._pm_global, self._pu_global

    def price_european(self, option):
        """
        Backward itératif (sans containers).
        Hypothèse : l'arbre a été construit via build_bottom_first(option).
        """
        N, dt = self.nb_steps, self.delta_t
        DF = math.exp(-self.market.rate * dt)

        # 1) Terminal : colonne N
        headN = self._head_last if self._head_last is not None else self._level_head(N)
        for nd in headN.iter_right():
            nd.value = option.payoff(nd.under)

        # 2) Backward i = N-1 .. 0
        for i in range(N - 1, -1, -1):
            head_i = self._level_head(i)
            for nd in head_i.iter_right():
                pu_nd, pm_nd, pd_nd = self._probas_for_node(nd)
                v_up   = nd.up.value
                v_mid  = nd.mid.value
                v_down = nd.down.value
                nd.value = DF * (pu_nd * v_up + pm_nd * v_mid + pd_nd * v_down)
        return self.root.value

    def price_european_recursive(self, option):
        """
        Version récursive avec memo implicite (lru_cache sur l'identité des nœuds).
        Sans containers (on part de la racine et on suit les pointeurs).
        """
        from functools import lru_cache
        N, dt = self.nb_steps, self.delta_t
        DF = math.exp(-self.market.rate * dt)

        @lru_cache(maxsize=None)
        def V(node_obj: Node):
            # terminal si pas d'enfants (colonne N)
            if node_obj.down is None:
                return option.payoff(node_obj.under)
            pd_nd, pm_nd, pu_nd = self._probas_for_node(node_obj)
            return DF * (
                pu_nd * V(node_obj.up) +
                pm_nd * V(node_obj.mid) +
                pd_nd * V(node_obj.down)
            )

        return V(self.root)

    def price_american(self, option):
        """
        Backward itératif avec exercice anticipé (PUT/option américaine).
        """
        N, dt = self.nb_steps, self.delta_t
        DF = math.exp(-self.market.rate * dt)

        # terminal
        headN = self._head_last if self._head_last is not None else self._level_head(N)
        for nd in headN.iter_right():
            nd.value = option.payoff(nd.under)

        # backward
        for i in range(N - 1, -1, -1):
            head_i = self._level_head(i)
            for nd in head_i.iter_right():
                pu_nd, pm_nd, pd_nd = self._probas_for_node(nd)
                hold = DF * (
                    pu_nd * nd.up.value +
                    pm_nd * nd.mid.value +
                    pd_nd * nd.down.value
                )
                exercise = option.payoff(nd.under)
                nd.value = max(exercise, hold)
        return self.root.value

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
    # ---------- greeks (à la racine, sans bump) ----------
    def local_greeks_no_bump(self, option):

        """
        Delta/Gamma/Theta à la racine en se basant sur la colonne 1.
        Nécessite que l'arbre soit construit ET que price_european(option) ait été appelé (pour remplir .value).
        """
        dt = self.delta_t
        r = self.market.rate

        # enfants de la racine
        up = self.root.up
        mid = self.root.mid
        down = self.root.down
        if up is None or mid is None or down is None:
            raise RuntimeError("Construis d'abord l'arbre et/ou appelle price_european(option).")

        S_up, S_mid, S_down = up.under, mid.under, down.under
        V_up, V_mid, V_down = up.value, mid.value, down.value
        V0 = self.root.value

        # Delta
        denom = (S_up - S_down)
        if abs(denom) < 1e-15:
            raise ZeroDivisionError("Maille du 1er étage dégénérée.")
        delta = (V_up - V_down) / denom

        # Gamma (maille non uniforme ok)
        h_up = (S_up - S_mid)
        h_dn = (S_mid - S_down)
        if abs(h_up) < 1e-15 or abs(h_dn) < 1e-15:
            raise ZeroDivisionError("Maille du 1er étage trop fine.")
        gamma = 2.0 * ((V_up - V_mid) / h_up - (V_mid - V_down) / h_dn) / (h_up + h_dn)

        # Theta (différence temporelle sur un pas)
        DF = math.exp(-r * dt)
        # Valeur "au pas suivant" au mid (attendue) ~ combi des enfants (déjà dans V_mid)
        theta = (V_mid - V0) / (-dt)

        return {"price": V0, "delta": delta, "gamma": gamma, "theta": theta}