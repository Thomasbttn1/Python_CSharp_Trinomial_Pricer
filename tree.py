# tree.py
from functools import lru_cache
from node import Node
import math
import numpy as np
import matplotlib.pyplot as plt
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

        self.prune_threshold = 10**(-7)

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

    def _clear_values(self):
        """Remet .value = None sur tous les nœuds (pour mémo récursive propre)."""
        N = self.nb_steps
        for i in range(0, N + 1):
            head_i = self._head_last if (i == N and self._head_last is not None) else self._level_head(i)
            for nd in head_i.iter_right():
                nd.value = None

    def _should_prune(self, reach_prob: float) -> bool:
        return (self.prune_threshold is not None) and (reach_prob < self.prune_threshold)
    # ---------- construction de l'arbre ----------
    def build_bottom_first(self, option):
        """
        Méthode de construction : approche "colonne par colonne" (sans liste ni dictionnaire).
        -------------------------------------------------------------------------------
        - On part du tronc (racine) : le nœud (i=0, j=0) avec S0.
        - À chaque étape i :
            On calcule le forward local (fwd_mid) centré sur la valeur médiane courante.
            On crée la colonne i (chaînée gauche → droite) :
                - Chaque nœud est relié à son voisin via .right et .left
                - Ses prix sont espacés géométriquement avec le facteur alpha.
            On relie la colonne i−1 à la colonne i :
                - Chaque nœud parent pointe vers ses trois enfants : down, mid, up.
            Si un dividende est payé à cette étape :
                - On calcule les probabilités (p_down, p_mid, p_up) pour chaque nœud source.
                - Ces proba sont stockées directement dans les nœuds sources.
            On repère le nœud dont le prix est le plus proche du forward local
                (c’est notre “centre” du pas suivant).
        - On répète jusqu’à atteindre la profondeur N.
        - À la fin, self._head_last pointe sur la colonne finale (pour le pricing).
        -------------------------------------------------------------------------------
        Cette méthode ne crée aucune liste ni dictionnaire :
        l’arbre est formé uniquement par une succession de nœuds reliés entre eux.
        """

        # Factorisation: pré-calculs et boucle principale délégués à des helpers
        N, dt, a = self.nb_steps, self.delta_t, self.alpha
        S0, r, sigma = self.market.underlying, self.market.rate, self.market.vol
        i_div, D = self._div_step_index(option)

        exp_rdt, exp_2rdt, var_factor = self._build_precomputations(dt, sigma, r)

        self.trinomial_prob_no_div()  # calcule la triplette globale no-div
        current_mid = S0
        prev_head = None

        for i in range(1, N + 1):
            D_next = D if (i_div is not None and i == i_div) else 0.0
            fwd_mid = current_mid * exp_rdt - D_next

            head_i, nxt_mid_S = self._create_column(i, fwd_mid, a)
            self._link_parent_to_child(prev_head, head_i)

            if D_next != 0.0:
                self._assign_dividend_probs(D_next, prev_head, exp_rdt, exp_2rdt, var_factor, a)

            prev_head = head_i
            current_mid = nxt_mid_S

        self._head_last = prev_head
        return self

    # helpers pour build_bottom_first: tous privés et courts
    def _build_precomputations(self, dt, sigma, r):
        exp_rdt = math.exp(r * dt)
        exp_2rdt = math.exp(2.0 * r * dt)
        var_factor = math.exp((sigma ** 2) * dt) - 1.0
        return exp_rdt, exp_2rdt, var_factor

    def _create_column(self, i, fwd_mid, a):
        head_i = Node(self, i, -i)
        head_i.under = fwd_mid * (a ** (-i))

        cur = head_i
        nxt_mid_node = head_i
        min_dev = abs(head_i.under - fwd_mid)

        for j in range(-i + 1, i + 1):
            nd = Node(self, i, j)
            nd.under = fwd_mid * (a ** j)
            cur.right = nd
            nd.left = cur
            cur = nd

            dev = abs(nd.under - fwd_mid)
            if dev < min_dev:
                min_dev = dev
                nxt_mid_node = nd

        nxt_mid_node.forward = fwd_mid
        nxt_mid_S = nxt_mid_node.under
        return head_i, nxt_mid_S

    def _link_parent_to_child(self, prev_head, head_i):
        if prev_head is None:
            parent = self.root
            child = head_i
            parent.down = child
            parent.mid = child.right
            parent.up = child.right.right
            return

        parent = prev_head
        child = head_i
        while parent is not None:
            parent.down = child
            parent.mid = child.right
            parent.up = child.right.right
            parent = parent.right
            child = child.right

    def _assign_dividend_probs(self, D_next, prev_head, exp_rdt, exp_2rdt, var_factor, a):
        if prev_head is None:
            sources = [self.root]
        else:
            sources = list(prev_head.iter_right())

        for nd_prev in sources:
            S_prev = nd_prev.under
            E_j = S_prev * exp_rdt - D_next
            V_j = (S_prev ** 2) * exp_2rdt * var_factor
            mid_child = nd_prev.mid
            p_d, p_m, p_u = self._solve_trinomial_probs(E_j, V_j, mid_child.under, a)
            nd_prev.pd, nd_prev.pm, nd_prev.pu = p_d, p_m, p_u

    # ---------- pricing ----------

    def price(self, option, engine: str = "backward", style: str = "european"):
        """
        Point d'entrée unique pour pricer:
        - engine:   "backward" | "recursive"
        - style:    "european" | "american"
        Redirige vers la bonne méthode spécialisée.
        """
        engine = engine.lower().strip()
        style  = style.lower().strip()

        if style == "european" and engine == "backward":
            return self.price_european_backward(option)
        elif style == "european" and engine == "recursive":
            return self.price_european_recursive(option)
        elif style == "american" and engine == "backward":
            return self.price_american_backward(option)
        elif style == "american" and engine == "recursive":
            return self.price_american_recursive(option)
        else:
            raise ValueError("Combinaison (engine, style) invalide. "
                             "engine ∈ {'backward','recursive'}, style ∈ {'european','american'}.")

    def _probas_for_node(self, nd):
        """
        Renvoie (pd, pm, pu) applicables au step sortant du nœud nd.
        - Si nd a des probas locales (ex-div), on les utilise.
        - Sinon, on prend la triplette globale "no-div".
        """
        if nd.pd is not None and nd.pm is not None and nd.pu is not None:
            return nd.pd, nd.pm, nd.pu
        return self._pd_global, self._pm_global, self._pu_global

    def price_european_backward(self, option):
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
                pd_nd, pm_nd, pu_nd = self._probas_for_node(nd)
                v_up   = nd.up.value
                v_mid  = nd.mid.value
                v_down = nd.down.value
                nd.value = DF * (pu_nd * v_up + pm_nd * v_mid + pd_nd * v_down)

        return self.root.value

    def price_european_recursive(self, option):
        """
        Récursif mémoïsé directement sur les nœuds (pas de lru_cache).
        → beaucoup moins d'overhead que de hasher des objets Node.
        """
        DF = math.exp(-self.market.rate * self.delta_t)
        payoff = option.payoff
        probas = self._probas_for_node

        self._clear_values()

        def V(nd: Node):
            # mémo local
            if nd.value is not None:
                return nd.value
            # terminal: pas d'enfants
            if nd.down is None:
                val = payoff(nd.under)
                nd.value = val
                return val
            pd_nd, pm_nd, pu_nd = probas(nd)
            up, mid, down = nd.up, nd.mid, nd.down
            cont = DF * (pu_nd * V(up) + pm_nd * V(mid) + pd_nd * V(down))
            nd.value = cont
            return cont

        return V(self.root)

    def price_american_recursive(self, option):
        """
        Récursif mémoïsé directement sur les nœuds (pas de lru_cache), style américain.
        """
        DF = math.exp(-self.market.rate * self.delta_t)
        payoff = option.payoff
        probas = self._probas_for_node

        self._clear_values()

        def V(nd: Node):
            if nd.value is not None:
                return nd.value
            if nd.down is None:
                val = payoff(nd.under)
                nd.value = val
                return val
            pd_nd, pm_nd, pu_nd = probas(nd)
            up, mid, down = nd.up, nd.mid, nd.down
            cont = DF * (pu_nd * V(up) + pm_nd * V(mid) + pd_nd * V(down))
            exer = payoff(nd.under)
            val = cont if cont >= exer else exer
            nd.value = val
            return val

        return V(self.root)
    
    def price_american_backward(self, option):
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
                pd_nd, pm_nd, pu_nd = self._probas_for_node(nd)  # ordre correct des probas
                hold = DF * (
                    pu_nd * nd.up.value +
                    pm_nd * nd.mid.value +
                    pd_nd * nd.down.value
                )
                exercise = option.payoff(nd.under)
                nd.value = max(exercise, hold)
        return self.root.value

    def plot_pointer(self, annotate=True, figsize=(10, 6), dpi=110):
        """
        Trace l'arbre en parcourant les pointeurs (root, right, down/mid/up).
        Compatible avec la version sans listes/dictionnaires.
        -----------------------------------------------------------------------
        - Chaque colonne i est parcourue via ._level_head(i) et .right
        - Les arêtes sont tracées vers .down, .mid et .up
        - Les nœuds sont affichés avec leurs prix sous-jacents
        -----------------------------------------------------------------------
        """

        N = self.nb_steps
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

        segments = []
        xs, ys = [], []

        # Parcours des colonnes 0..N
        for i in range(0, N + 1):
            # Tête de la colonne i
            head_i = self.root if i == 0 else self._level_head(i)
            nd = head_i

            # Parcours horizontal
            while nd is not None:
                xs.append(i)
                ys.append(nd.under)

                # Liaisons vers la colonne suivante
                if i < N:
                    if nd.down is not None:
                        segments.append([(i, nd.under), (i + 1, nd.down.under)])
                    if nd.mid is not None:
                        segments.append([(i, nd.under), (i + 1, nd.mid.under)])
                    if nd.up is not None:
                        segments.append([(i, nd.under), (i + 1, nd.up.under)])

                nd = nd.right  # passer au noeud suivant

        # Arêtes
        lc = LineCollection(segments, colors="gray", linewidths=0.7, alpha=0.7)
        ax.add_collection(lc)

        # Nœuds
        ax.scatter(xs, ys, s=20, color="C0", zorder=3)
        if annotate:
            for (x, y) in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=8)

        # Mise en forme
        ax.set_title(f"Arbre trinomial (N={N}) — version pointeurs")
        ax.set_xlabel("Étape (i)")
        ax.set_ylabel("Sous-jacent S(i,j)")
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xlim(-0.2, N + 0.2)

        ymin, ymax = min(ys), max(ys)
        marg = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
        ax.set_ylim(ymin - marg, ymax + marg)

        plt.tight_layout()
        plt.show()

    # ---------- greeks (à la racine, sans bump) ----------
    def local_greeks_no_bump(self, option):
        """
        Delta/Gamma à la racine (colonne 1 du tree) + Vega (proxy Black-Scholes, sans bump).

        Cette fonction s'assure que les valeurs node.value nécessaires sont remplies
        en appelant automatiquement le pricer backward correspondant au style de
        l'option (`option.type` attendu : 'european' ou 'american').

        Prérequis :
          - L'arbre est construit (build_bottom_first(option))
          - option.type doit être 'european' ou 'american' (par défaut 'european')
          - option.strike existe
        """
        import math

        dt = self.delta_t
        r = self.market.rate

        # Détermine le style à utiliser pour remplir .value
        style = getattr(option, "type", "european")
        if style is None:
            style = "european"
        style = str(style).lower().strip()
        if style not in ("european", "american"):
            raise ValueError("option.type doit être 'european' ou 'american'.")

        # Remplit les .value du tree en exécutant le pricer backward adapté
        # (on force engine='backward' car nous lisons les valeurs des enfants immédiats)
        try:
            # price() mettra à jour .value sur tous les nœuds via l'algorithme backward
            self.price(option, engine="backward", style=style)
        except Exception as e:
            raise RuntimeError(f"Impossible de calculer les valeurs du tree pour calculer les grecs : {e}") from e

        # enfants de la racine
        up = self.root.up
        mid = self.root.mid
        down = self.root.down
        if up is None or mid is None or down is None:
            raise RuntimeError("Construis d'abord l'arbre (build_bottom_first) avant de calculer les grecs.")

        S_up,  S_mid,  S_down  = up.under,  mid.under,  down.under
        V_up,  V_mid,  V_down  = up.value,  mid.value,  down.value
        V0 = self.root.value

        # --- Delta
        denom = (S_up - S_down)
        if abs(denom) < 1e-15:
            raise ZeroDivisionError("Maille du 1er étage dégénérée.")
        delta = (V_up - V_down) / denom

        # --- Gamma (maille non uniforme ok)
        h_up = (S_up - S_mid)
        h_dn = (S_mid - S_down)
        if abs(h_up) < 1e-15 or abs(h_dn) < 1e-15:
            raise ZeroDivisionError("Maille du 1er étage trop fine.")
        gamma = 2.0 * ((V_up - V_mid) / h_up - (V_mid - V_down) / h_dn) / (h_up + h_dn)

        # --- Vega (proxy Black-Scholes, S = spot racine, T = nb_steps*dt)
        try:
            K = float(option.strike)
        except Exception as e:
            raise AttributeError("L'option doit avoir un attribut 'strike' pour calculer la vega (proxy BS).") from e

        S0     = float(self.root.under)
        sigma  = float(self.market.vol)
        T_total = max(1e-12, float(self.nb_steps) * float(dt))  # maturité totale
        if sigma <= 0.0:
            raise ValueError("market.vol doit être > 0 pour calculer la vega (proxy BS).")

        # densité normale standard
        def _phi(x):
            return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)

        # d1 et vega BS (identique call/put)
        d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T_total) / (sigma * math.sqrt(T_total))
        vega = S0 * math.sqrt(T_total) * _phi(d1) / 100  # non annualisée (≈ ∂V/∂σ)

        return {"price": V0, "delta": delta, "gamma": gamma, "vega": vega}