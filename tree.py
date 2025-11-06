# tree.py  # arbre trinomial par pointeurs
from functools import lru_cache  # utilitaire cache
from node import Node  # nœud de l’arbre
import math  # fonctions math
import numpy as np  # calculs numériques
import matplotlib.pyplot as plt  # tracés
from matplotlib.collections import LineCollection   # segments pour plots

class Tree:
    """
    Arbre trinomial recombinant sans listes/dictionnaires :
    - Seuls des nœuds "chaînés" sont utilisés.
    - On garde la racine et on parcourt via pointeurs.
    - Les probas "no-div" sont globales (triplette scalaire).
    - Les probas "ex-div" sont stockées sur chaque nœud source (pd/pm/pu).
    """

    def __init__(self, market, nb_steps: int, delta_t: float):  # init du tree
        self.market = market  # marché
        self.nb_steps = nb_steps  # nombre de pas
        self.delta_t = delta_t  # taille de pas

        self.alpha = math.exp(market.vol * math.sqrt(3.0 * delta_t))  # facteur u
        self.root = Node(self, 0, 0)  # racine
        self.root.under = market.underlying  # S0
        self.root.forward = market.underlying  # forward initial

        # Probas "no-div" globales (remplies à la 1ère demande)
        self._pd_global = None  # p_down global
        self._pm_global = None  # p_mid global
        self._pu_global = None  # p_up global

        # Pointeur (optionnel) vers la tête de la colonne finale (N)
        self._head_last = None  # tête colonne N

        self.prune_threshold = 10**(-7)  # seuil de pruning

    # ---------- utilitaires ----------
    def trinomial_prob_no_div(self):  # calcule p_down/mid/up sans div
        """
        Probabilités fermées quand D = 0 sur un step.
        Renvoie (p_down, p_mid, p_up) et met en cache dans l'objet.
        """
        if self._pd_global is None:  # calcule une fois
            a = self.alpha  # facteur
            v = math.exp((self.market.vol ** 2) * self.delta_t) - 1.0  # var factor
            denom = (1.0 - a) * ((a**-2) - 1.0)  # dénominateur
            p_down = v / denom  # p_down
            p_up   = p_down / a  # p_up
            p_mid  = 1.0 - p_up - p_down  # p_mid
            self._pd_global, self._pm_global, self._pu_global = p_down, p_mid, p_up  # cache
        return self._pd_global, self._pm_global, self._pu_global  # retour

    def _solve_trinomial_probs(self, E_target: float, V_next: float, nxt_mid_S: float, alpha: float):  # probas ex-div
        """
        Résout (p_down, p_mid, p_up) à partir d'E[S] et Var[S] conditionnels
        avec (Sup, Smid, Sdown) = (nxt_mid*a, nxt_mid, nxt_mid/a).
        """
        S_mid = float(nxt_mid_S)  # Smid
        a     = float(alpha)  # facteur
        inv_a = 1.0 / a  # inverse

        num = (V_next + E_target * E_target) / (S_mid * S_mid) - 1.0 - (a + 1.0) * ((E_target / S_mid) - 1.0)  # num
        den = (1.0 - a) * ((inv_a * inv_a) - 1.0)  # den
        p_down = num / den  # p_down

        rhs   = (E_target / S_mid) - 1.0  # rhs
        p_up  = (rhs - (inv_a - 1.0) * p_down) / (a - 1.0)  # p_up
        p_mid = 1.0 - p_up - p_down  # p_mid
        return p_down, p_mid, p_up  # triplette

    def _div_step_index(self, option):  # mappe la date de div en index de pas
        """
        Mappe la date de dividende en indice de pas i_div (1..N).
        Retourne (i_div, D) ; i_div peut être None si pas de dividende dans [0,T].
        """
        if option is None or not getattr(option, "div", 0) or option.div_date is None:  # pas de div
            return None, 0.0  # rien
        t_div = option.get_div_time_in_years()  # temps du div
        T = self.nb_steps * self.delta_t  # maturité totale
        if not (0.0 < t_div <= T + 1e-12):  # hors fenêtre
            return None, 0.0  # ignore
        i_div = int(math.ceil((t_div - 1e-12) / self.delta_t))  # index
        i_div = max(1, min(self.nb_steps, i_div))  # clamp
        return i_div, float(option.div)  # retour

    def _level_head(self, i: int):  # retourne la tête de colonne i
        """
        Retourne la tête (gauche) de la colonne i en descendant 'down' i fois
        depuis la racine. Hypothèse : structure recombinante correctement liée.
        """
        nd = self.root  # départ
        for _ in range(i):  # descend
            nd = nd.down  # vers bas
        return nd  # tête

    def _clear_values(self):  # reset des valeurs de nœuds
        """Remet .value = None sur tous les nœuds (pour mémo récursive propre)."""
        N = self.nb_steps  # profondeur
        for i in range(0, N + 1):  # colonnes
            head_i = self._head_last if (i == N and self._head_last is not None) else self._level_head(i)  # tête
            for nd in head_i.iter_right():  # parcours droite
                nd.value = None  # reset

    def _should_prune(self, reach_prob: float) -> bool:  # test de pruning
        return (self.prune_threshold is not None) and (reach_prob < self.prune_threshold)  # bool
    # ---------- construction de l'arbre ----------
    def build_bottom_first(self, option):  # construit colonne par colonne
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
        N, dt, a = self.nb_steps, self.delta_t, self.alpha  # raccourcis
        S0, r, sigma = self.market.underlying, self.market.rate, self.market.vol  # marché
        i_div, D = self._div_step_index(option)  # indice du div

        exp_rdt, exp_2rdt, var_factor = self._build_precomputations(dt, sigma, r)  # pré-calculs

        self.trinomial_prob_no_div()  # calcule la triplette globale no-div
        current_mid = S0  # centre courant
        prev_head = None  # colonne précédente

        for i in range(1, N + 1):  # boucle colonnes
            D_next = D if (i_div is not None and i == i_div) else 0.0  # div éventuel
            fwd_mid = current_mid * exp_rdt - D_next  # forward local

            head_i, nxt_mid_S = self._create_column(i, fwd_mid, a)  # colonne
            self._link_parent_to_child(prev_head, head_i)  # liaisons

            if D_next != 0.0:  # étape de div
                self._assign_dividend_probs(D_next, prev_head, exp_rdt, exp_2rdt, var_factor, a)  # probas

            prev_head = head_i  # avance
            current_mid = nxt_mid_S  # met à jour

        self._head_last = prev_head  # tête finale
        return self  # fluent

    # helpers pour build_bottom_first: tous privés et courts
    def _build_precomputations(self, dt, sigma, r):  # pré-calculs
        exp_rdt = math.exp(r * dt)  # e^{r dt}
        exp_2rdt = math.exp(2.0 * r * dt)  # e^{2 r dt}
        var_factor = math.exp((sigma ** 2) * dt) - 1.0  # var factor
        return exp_rdt, exp_2rdt, var_factor  # tuple

    def _create_column(self, i, fwd_mid, a):  # crée la colonne i
        head_i = Node(self, i, -i)  # premier nœud
        head_i.under = fwd_mid * (a ** (-i))  # sous-jacent

        cur = head_i  # curseur
        nxt_mid_node = head_i  # candidat centre
        min_dev = abs(head_i.under - fwd_mid)  # écart min

        for j in range(-i + 1, i + 1):  # reste de la colonne
            nd = Node(self, i, j)  # nœud
            nd.under = fwd_mid * (a ** j)  # sous-jacent
            cur.right = nd  # lien droite
            nd.left = cur  # lien gauche
            cur = nd  # avance

            dev = abs(nd.under - fwd_mid)  # écart au centre
            if dev < min_dev:  # meilleur centre
                min_dev = dev  # maj écart
                nxt_mid_node = nd  # maj centre

        nxt_mid_node.forward = fwd_mid  # forward au centre
        nxt_mid_S = nxt_mid_node.under  # S du centre
        return head_i, nxt_mid_S  # tête et S_mid

    def _link_parent_to_child(self, prev_head, head_i):  # relie i-1 à i
        if prev_head is None:  # cas racine
            parent = self.root  # parent
            child = head_i  # enfant tête
            parent.down = child  # bas
            parent.mid = child.right  # milieu
            parent.up = child.right.right  # haut
            return  # done

        parent = prev_head  # première source
        child = head_i  # premier enfant
        while parent is not None:  # parcours horizontal
            parent.down = child  # lien bas
            parent.mid = child.right  # lien mid
            parent.up = child.right.right  # lien haut
            parent = parent.right  # parent suivant
            child = child.right  # enfant suivant

    def _assign_dividend_probs(self, D_next, prev_head, exp_rdt, exp_2rdt, var_factor, a):  # probas ex-div
        if prev_head is None:  # si étape 1
            sources = [self.root]  # racine seule
        else:
            sources = list(prev_head.iter_right())  # tous les parents

        for nd_prev in sources:  # pour chaque source
            S_prev = nd_prev.under  # S(i,j)
            E_j = S_prev * exp_rdt - D_next  # espérance
            V_j = (S_prev ** 2) * exp_2rdt * var_factor  # variance
            mid_child = nd_prev.mid  # enfant mid
            p_d, p_m, p_u = self._solve_trinomial_probs(E_j, V_j, mid_child.under, a)  # probas
            nd_prev.pd, nd_prev.pm, nd_prev.pu = p_d, p_m, p_u  # stocke

    # ---------- pricing ----------

    def price(self, option, engine: str = "backward", style: str = "european"):  # routeur de pricing
        """
        Point d'entrée unique pour pricer:
        - engine:   "backward" | "recursive"
        - style:    "european" | "american"
        Redirige vers la bonne méthode spécialisée.
        """
        engine = engine.lower().strip()  # normalise
        style  = style.lower().strip()  # normalise

        if style == "european" and engine == "backward":  # EU backward
            return self.price_european_backward(option)  # appel
        elif style == "european" and engine == "recursive":  # EU récursif
            return self.price_european_recursive(option)  # appel
        elif style == "american" and engine == "backward":  # US backward
            return self.price_american_backward(option)  # appel
        elif style == "american" and engine == "recursive":  # US récursif
            return self.price_american_recursive(option)  # appel
        else:
            raise ValueError("Combinaison (engine, style) invalide. "
                             "engine ∈ {'backward','recursive'}, style ∈ {'european','american'}.")  # erreur

    def _probas_for_node(self, nd):  # renvoie (pd, pm, pu) applicables
        """
        Renvoie (pd, pm, pu) applicables au step sortant du nœud nd.
        - Si nd a des probas locales (ex-div), on les utilise.
        - Sinon, on prend la triplette globale "no-div".
        """
        if nd.pd is not None and nd.pm is not None and nd.pu is not None:  # locales
            return nd.pd, nd.pm, nd.pu  # locales
        return self._pd_global, self._pm_global, self._pu_global  # globales

    def price_european_backward(self, option):  # backward EU
        """
        Backward itératif (sans containers).
        Hypothèse : l'arbre a été construit via build_bottom_first(option).
        """
        N, dt = self.nb_steps, self.delta_t  # raccourcis
        DF = math.exp(-self.market.rate * dt)  # facteur d’actualisation

        # 1) Terminal : colonne N
        headN = self._head_last if self._head_last is not None else self._level_head(N)  # tête N
        for nd in headN.iter_right():  # parcours
            nd.value = option.payoff(nd.under)  # payoff

        # 2) Backward i = N-1 .. 0
        for i in range(N - 1, -1, -1):  # colonnes
            head_i = self._level_head(i)  # tête i
            for nd in head_i.iter_right():  # noeuds
                pd_nd, pm_nd, pu_nd = self._probas_for_node(nd)  # probas
                v_up   = nd.up.value  # V up
                v_mid  = nd.mid.value  # V mid
                v_down = nd.down.value  # V down
                nd.value = DF * (pu_nd * v_up + pm_nd * v_mid + pd_nd * v_down)  # valeur

        return self.root.value  # prix

    def price_european_recursive(self, option):  # récursif EU
        """
        Récursif mémoïsé directement sur les nœuds (pas de lru_cache).
        → beaucoup moins d'overhead que de hasher des objets Node.
        """
        DF = math.exp(-self.market.rate * self.delta_t)  # discount
        payoff = option.payoff  # alias
        probas = self._probas_for_node  # alias

        self._clear_values()  # reset cache

        def V(nd: Node):  # valeur au nœud
            # mémo local
            if nd.value is not None:  # déjà calculé
                return nd.value  # retourne
            # terminal: pas d'enfants
            if nd.down is None:  # feuille
                val = payoff(nd.under)  # payoff
                nd.value = val  # store
                return val  # retour
            pd_nd, pm_nd, pu_nd = probas(nd)  # probas
            up, mid, down = nd.up, nd.mid, nd.down  # enfants
            cont = DF * (pu_nd * V(up) + pm_nd * V(mid) + pd_nd * V(down))  # continuation
            nd.value = cont  # store
            return cont  # retour

        return V(self.root)  # prix

    def price_american_recursive(self, option):  # récursif US
        """
        Récursif mémoïsé directement sur les nœuds (pas de lru_cache), style américain.
        """
        DF = math.exp(-self.market.rate * self.delta_t)  # discount
        payoff = option.payoff  # alias
        probas = self._probas_for_node  # alias

        self._clear_values()  # reset

        def V(nd: Node):  # valeur au nœud
            if nd.value is not None:  # cache
                return nd.value  # retour
            if nd.down is None:  # feuille
                val = payoff(nd.under)  # payoff
                nd.value = val  # store
                return val  # retour
            pd_nd, pm_nd, pu_nd = probas(nd)  # probas
            up, mid, down = nd.up, nd.mid, nd.down  # enfants
            cont = DF * (pu_nd * V(up) + pm_nd * V(mid) + pd_nd * V(down))  # continuation
            exer = payoff(nd.under)  # exercice
            val = cont if cont >= exer else exer  # max
            nd.value = val  # store
            return val  # retour

        return V(self.root)  # prix
    
    def price_american_backward(self, option):  # backward US
        """
        Backward itératif avec exercice anticipé (PUT/option américaine).
        """
        N, dt = self.nb_steps, self.delta_t  # raccourcis
        DF = math.exp(-self.market.rate * dt)  # discount

        # terminal
        headN = self._head_last if self._head_last is not None else self._level_head(N)  # tête N
        for nd in headN.iter_right():  # parcours
            nd.value = option.payoff(nd.under)  # payoff

        # backward
        for i in range(N - 1, -1, -1):  # colonnes
            head_i = self._level_head(i)  # tête i
            for nd in head_i.iter_right():  # noeuds
                pd_nd, pm_nd, pu_nd = self._probas_for_node(nd)  # ordre correct des probas
                hold = DF * (  # valeur en hold
                    pu_nd * nd.up.value +
                    pm_nd * nd.mid.value +
                    pd_nd * nd.down.value
                )
                exercise = option.payoff(nd.under)  # exercice
                nd.value = max(exercise, hold)  # max
        return self.root.value  # prix

    def plot_pointer(self, annotate=True, figsize=(10, 6), dpi=110):  # tracé de l’arbre
        """
        Trace l'arbre en parcourant les pointeurs (root, right, down/mid/up).
        Compatible avec la version sans listes/dictionnaires.
        -----------------------------------------------------------------------
        - Chaque colonne i est parcourue via ._level_head(i) et .right
        - Les arêtes sont tracées vers .down, .mid et .up
        - Les nœuds sont affichés avec leurs prix sous-jacents
        -----------------------------------------------------------------------
        """

        N = self.nb_steps  # profondeur
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)  # figure

        segments = []  # arêtes
        xs, ys = [], []  # points

        # Parcours des colonnes 0..N
        for i in range(0, N + 1):  # colonnes
            # Tête de la colonne i
            head_i = self.root if i == 0 else self._level_head(i)  # tête
            nd = head_i  # curseur

            # Parcours horizontal
            while nd is not None:  # tant qu’il y a des nœuds
                xs.append(i)  # x
                ys.append(nd.under)  # y

                # Liaisons vers la colonne suivante
                if i < N:  # si pas dernière
                    if nd.down is not None:  # bas
                        segments.append([(i, nd.under), (i + 1, nd.down.under)])  # segment
                    if nd.mid is not None:  # milieu
                        segments.append([(i, nd.under), (i + 1, nd.mid.under)])  # segment
                    if nd.up is not None:  # haut
                        segments.append([(i, nd.under), (i + 1, nd.up.under)])  # segment

                nd = nd.right  # passer au noeud suivant

        # Arêtes
        lc = LineCollection(segments, colors="gray", linewidths=0.7, alpha=0.7)  # collection
        ax.add_collection(lc)  # ajoute

        # Nœuds
        ax.scatter(xs, ys, s=20, color="C0", zorder=3)  # points
        if annotate:  # labels
            for (x, y) in zip(xs, ys):  # boucle
                ax.annotate(f"{y:.2f}", (x, y),
                            textcoords="offset points", xytext=(0, 6),
                            ha="center", fontsize=8)  # annotation

        # Mise en forme
        ax.set_title(f"Arbre trinomial (N={N}) — version pointeurs")  # titre
        ax.set_xlabel("Étape (i)")  # label x
        ax.set_ylabel("Sous-jacent S(i,j)")  # label y
        ax.grid(True, linestyle=":", alpha=0.4)  # grille
        ax.set_xlim(-0.2, N + 0.2)  # xlim

        ymin, ymax = min(ys), max(ys)  # bornes y
        marg = 0.05 * (ymax - ymin) if ymax > ymin else 1.0  # marge
        ax.set_ylim(ymin - marg, ymax + marg)  # ylim

        plt.tight_layout()  # layout
        plt.show()  # affiche

    # ---------- greeks (à la racine, sans bump) ----------
    def local_greeks_no_bump(self, option):  # grecs à la racine
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
        import math  # local pour éviter import global

        dt = self.delta_t  # pas
        r = self.market.rate  # taux

        # Détermine le style à utiliser pour remplir .value
        style = getattr(option, "type", "european")  # style
        if style is None:  # défaut
            style = "european"  # EU
        style = str(style).lower().strip()  # normalise
        if style not in ("european", "american"):  # garde-fou
            raise ValueError("option.type doit être 'european' ou 'american'.")  # erreur

        # Remplit les .value du tree en exécutant le pricer backward adapté
        # (on force engine='backward' car nous lisons les valeurs des enfants immédiats)
        try:
            # price() mettra à jour .value sur tous les nœuds via l'algorithme backward
            self.price(option, engine="backward", style=style)  # calcule
        except Exception as e:  # capture
            raise RuntimeError(f"Impossible de calculer les valeurs du tree pour calculer les grecs : {e}") from e  # wrap

        # enfants de la racine
        up = self.root.up  # up
        mid = self.root.mid  # mid
        down = self.root.down  # down
        if up is None or mid is None or down is None:  # vérif
            raise RuntimeError("Construis d'abord l'arbre (build_bottom_first) avant de calculer les grecs.")  # erreur

        S_up,  S_mid,  S_down  = up.under,  mid.under,  down.under  # S
        V_up,  V_mid,  V_down  = up.value,  mid.value,  down.value  # V
        V0 = self.root.value  # valeur à la racine

        # --- Delta
        denom = (S_up - S_down)  # pas
        if abs(denom) < 1e-15:  # garde-fou
            raise ZeroDivisionError("Maille du 1er étage dégénérée.")  # erreur
        delta = (V_up - V_down) / denom  # pente

        # --- Gamma (maille non uniforme ok)
        h_up = (S_up - S_mid)  # pas haut
        h_dn = (S_mid - S_down)  # pas bas
        if abs(h_up) < 1e-15 or abs(h_dn) < 1e-15:  # vérif
            raise ZeroDivisionError("Maille du 1er étage trop fine.")  # erreur
        gamma = 2.0 * ((V_up - V_mid) / h_up - (V_mid - V_down) / h_dn) / (h_up + h_dn)  # courbure

        # --- Vega (proxy Black-Scholes, S = spot racine, T = nb_steps*dt)
        try:
            K = float(option.strike)  # strike
        except Exception as e:  # catch
            raise AttributeError("L'option doit avoir un attribut 'strike' pour calculer la vega (proxy BS).") from e  # erreur

        S0     = float(self.root.under)  # spot racine
        sigma  = float(self.market.vol)  # vol
        T_total = max(1e-12, float(self.nb_steps) * float(dt))  # maturité totale
        if sigma <= 0.0:  # vérif
            raise ValueError("market.vol doit être > 0 pour calculer la vega (proxy BS).")  # erreur

        # densité normale standard
        def _phi(x):  # pdf normal
            return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)  # densité

        # d1 et vega BS (identique call/put)
        d1 = (math.log(S0 / K) + (r + 0.5 * sigma * sigma) * T_total) / (sigma * math.sqrt(T_total))  # d1
        vega = S0 * math.sqrt(T_total) * _phi(d1) / 100  # vega (par 1% de vol)

        return {"price": V0, "delta": delta, "gamma": gamma, "vega": vega}