import os
import sys
import time
from datetime import date, datetime  # dates UI

# Assure les imports locaux même si Streamlit est lancé d'ailleurs
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))  # dossier courant
if CURRENT_DIR not in sys.path:  # évite doublon
    sys.path.insert(0, CURRENT_DIR)  # ajoute au path

import numpy as np  # calculs
import matplotlib.pyplot as plt  # tracés
from matplotlib.collections import LineCollection  # segments pour tracer l'arbre
import streamlit as st  # UI
from math import log, sqrt, exp  # BS
from scipy.stats import norm  # loi normale

from market import Market  # marché
from option import Option  # option
from tree import Tree  # arbre

# ===========================
# Utils
# ===========================
def year_fraction(d0: date, d1: date) -> float:  # fraction d'année
    return (d1 - d0).days / 365.0  # base 365

def black_scholes_price(S0, K, T, r, sigma, call_put="call"):  # prix BS EU
    if T <= 0:  # maturité écoulée
        return max(0.0, (S0 - K) if call_put == "call" else (K - S0))  # payoff direct
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))  # d1
    d2 = d1 - sigma * sqrt(T)  # d2
    if call_put.lower() == "call":  # call
        return S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)  # formule call
    else:  # put
        return K * exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)  # formule put

def build_tree_and_price(market: Market, option: Option, N: int,  # build + prix
                         do_euro: bool, do_amer: bool,
                         engine_eu: str = "backward", engine_us: str = "backward"):
    dt = option.t / max(N, 1)  # pas de temps
    tree = Tree(market=market, nb_steps=N, delta_t=dt)  # instancie arbre

    t0 = time.perf_counter()  # start chrono
    tree.build_bottom_first(option)  # construction
    tree.trinomial_prob_no_div()  # probas globales
    t_build = time.perf_counter() - t0  # durée build

    res = {"t_build": t_build}  # sortie

    if do_euro:  # prix EU
        t1 = time.perf_counter()  # chrono
        price_eu = tree.price(option, engine=engine_eu, style="european")  # pricing EU
        res["price_eu"] = price_eu  # store
        res["t_eu"] = time.perf_counter() - t1  # temps EU

    if do_amer:  # prix US
        t2 = time.perf_counter()  # chrono
        price_us = tree.price(option, engine=engine_us, style="american")  # pricing US
        res["price_us"] = price_us  # store
        res["t_us"] = time.perf_counter() - t2  # temps US

    # Greeks à la racine (no-bump)
    try:
        g = tree.local_greeks_no_bump(option)  # grecs locaux
        res["greeks"] = g  # stocke
    except Exception as e:  # robustesse
        res["greeks"] = {"error": str(e)}  # message

    res["tree"] = tree  # renvoie aussi l’arbre
    return res  # dict résultat

def compute_convergence(market: Market, option: Option, Ns, engine="backward", style="european"):  # courbe conv
    prices, t_builds, t_prices = [], [], []  # buffers
    for N in Ns:  # boucle N
        dt = option.t / max(N, 1)  # pas
        tree = Tree(market=market, nb_steps=N, delta_t=dt)  # arbre
        t0 = time.perf_counter()  # chrono
        tree.build_bottom_first(option)  # build
        tree.trinomial_prob_no_div()  # probas
        t_build = time.perf_counter() - t0  # temps build

        t1 = time.perf_counter()  # chrono
        p = tree.price(option, engine=engine, style=style)  # prix
        t_price = time.perf_counter() - t1  # temps prix

        prices.append(p)  # push
        t_builds.append(t_build)  # push
        t_prices.append(t_price)  # push
    return np.array(Ns), np.array(prices), np.array(t_builds), np.array(t_prices)  # arrays

def compute_gap_vs_strike(market: Market, option: Option, N, K_min, K_max, nK=61):  # gap vs K
    strikes = np.linspace(K_min, K_max, nK)  # grille K
    gaps = []  # liste écart
    for K in strikes:  # loop K
        bs = black_scholes_price(market.underlying, K, option.t, market.rate, market.vol, option.call_put)  # BS
        optK = Option(t=option.t, call_put=option.call_put, K=float(K), type="european", div=option.div, div_date=option.div_date)  # opt K
        dt = optK.t / max(N, 1)  # pas
        tree = Tree(market=market, nb_steps=N, delta_t=dt)  # arbre
        tree.build_bottom_first(optK)  # build
        tree.trinomial_prob_no_div()  # probas
        tree_price = tree.price(optK, engine="backward", style="european")  # prix EU
        gaps.append(tree_price - bs)  # gap
    return strikes, np.array(gaps)  # sorties

def _tree_figure(tree: Tree, annotate: bool = False):  # figure matplotlib de l’arbre
    N = tree.nb_steps  # profondeur
    fig, ax = plt.subplots(figsize=(8, 5), dpi=110)  # figure

    segments = []  # arêtes
    xs, ys = [], []  # points

    for i in range(0, N + 1):  # colonnes
        head_i = tree.root if i == 0 else tree._level_head(i)  # tête colonne
        nd = head_i  # curseur
        while nd is not None:  # parcours horizontal
            xs.append(i)  # x
            ys.append(nd.under)  # y
            if i < N:  # arêtes vers i+1
                if nd.down is not None:  # bas
                    segments.append([(i, nd.under), (i + 1, nd.down.under)])  # segment
                if nd.mid is not None:  # milieu
                    segments.append([(i, nd.under), (i + 1, nd.mid.under)])  # segment
                if nd.up is not None:  # haut
                    segments.append([(i, nd.under), (i + 1, nd.up.under)])  # segment
            nd = nd.right  # suivant

    lc = LineCollection(segments, colors="gray", linewidths=0.7, alpha=0.7)  # collection d’arêtes
    ax.add_collection(lc)  # ajoute
    ax.scatter(xs, ys, s=18, color="C0", zorder=3)  # nœuds

    if annotate and N <= 25 and len(xs) <= 1500:  # labels si petit arbre
        for (x, y) in zip(xs, ys):  # boucle
            ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)  # valeur

    ax.set_title(f"Arbre trinomial (N={N})")  # titre
    ax.set_xlabel("Étape (i)")  # axe X
    ax.set_ylabel("Sous-jacent S(i,j)")  # axe Y
    ax.grid(True, linestyle=":", alpha=0.4)  # grille
    ax.set_xlim(-0.2, N + 0.2)  # bornes X

    if ys:  # bornes Y
        ymin, ymax = min(ys), max(ys)  # min/max
        marg = 0.05 * (ymax - ymin) if ymax > ymin else 1.0  # marge
        ax.set_ylim(ymin - marg, ymax + marg)  # limites

    fig.tight_layout()  # layout
    return fig  # figure

# ===========================
# UI
# ===========================
st.set_page_config(page_title="Trinomial Pricer", page_icon="📈", layout="wide")  # config
st.title("📈 Trinomial Pricer — Streamlit")  # titre

with st.sidebar:  # panneau latéral
    st.header("Paramètres marché")  # section
    S0 = st.number_input("S0 (spot)", value=102.45, step=0.1)  # spot
    r = st.number_input("Taux r", value=0.04, step=0.001, format="%.4f")  # taux
    sigma = st.number_input("Vol σ", value=0.28, step=0.001, format="%.4f")  # vol

    st.subheader("Dividende (optionnel)")  # dividende
    use_div = st.checkbox("Inclure un dividende discret", value=True)  # toggle
    div_amt = st.number_input("Montant dividende", value=3.0, step=0.1) if use_div else 0.0  # montant
    div_date = st.date_input("Date dividende", value=date(2026, 6, 9)) if use_div else None  # date

    st.header("Paramètres option")  # option
    call_put = st.selectbox("Type (Call/Put)", ["call", "put"], index=0)  # type
    style = st.selectbox("Style (EU/US)", ["european", "american"], index=0)  # style
    K = st.number_input("Strike K", value=90.0, step=0.5)  # strike

    st.subheader("Maturité")  # maturité
    start_d = st.date_input("Date de départ", value=date(2025, 10, 29))  # start
    end_d = st.date_input("Date d’échéance", value=date(2026, 8, 27))  # fin
    T = year_fraction(start_d, end_d)  # T
    st.caption(f"T ≈ {T:.6f} an(s)")  # info

    st.header("Arbre")  # arbre
    N = st.number_input("Pas N", min_value=1, max_value=3000, value=400, step=50)  # pas
    engine_eu = st.selectbox("Moteur (EU)", ["backward", "recursive"], index=0)  # moteur EU
    engine_us = st.selectbox("Moteur (US)", ["backward", "recursive"], index=0)  # moteur US

    st.header("Analyses")  # analyses
    show_convergence = st.checkbox("Convergence vs N", value=False)  # conv
    show_gap = st.checkbox("Gap (Tree − BS) vs Strike", value=False)  # gap
    show_tree = st.checkbox("Afficher l’arbre", value=False)  # arbre plot
    annotate_tree = st.checkbox("Annoter les nœuds (petits N)", value=False)  # labels

    if show_convergence:  # params conv
        Nmin = st.number_input("N min", min_value=1, value=20, step=1)  # borne min
        Nmax = st.number_input("N max", min_value=Nmin+1, value=600, step=10)  # borne max
        points = st.slider("Nb points", min_value=3, max_value=12, value=6)  # nb points
        conv_engine = st.selectbox("Moteur convergence", ["backward", "recursive"], index=0)  # eng
        conv_style = st.selectbox("Style convergence", ["european", "american"], index=0)  # sty

    if show_gap:  # params gap
        mult_min, mult_max = st.slider("Range K / S0", 0.4, 2.0, (0.6, 1.4), step=0.05)  # range K
        nK = st.slider("Points K", 11, 101, 41, step=2)  # densité
        N_gap = st.number_input("N (Gap)", min_value=5, max_value=2000, value=200, step=25)  # N gap

    run = st.button("Lancer le pricing", type="primary")  # bouton

# Affichage principal
col1, col2 = st.columns([2, 1])  # layout 2 colonnes

with col1:  # résultats
    st.subheader("Résultats")  # titre

    market = Market(underlying=S0, rate=r, vol=sigma)  # marché
    opt = Option(t=T, call_put=call_put, K=K, type=style, div=div_amt, div_date=datetime.combine(div_date, datetime.min.time()) if div_date else None)  # option

    if run:  # exécution
        with st.spinner("Construction de l’arbre et pricing..."):  # spinner
            res = build_tree_and_price(
                market=market, option=opt, N=int(N),  # inputs
                do_euro=True, do_amer=True,  # deux styles
                engine_eu=engine_eu, engine_us=engine_us  # moteurs
            )

        # Black–Scholes de référence (EU sans dividende)
        bs = black_scholes_price(S0, K, T, r, sigma, call_put)  # BS
        st.write(f"Black–Scholes (réf EU, sans dividende) = {bs:.6f}")  # affichage
        if use_div and div_amt not in (0, None):  # note div
            st.caption("Attention: BS ci-dessus n’intègre pas le dividende discret.")  # note

        # Prix
        cols = st.columns(2)  # deux colonnes
        with cols[0]:  # bloc EU
            if "price_eu" in res:  # guard
                st.metric("Prix Européen (engine EU)", f"{res['price_eu']:.6f}", delta=f"{res['price_eu']-bs:+.6f}")  # métrique
                st.caption(f"Temps EU ({engine_eu}) : {res['t_eu']:.4f}s")  # temps
        with cols[1]:  # bloc US
            if "price_us" in res:  # guard
                st.metric("Prix Américain (engine US)", f"{res['price_us']:.6f}")  # métrique
                st.caption(f"Temps US ({engine_us}) : {res['t_us']:.4f}s")  # temps

        st.caption(f"Temps construction de l’arbre : {res['t_build']:.4f}s")  # build time

        # Greeks
        g = res.get("greeks", {})  # grecs
        if "error" in g:  # erreur
            st.warning(f"Greeks indisponibles: {g['error']}")  # message
        else:  # affiche
            gcol1, gcol2, gcol3, gcol4 = st.columns(4)  # layout
            g_price = g.get("price", np.nan)  # price
            g_delta = g.get("delta", np.nan)  # delta
            g_gamma = g.get("gamma", np.nan)  # gamma
            g_vega = g.get("vega", np.nan)  # vega
            gcol1.metric("Price", f"{g_price:.6f}")  # metrique
            gcol2.metric("Delta", f"{g_delta:.6f}")  # metrique
            gcol3.metric("Gamma", f"{g_gamma:.6e}")  # metrique
            gcol4.metric("Vega", f"{g_vega:.6f}")  # metrique

        if show_tree:  # affichage arbre
            st.divider()  # séparateur
            st.subheader("Arbre trinomial")  # titre
            if int(N) > 300:  # garde-fou
                st.warning("N élevé: le tracé peut être peu lisible.")  # note
            fig_tree = _tree_figure(res["tree"], annotate=annotate_tree)  # figure
            st.pyplot(fig_tree, clear_figure=True)  # rendu

    else:  # pas lancé
        st.info("Renseigner les paramètres dans la barre latérale puis cliquer sur “Lancer le pricing”.")  # info

    # Convergence
    if run and show_convergence:  # section conv
        st.divider()  # séparateur
        st.subheader("Convergence vs N")  # titre
        Ns = np.linspace(Nmin, Nmax, num=points, dtype=int)  # grille
        with st.spinner("Calcul de la convergence..."):  # spinner
            X, P, TB, TP = compute_convergence(market, opt, Ns=Ns, engine=conv_engine, style=conv_style)  # calc

        fig, ax = plt.subplots(figsize=(6.5, 3.2))  # figure
        ax.plot(X, P, marker="o", label=f"Prix ({conv_style}, {conv_engine})")  # courbe
        if conv_style == "european":  # ref BS
            bs_ref = black_scholes_price(S0, K, T, r, sigma, call_put)  # BS
            ax.axhline(bs_ref, color="gray", linestyle="--", linewidth=1, label="BS (réf EU)")  # ligne
        ax.set_xlabel("N")  # x
        ax.set_ylabel("Prix")  # y
        ax.grid(True, alpha=0.25)  # grille
        ax.legend()  # légende
        st.pyplot(fig, clear_figure=True)  # rendu

        fig2, ax2 = plt.subplots(figsize=(6.5, 3.2))  # figure temps
        ax2.plot(X, TB, marker="s", label="Build time (s)")  # build
        ax2.plot(X, TP, marker="^", label="Price time (s)")  # price
        ax2.set_xlabel("N")  # x
        ax2.set_ylabel("Temps (s)")  # y
        ax2.set_yscale("log")  # échelle log
        ax2.grid(True, which="both", alpha=0.25)  # grille
        ax2.legend()  # légende
        st.pyplot(fig2, clear_figure=True)  # rendu

    # Gap vs Strike
    if run and show_gap:  # section gap
        st.divider()  # séparateur
        st.subheader("Gap (Tree − BS) vs Strike — EU")  # titre
        Kmin, Kmax = S0 * float(mult_min), S0 * float(mult_max)  # bornes
        with st.spinner("Calcul du gap vs strike..."):  # spinner
            strikes, gaps = compute_gap_vs_strike(market, opt, N=int(N_gap), K_min=Kmin, K_max=Kmax, nK=int(nK))  # calc
        fig3, ax3 = plt.subplots(figsize=(6.5, 3.2))  # figure
        ax3.plot(strikes, gaps, label="Gap")  # courbe
        ax3.axhline(0.0, color="gray", linestyle="--", linewidth=1)  # zéro
        ax3.set_xlabel("Strike K")  # x
        ax3.set_ylabel("Gap (Tree − BS)")  # y
        ax3.grid(True, alpha=0.25)  # grille
        ax3.legend()  # légende
        st.pyplot(fig3, clear_figure=True)  # rendu

with col2:  # configuration
    st.subheader("Configuration")  # titre
    st.json({  # json résumé
        "market": {"S0": S0, "r": r, "sigma": sigma},  # marché
        "option": {"style": style, "call_put": call_put, "K": K, "T": T, "div": div_amt, "div_date": str(div_date) if div_date else None},  # opt
        "tree": {"N": int(N), "engine_eu": engine_eu, "engine_us": engine_us},  # arbre
        "analyses": {"convergence": show_convergence, "gap_vs_strike": show_gap, "tree": show_tree, "annotate_tree": annotate_tree}  # flags
    })  # fin json

st.caption("Note: BS affiché est la référence européenne sans dividende discret. Les prix arbre intègrent votre dividende.")  # note