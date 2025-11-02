# main.py 
# TODO : mettre un graph de temps en echelle log, et en echelle normale. + Prunning + grecques s'il faut
from market import Market
from option import Option
from tree import Tree
from math import log, sqrt, exp
from scipy.stats import norm

import sys
sys.setrecursionlimit(20000)

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import time

# ===========================================================
# Black–Scholes de référence
# ===========================================================
def black_scholes_price(S0, K, T, r, sigma, call_put="call"):
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if call_put.lower() == "call":
        return S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# ===========================================================
# Convergence arbre (backward vs récursif vs BS)
# ===========================================================
def run_convergence(market, option, Ns):
    bs = black_scholes_price(
        market.underlying, option.K, option.t, market.rate, market.vol, option.call_put
    )
    print(f"\nPrix Black–Scholes (référence fermée) : {bs:.6f}\n")

    print("\nComparaison des temps Backward vs Récursif :\n")
    header = (
        "   N |  Backward  |  Récursif  |  ΔPrix(B−R) |  Gap(Tree−BS) |  t_B (s)  |  t_R (s)  |  Ratio t_R/t_B"
    )
    print(header)
    print("-" * len(header))

    for N in sorted(Ns):
        dt = option.t / N
        tree = Tree(market=market, nb_steps=N, delta_t=dt)
        tree.build_bottom_first(option)
        # Force le calcul des probas no-div (pas obligatoire pour l'algo)
        tree.trinomial_prob_no_div()

        # Backward (européen)
        t0 = time.perf_counter()
        price_back = tree.price(option, engine="backward", style="european")
        t1 = time.perf_counter()

        # Récursif (européen, mémo interne)
        t2 = time.perf_counter()
        price_rec = tree.price(option, engine="recursive", style="european")
        t3 = time.perf_counter()

        diff_price = price_back - price_rec
        gap_bs = price_back - bs
        t_back = t1 - t0
        t_rec = t3 - t2
        ratio = t_rec / t_back if t_back > 0 else float("inf")

        print(
            f"{N:4d} | {price_back:10.6f} | {price_rec:10.6f} | "
            f"{diff_price:+11.3e} | {gap_bs:+11.3e} | "
            f"{t_back:8.4f} | {t_rec:8.4f} | {ratio:8.2f}"
        )

    print(f"\n\n Prix Black–Scholes: {bs:.6f}\n\n")

# ===========================================================
# GAP(K) = Prix_arbre(K) – Prix_BS(K) pour une grille de K
# ===========================================================
def price_trinomial_for_strike(option, market, T, N, K, call_put="call"):
    opt = Option(t=T, call_put=call_put, K=K, type="european")
    dt = T / N
    tree = Tree(market=market, nb_steps=N, delta_t=dt)
    tree.build_bottom_first(option)
    tree.trinomial_prob_no_div()
    return tree.price_european(opt)

def compute_gap_vs_strike(market, option, N, K_min, K_max, nK=61):
    strikes = np.linspace(K_min, K_max, nK)
    gap = np.empty_like(strikes)

    for i, K in enumerate(strikes):
        bs = black_scholes_price(
            market.underlying, K, option.t, market.rate, market.vol, option.call_put
        )
        tree_price = price_trinomial_for_strike(
            option, market, option.t, N, K, call_put=option.call_put
        )
        gap[i] = tree_price - bs

    return strikes, gap

def plot_gap_vs_strike(market, option, N, K_min, K_max, nK=61):
    strikes, gap = compute_gap_vs_strike(market, option, N, K_min, K_max, nK)
    plt.figure()
    plt.plot(strikes, gap, label="Gap = Prix arbre − Prix BS")
    plt.axhline(0.0, linestyle="--", linewidth=1)
    plt.xlabel("Strike K")
    plt.ylabel("Gap (arbre − BS)")
    plt.title(f"Gap en fonction du strike (option {option.call_put}, N={N})")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ===========================================================
# Comparaison Euro vs Américain (CALL & PUT)
# ===========================================================
def price_european_tree(option, market: Market, T: float, N: int, K: float, call_put: str) -> float:
    opt = Option(t=T, call_put=call_put, K=K, type="european")
    dt = T / N
    tree = Tree(market=market, nb_steps=N, delta_t=dt)
    tree.build_bottom_first(option)
    tree.trinomial_prob_no_div()
    return tree.price_european(opt)

def price_american_tree(option, market: Market, T: float, N: int, K: float, call_put: str) -> float:
    # NB : on réutilise "option" pour le payoff et l'exercice (comme dans ton code)
    dt = T / N
    tree = Tree(market=market, nb_steps=N, delta_t=dt)
    tree.build_bottom_first(option)
    tree.trinomial_prob_no_div()
    return tree.price_american(option)

def compare_euro_amer_call_put(option, market: Market, K: float, T: float, Ns):
    S0, r, sigma = market.underlying, market.rate, market.vol

    for call_put in ("call", "put"):
        print("\n" + "=" * 78)
        print(f" COMPARAISON {call_put.upper()} — EUROPÉEN vs AMÉRICAIN ".center(78, "="))
        print("=" * 78)
        bs = black_scholes_price(S0, K, T, r, sigma, call_put)
        print(f"Black–Scholes (européen, ref) : {bs:.6f}\n")

        header = "   N |   Euro(Tree)   Δ(Euro−BS) |   Amér(Tree)   Δ(Amér−Euro)"
        print(header)
        print("-" * len(header))

        last_euro = last_amer = None
        for N in sorted(Ns):
            euro = price_european_tree(option, market, T, N, K, call_put)
            amer = price_american_tree(option, market, T, N, K, call_put)
            print(
                f"{N:4d} | {euro:12.6f}  {euro - bs:+11.4e} | "
                f"{amer:12.6f}  {amer - euro:+11.4e}"
            )
            last_euro, last_amer = euro, amer

        if call_put == "call":
            print("\nNote: Sans dividendes, CALL américain ≈ CALL européen (pas d'intérêt à exercer tôt).")
            if last_amer is not None and last_euro is not None and abs(last_amer - last_euro) > 1e-2:
                print("⚠️  Alerte: l'écart call amér/euro reste élevé — vérifier price_american().")
        else:
            print("\nNote: PUT américain doit être ≥ PUT européen (exercice anticipé possible).")
            if last_amer is not None and last_euro is not None and last_amer + 1e-10 < last_euro:
                print("⚠️  Alerte: put américain < européen — vérifier price_american().")

def main():

    # Marché et option "de base"
    dividend_date = datetime(2026, 6, 9) 
    market = Market(underlying=102.45, rate=0.04, vol=0.28)
    end_date = datetime(2026, 8, 27)
    start_date = datetime(2025, 10, 29)
    t = (end_date - start_date).days / 365.0
    option = Option(t, call_put="call", K=90, type="european", div=3, div_date=dividend_date)

    # --- Prix avec un N "grand" pour référence ---
    N_large = 2000
    tree_highres = Tree(market=market, nb_steps=N_large, delta_t=option.t / N_large)

    # 1) Construction
    
    t0 = time.time()
    tree_highres.build_bottom_first(option)
    t1 = time.time()
    print(f"⏱ Temps de construction de l’arbre : {t1 - t0:.4f} secondes")

    
    # 2) Pricing — EUROPÉEN
    t2 = time.time()
    price_euro_back = tree_highres.price(option, engine="backward", style="european")
    t3 = time.time()
    t_back_eu = t3 - t2
    print(f"⏱ Temps de pricing backward (EU)  : {t_back_eu:.4f} secondes")

    t4 = time.time()
    price_euro_rec = tree_highres.price(option, engine="recursive", style="european")
    t5 = time.time()
    t_rec_eu = t5 - t4
    print(f"⏱ Temps de pricing récursif (EU) : {t_rec_eu:.4f} secondes")

    # 3) Pricing — AMÉRICAIN
    t6 = time.time()
    price_amer_back = tree_highres.price(option, engine="backward", style="american")
    t7 = time.time()
    t_back_us = t7 - t6
    print(f"⏱ Temps de pricing backward (US) : {t_back_us:.4f} secondes")

    t8 = time.time()
    price_amer_rec = tree_highres.price(option, engine="recursive", style="american")
    t9 = time.time()
    t_rec_us = t9 - t8
    print(f"⏱ Temps de pricing récursif (US) : {t_rec_us:.4f} secondes")

    # 4) Récap / affichage
    print("=" * 70)
    print(f"\nPrix EU avec N = {N_large} pas : {price_euro_back:.6f} (backward)  |  {price_euro_rec:.6f} (récursif)")
    print(f"Prix US avec N = {N_large} pas : {price_amer_back:.6f} (backward)  |  {price_amer_rec:.6f} (récursif)\n")
    print(f"Prix Black–Scholes (référence fermée, EU sans div) : "
          f"{black_scholes_price(market.underlying, option.K, option.t, market.rate, market.vol, option.call_put):.6f}\n")
    print("=" * 70)
    

    # Exemple convergence
    """
    Ns = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    run_convergence(market, option, Ns)
    """
    # Exemple GAP vs K
    """
    S0 = market.underlying
    K_min, K_max = 0.6 * S0, 1.4 * S0
    N_gap = 200
    nK_points = 61
    plot_gap_vs_strike(market, option, N=N_gap, K_min=K_min, K_max=K_max, nK=nK_points)
    """

    # Greeks locaux (optionnel)
    
    greeks = tree_highres.local_greeks_no_bump(option)
    print("\n=== Greeks (no-bump, à la racine) ===")
    print(f"Price = {greeks['price']:.6f}")
    print(f"Delta = {greeks['delta']:.6f}")
    print(f"Gamma = {greeks['gamma']:.6e}")
    print(f"Vega = {greeks['vega']:.6f}")
    

    #Exemple de plot avec N = 5
    """
    N_plot = 5
    tree_plot = Tree(market=market, nb_steps=N_plot, delta_t=option.t / N_plot)
    tree_to_plot = tree_plot.build_bottom_first(option)
    tree_to_plot.plot_pointer(option)
    """

if __name__ == "__main__":
    main()
    """
    # === Test candidats pour vega = 0.30 (put américain) ===
    candidates = {
        'a': (85.27, 119.54), 'b': (85.70, 120.80), 'c': (86.32, 120.68), 'd': (89.08, 118.75),
        'e': (88.31, 120.14), 'f': (86.39, 124.51), 'g': (88.45, 122.60), 'h': (86.71, 124.69),
        'i': (88.55, 123.05), 'j': (89.64, 122.66), 'k': (88.59, 123.81), 'l': (86.24, 126.16),
        'm': (89.56, 123.71), 'n': (89.40, 124.75), 'o': (92.60, 121.90), 'p': (88.95, 125.70),
        'q': (88.66, 127.13), 'r': (88.29, 128.28), 's': (91.14, 126.20), 't': (93.67, 127.55)
    }

    # re-create market and base start_date consistent with main
    market = Market(underlying=102.45, rate=0.04, vol=0.28)
    start_date = datetime(2025, 10, 29)
    end_date = datetime(2026, 8, 27)
    t = (end_date - start_date).days / 365.0

    print('\nTest des vegas pour put américain (cible vega = 0.30) :')
    print('Lettre |     K1     |   vega1   |    K2     |   vega2   |  Match?')
    print('-' * 70)

    N_test = 400
    for letter, (K1, K2) in candidates.items():
        vegas = []
        for K in (K1, K2):
            opt_test = Option(t=t, call_put='put', K=K, type='american', div=3, div_date=datetime(2026,7,9))
            dt = t / N_test if N_test > 0 else t
            tree_test = Tree(market=market, nb_steps=N_test, delta_t=dt)
            tree_test.build_bottom_first(opt_test)
            try:
                g = tree_test.local_greeks_no_bump(opt_test)
                vegas.append(g.get('vega', float('nan')))
            except Exception as e:
                print(f"Erreur pour K={K}: {e}")
                vegas.append(float('nan'))

        match = all([abs(v - 0.30) < 1e-3 for v in vegas])
        print(f"  {letter}   | {K1:8.2f} | {vegas[0]:8.4f} | {K2:8.2f} | {vegas[1]:8.4f} | {match}")
    """
