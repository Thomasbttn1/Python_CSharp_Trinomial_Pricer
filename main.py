# main.py
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
    bs = black_scholes_price(market.underlying, option.K, option.t, market.rate, market.vol, option.call_put)
    print(f"\nPrix Black–Scholes (référence fermée) : {bs:.6f}\n")

    print("\nComparaison des temps Backward vs Récursif :\n")
    header = "   N |  Backward  |  Récursif  |  ΔPrix(B−R) |  Gap(Tree−BS) |  t_B (s)  |  t_R (s)  |  Ratio t_R/t_B"
    print(header)
    print("-" * len(header))

    for N in sorted(Ns):
        dt = option.t / N
        tree = Tree(market=market, nb_steps=N, delta_t=dt)
        tree.build_bottom_first(option)
        # Force le calcul des probas no-div (pas obligatoire pour l'algo)
        tree.p_down, tree.p_mid, tree.p_up = tree.trinomial_prob_no_div()

        # Backward
        t0 = time.perf_counter()
        price_back = tree.price_european(option)
        t1 = time.perf_counter()

        # Récursif (mémo interne via lru_cache, sans containers)
        t2 = time.perf_counter()
        price_rec  = tree.price_european_recursive(option)
        t3 = time.perf_counter()

        diff_price = price_back - price_rec
        gap_bs = price_back - bs
        t_back = t1 - t0
        t_rec  = t3 - t2
        ratio  = t_rec / t_back if t_back > 0 else float("inf")

        print(f"{N:4d} | {price_back:10.6f} | {price_rec:10.6f} | "
              f"{diff_price:+11.3e} | {gap_bs:+11.3e} | "
              f"{t_back:8.4f} | {t_rec:8.4f} | {ratio:8.2f}")

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
    dividend_date = datetime(2026, 3, 1)
    market = Market(underlying=100.0, rate=0.03, vol=0.20)
    option = Option(t=1.0, call_put="put", K=100.0, type="european", div=2, div_date=dividend_date)

    # --- Prix avec un N "grand" pour référence ---
    N_large = 1000
    tree_highres = Tree(market=market, nb_steps=N_large, delta_t=option.t / N_large)

    # 1) Construction
    t0 = time.time()
    tree_highres.build_bottom_first(option)
    t1 = time.time()
    print(f"⏱ Temps de construction de l’arbre : {t1 - t0:.4f} secondes")

    # 2) Pricing backward
    t2 = time.time()
    price_highres = tree_highres.price_european(option)
    t3 = time.time()
    print(f"⏱ Temps de pricing backward : {t3 - t2:.4f} secondes")

    # 3) Pricing récursif (memo lru_cache)
    t4 = time.time()
    price_highres_rec = tree_highres.price_european_recursive(option)
    t5 = time.time()
    print(f"⏱ Temps de pricing récursif : {t5 - t4:.4f} secondes")

    # 4) American
    t6 = time.time()
    price_highres_american = tree_highres.price_american(option)
    t7 = time.time()
    print(f"⏱ Temps de pricing américain : {t7 - t6:.4f} secondes")

    print("=" * 70)
    print(f" \nPrix de l'option avec N = {N_large} pas : {price_highres:.6f}")
    print(f" (via backward) et {price_highres_rec:.6f} (via récursif)\n")
    print(f"Prix American avec N = {N_large} pas : {price_highres_american:.6f}\n")
    print(f"\nPrix Black–Scholes (référence fermée) sans div: "
          f"{black_scholes_price(market.underlying, option.K, option.t, market.rate, market.vol, option.call_put):.6f}\n")
    print("=" * 70)

    # Exemple convergence (décommenter au besoin)
    """
    Ns = [2, 5, 10, 20, 50, 100, 200, 500, 1000]
    run_convergence(market, option, Ns)
    """

    # Exemple GAP vs K (décommenter au besoin)
    """
    S0 = market.underlying
    K_min, K_max = 0.6 * S0, 1.4 * S0
    N_gap = 200
    nK_points = 61
    plot_gap_vs_strike(market, option, N=N_gap, K_min=K_min, K_max=K_max, nK=nK_points)
    """

    # Greeks locaux (optionnel)
    """
    greeks = tree_highres.local_greeks_no_bump(option)
    print("\n=== Greeks (no-bump, à la racine) ===")
    print(f"Price = {greeks['price']:.6f}")
    print(f"Delta = {greeks['delta']:.6f}")
    print(f"Gamma = {greeks['gamma']:.6e}")
    print(f"Theta = {greeks['theta']:.6f}")
    """

if __name__ == "__main__":
    main()
