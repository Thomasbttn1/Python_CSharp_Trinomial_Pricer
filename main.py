# Script console de démonstration du pricer trinomial
# - Ajoute le sous-dossier "Python" au chemin d'import
# - Prix EU/US via backward et récursif
# - Référence Black–Scholes
# - Greeks locales (no-bump)
# - Outils de convergence et gap vs strike (optionnels)

import os
import csv
import sys
import time
from math import log, sqrt, exp
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Rendez les imports locaux robustes quand on lance depuis la racine du projet
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))            # répertoire du script
PY_DIR = os.path.join(CURRENT_DIR, "Python")                         # sous-dossier contenant market/option/tree
if PY_DIR not in sys.path:                                           # évite les doublons
    sys.path.insert(0, PY_DIR)                                       # priorité d'import

# Imports projet (après l’ajout au sys.path)
from market import Market                                            # marché (S0, r, sigma)
from option import Option                                            # description de l’option
from tree import Tree                                                # moteur d’arbre trinomial

# Sécurité pour l’engine récursif
sys.setrecursionlimit(20000)                                         # profondeur suffisante

# ===========================================================
# Black–Scholes (référence EU, sans dividende discret)
# ===========================================================
def black_scholes_price(S0, K, T, r, sigma, call_put="call"):
    # Gestion de la maturité écoulée (payoff direct)
    if T <= 0:
        return max(0.0, (S0 - K) if call_put.lower() == "call" else (K - S0))
    # Paramètres d1/d2
    d1 = (log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    # Formules call/put
    if call_put.lower() == "call":
        return S0 * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    return K * exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)

# ===========================================================
# Convergence (backward vs récursif) + écart à BS
# ===========================================================
def run_convergence(market, option, Ns):
    # Prix BS de référence (EU, sans dividende discret)
    bs = black_scholes_price(market.underlying, option.K, option.t, market.rate, market.vol, option.call_put)
    print(f"\nPrix Black–Scholes (référence fermée) : {bs:.6f}\n")

    # En-têtes lisibles pour comparer prix et temps
    header = "   N |  Backward  |  Récursif  |  ΔPrix(B−R) |  Gap(Tree−BS) |  t_B (s)  |  t_R (s)  |  Ratio t_R/t_B"
    print(header)
    print("-" * len(header))

    # Boucle sur une grille de pas N
    for N in sorted(Ns):
        dt = option.t / max(1, N)                                    # pas de temps
        tree = Tree(market=market, nb_steps=N, delta_t=dt)           # arbre pour ce N
        tree.build_bottom_first(option)                              # construction des noeuds
        tree.trinomial_prob_no_div()                                 # proba (cas sans div discret)

        # Backward (européen)
        t0 = time.perf_counter()
        price_back = tree.price(option, engine="backward", style="european")
        t1 = time.perf_counter()

        # Récursif (européen)
        t2 = time.perf_counter()
        price_rec = tree.price(option, engine="recursive", style="european")
        t3 = time.perf_counter()

        # Indicateurs de qualité et de performance
        diff_price = price_back - price_rec
        gap_bs = price_back - bs
        t_back = t1 - t0
        t_rec = t3 - t2
        ratio = t_rec / t_back if t_back > 0 else float("inf")

        # Sortie compacte
        print(
            f"{N:4d} | {price_back:10.6f} | {price_rec:10.6f} | "
            f"{diff_price:+11.3e} | {gap_bs:+11.3e} | "
            f"{t_back:8.4f} | {t_rec:8.4f} | {ratio:8.2f}"
        )

    print(f"\nPrix Black–Scholes: {bs:.6f}\n")

# ===========================================================
# GAP(K) = Prix_arbre(K) – Prix_BS(K) sur une grille de strikes
# ===========================================================
def price_trinomial_for_strike(market, T, N, K, call_put="call"):
    # Construit une option EU pour ce strike puis price sur arbre
    opt_K = Option(t=T, call_put=call_put, K=K, type="european")
    dt = T / max(1, N)
    tree = Tree(market=market, nb_steps=N, delta_t=dt)
    tree.build_bottom_first(opt_K)
    tree.trinomial_prob_no_div()
    return tree.price_european(opt_K)

def compute_gap_vs_strike(market, option, N, K_min, K_max, nK=61):
    # Échantillonne des strikes et calcule l’écart Tree − BS
    strikes = np.linspace(K_min, K_max, nK)
    gap = np.empty_like(strikes)
    for i, K in enumerate(strikes):
        bs = black_scholes_price(market.underlying, K, option.t, market.rate, market.vol, option.call_put)
        tree_price = price_trinomial_for_strike(market, option.t, N, float(K), call_put=option.call_put)
        gap[i] = tree_price - bs
    return strikes, gap

def plot_gap_vs_strike(market, option, N, K_min, K_max, nK=61):
    # Trace Gap(Tree − BS) en fonction du strike
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
# Wrappers EU / US pour comparer styles et moteurs
# ===========================================================
def price_european_tree(market, T, N, K, call_put):
    # Crée une option EU puis price via méthodes dédiées
    opt = Option(t=T, call_put=call_put, K=K, type="european")
    dt = T / max(1, N)
    tree = Tree(market=market, nb_steps=N, delta_t=dt)
    tree.build_bottom_first(opt)
    tree.trinomial_prob_no_div()
    return tree.price_european(opt)

def price_american_tree(market, T, N, K, call_put):
    # Crée une option US puis price avec exercice anticipé
    opt = Option(t=T, call_put=call_put, K=K, type="american")
    dt = T / max(1, N)
    tree = Tree(market=market, nb_steps=N, delta_t=dt)
    tree.build_bottom_first(opt)
    tree.trinomial_prob_no_div()
    return tree.price_american(opt)

def compare_euro_amer_call_put(market, K, T, Ns):
    # Compare EU vs US pour call et put, avec référence BS sur EU
    for cp in ("call", "put"):
        print("\n" + "=" * 78)
        print(f" COMPARAISON {cp.upper()} — EUROPÉEN vs AMÉRICAIN ".center(78, "="))
        print("=" * 78)
        bs = black_scholes_price(market.underlying, K, T, market.rate, market.vol, cp)
        print(f"Black–Scholes (européen, ref) : {bs:.6f}\n")

        header = "   N |   Euro(Tree)   Δ(Euro−BS) |   Amér(Tree)   Δ(Amér−Euro)"
        print(header)
        print("-" * len(header))

        last_eu = last_us = None
        for N in sorted(Ns):
            eu = price_european_tree(market, T, N, K, cp)
            us = price_american_tree(market, T, N, K, cp)
            print(f"{N:4d} | {eu:12.6f}  {eu - bs:+11.4e} | {us:12.6f}  {us - eu:+11.4e}")
            last_eu, last_us = eu, us

        if cp == "call":
            print("\nNote: Sans dividendes, CALL américain ≈ CALL européen.")
        else:
            print("\nNote: PUT américain doit être ≥ PUT européen (exercice anticipé).")


# ===========================================================
# Point d’entrée: scénario simple et lisible
# ===========================================================
def main():
    # Paramètres marché et dates d’exemple
    market = Market(underlying=102.45, rate=0.04, vol=0.28)
    start_date = datetime(2025, 10, 29)
    end_date = datetime(2026, 8, 27)

    # Maturité en années (base 365) et option de travail
    T = (end_date - start_date).days / 365.0
    option = Option(t=T, call_put="call", K=90, type="european", div=3, div_date=datetime(2026, 6, 9))

    # Maillage de référence
    N_large = 1000
    tree = Tree(market=market, nb_steps=N_large, delta_t=T / N_large)

    # Construction de l’arbre (timé)
    t0 = time.perf_counter()
    tree.build_bottom_first(option)
    t1 = time.perf_counter()
    print(f"⏱ Construction arbre : {t1 - t0:.4f} s")

    # Pricing EU — backward et récursif
    t2 = time.perf_counter()
    eu_back = tree.price(option, engine="backward", style="european")
    t3 = time.perf_counter()
    t4 = time.perf_counter()
    eu_rec = tree.price(option, engine="recursive", style="european")
    t5 = time.perf_counter()

    # Pricing US — backward et récursif
    t6 = time.perf_counter()
    us_back = tree.price(option, engine="backward", style="american")
    t7 = time.perf_counter()
    t8 = time.perf_counter()
    us_rec = tree.price(option, engine="recursive", style="american")
    t9 = time.perf_counter()

    # Affichage synthétique
    print("=" * 70)
    print(f"EU (N={N_large}) : {eu_back:.6f} (backward, {t3 - t2:.4f}s) | {eu_rec:.6f} (recursive, {t5 - t4:.4f}s)")
    print(f"US (N={N_large}) : {us_back:.6f} (backward, {t7 - t6:.4f}s) | {us_rec:.6f} (recursive, {t9 - t8:.4f}s)")
    bs_ref = black_scholes_price(market.underlying, option.K, T, market.rate, market.vol, option.call_put)
    print(f"BS (EU, sans div) : {bs_ref:.6f}")
    print("=" * 70)

    # Greeks locales à la racine (no-bump)
    greeks = tree.local_greeks_no_bump(option)
    print("\n=== Greeks (no-bump, à la racine) ===")
    print(f"Price = {greeks.get('price', np.nan):.6f}")
    print(f"Delta = {greeks.get('delta', np.nan):.6f}")
    print(f"Gamma = {greeks.get('gamma', np.nan):.6e}")
    print(f"Vega  = {greeks.get('vega',  np.nan):.6f}")

    # Exemples optionnels (décommentez si besoin)
    # run_convergence(market, option, Ns=[10, 25, 50, 100, 200, 400])
    # S0 = market.underlying; plot_gap_vs_strike(market, option, N=200, K_min=0.6*S0, K_max=1.4*S0, nK=41)


if __name__ == "__main__":
    main()