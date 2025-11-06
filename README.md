# Trinomial Pricer (Python + Streamlit)

Pricing d’options via un arbre trinomial:
- EU/US, moteurs backward et récursif
- Référence Black–Scholes (EU, sans dividende discret)
- Greeks locales (no-bump)
- Analyses: convergence vs N et Gap(Tree − BS) vs Strike
- Dividende discret optionnel (montant + date)

Code principal: [Python/streamlit_app.py](Python/streamlit_app.py) (UI), [Python/market.py](Python/market.py), [Python/option.py](Python/option.py), [Python/tree.py](Python/tree.py).  
Script console (sans UI): [main.py](main.py)

## Prérequis
- Python 3.9+ recommandé
- macOS ou Windows
- Dépendances: streamlit, numpy, scipy, matplotlib

## Installation rapide
- Avec venv (recommandé):
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install streamlit numpy scipy matplotlib
```

- Avec conda:
```bash
conda create -n trinomial python=3.11 -y
conda activate trinomial
pip install streamlit numpy scipy matplotlib
```

## Lancer l’app Streamlit
Choisir UNE des deux méthodes:

- Depuis la racine du projet:
```bash
streamlit run Python/streamlit_app.py
```

- Depuis le sous-dossier Python:
```bash
cd Python
streamlit run streamlit_app.py
```

Notes:
- Le fichier [Python/streamlit_app.py](Python/streamlit_app.py) gère les imports locaux; pas besoin de modifier PYTHONPATH.
- Assurez-vous d’utiliser le même interpréteur (venv/conda) où les paquets ont été installés.

## À propos de `main.py` (fonctions support et exécution)
- Le fichier `main.py` contient des fonctions supports réutilisables (ex.: `black_scholes_price`, `run_convergence`, `price_european_tree`, `price_american_tree`, `compute_gap_vs_strike`, `plot_gap_vs_strike`, `compare_euro_amer_call_put`, etc.).
- Pour lancer vos propres essais/scripts, placez votre logique dans la fonction `main()` et exécutez le fichier. Le patron est déjà présent :

```python
def main():
  # Votre scénario ici (construction Market/Option/Tree, appels de fonctions support, etc.)
  pass

if __name__ == "__main__":
  main()
```

## Script console (optionnel)
Pour exécuter la démo console (timings, grecs):
```bash
python main.py
```

Remarque : pour des scénarios personnalisés, éditez le contenu de `main()` (plutôt que d’ajouter du code en haut de fichier) afin de garder une exécution propre et reproductible.



