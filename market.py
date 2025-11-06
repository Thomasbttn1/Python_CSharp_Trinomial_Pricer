from datetime import datetime

# Paramètres de marché utilisés par le pricer
# On garde une classe minimale: S0 (spot), r (taux), sigma (vol)

class Market:
    """Conteneur simple pour les paramètres de marché."""
    def __init__(self, underlying, rate, vol):
        # Attributs principaux
        self.underlying = float(underlying)  # spot S0
        self.rate = float(rate)              # taux sans risque r
        self.vol = float(vol)

