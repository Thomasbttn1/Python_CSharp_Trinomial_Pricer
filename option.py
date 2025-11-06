from datetime import datetime  # gestion des dates

class Option:  # option vanille
    def __init__(self, t, call_put : str, K, type : str, div: float = 0.0, div_date: datetime = None):  # init
        self.t = t                # time to maturity
        self.call_put = call_put  # 'call' or 'put'
        self.K = K                # strike price
        self.type = type          # 'european' or 'american'
        self.div = div            # dividende discret (montant)
        self.div_date = div_date  # date du dividende (ou None)
        self.start_date = datetime(2025, 10, 29)  # date de départ pour le temps

    @property  # alias compatible
    def strike(self):  # retourne le strike
        """Backward-compatible alias for the strike price (expected by other modules)."""
        return self.K  # valeur du strike

    def payoff(self, S: float) -> float:  # payoff terminal
        if self.call_put == "call":  # cas call
            return max(S - self.K, 0.0)  # max(S-K,0)
        elif self.call_put == "put":  # cas put
            return max(self.K - S, 0.0)  # max(K-S,0)
        else:  # type inconnu
            raise ValueError("call_put doit être 'call' ou 'put'")  # garde-fou

    def get_div_time_in_years(self) -> float:  # date de div en années
        """Convertit la date du dividende en temps relatif (années)."""
        if not self.div or self.div <= 0 or not self.div_date:  # pas de div valide
            return None  # rien à retourner
        delta_days = (self.div_date - self.start_date).days  # nombre de jours
        if delta_days <= 0 or delta_days > self.t * 365:  # hors fenêtre [0, T]
            return None  # ignore
        return delta_days / 365.0  # conversion en années









