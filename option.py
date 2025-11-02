from datetime import datetime

class Option:
    def __init__(self, t, call_put : str, K, type : str, div: float = 0.0, div_date: datetime = None):
        self.t = t                # time to maturity
        self.call_put = call_put  # 'call' or 'put'
        self.K = K                # strike price
        self.type = type          # 'european' or 'american'
        self.div = div
        self.div_date = div_date
        self.start_date = datetime(2025, 10, 29)

    @property
    def strike(self):
        """Backward-compatible alias for the strike price (expected by other modules)."""
        return self.K

    def payoff(self, S: float) -> float:
        if self.call_put == "call":
            return max(S - self.K, 0.0)
        elif self.call_put == "put":
            return max(self.K - S, 0.0)
        else:
            raise ValueError("call_put doit être 'call' ou 'put'")

    def get_div_time_in_years(self) -> float:
        """Convertit la date du dividende en temps relatif (années)."""
        if not self.div or self.div <= 0 or not self.div_date:
            return None
        delta_days = (self.div_date - self.start_date).days
        if delta_days <= 0 or delta_days > self.t * 365:
            return None
        return delta_days / 365.0
    

        
    





