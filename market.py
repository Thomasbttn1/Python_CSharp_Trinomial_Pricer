from datetime import datetime

class Market: 
    def __init__(self, underlying, rate, vol):
        self.underlying = underlying  # underlying asset price
        self.rate = rate              # risk-free interest rate
        self.vol = vol                # volatility of the underlying asset

