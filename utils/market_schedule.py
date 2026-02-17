
import datetime
import pytz

class MarketSchedule:
    """
    Centralized logic for checking market hours.
    """
    @staticmethod
    def is_market_open(api_client=None) -> bool:
        """
        Return True if market is open.
        Prefer Alpaca clock (handles holidays/half-days). If Alpaca fails or client not provided,
        fall back to a simple 9:30-16:00 ET weekday window.
        """
        # 1) Alpaca clock (best)
        if api_client:
            try:
                clock = api_client.get_clock()
                return bool(getattr(clock, 'is_open', False))
            except Exception:
                pass

        # 2) Fallback time window (ET)
        try:
            tz = pytz.timezone('US/Eastern')
            now_et = datetime.datetime.now(tz)
            # weekday: Mon=0 .. Sun=6
            if now_et.weekday() >= 5:
                return False
            open_min = 9 * 60 + 30
            close_min = 16 * 60
            now_min = now_et.hour * 60 + now_et.minute
            return open_min <= now_min < close_min
        except Exception:
            # If even fallback fails, be safe: treat as closed.
            return False
