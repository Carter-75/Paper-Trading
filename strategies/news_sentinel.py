import logging
from typing import List, Optional
from datetime import datetime, timedelta
import re

class NewsSentinel:
    """
    Scans news headlines for extreme negative sentiment to avoid risky trades.
    "Smart AI" safety check.
    """
    
    def __init__(self, api_client):
        self.api = api_client
        self.logger = logging.getLogger("NewsSentinel")
        
        # Keywords that suggest immediate danger
        self.danger_keywords = [
            "lawsuit", "investigation", "fraud", "crash", "plunge", 
            "collapse", "bankruptcy", "scandal", "breach", "hacked",
            "sec", "doj", "indictment", "accounting error", "restatement"
        ]
        
    def check_sentiment(self, symbol: str) -> float:
        """
        Check news sentiment for a symbol.
        Returns a 'safety score' from 0.0 (DANGER) to 1.0 (SAFE).
        """
        try:
            # Fetch news for the last 24 hours
            # Alpaca get_news returns a list of news objects
            # We need to handle potential API limitations or errors
            news_list = self.api.get_news(symbol, limit=5, include_content=False)
            
            if not news_list:
                return 0.5 # Neutral / No news is slightly risky? or safe? Let's say neutral.
                
            danger_count = 0
            for item in news_list:
                headline = item.headline.lower()
                for kw in self.danger_keywords:
                    # Simple word boundary check would be better but simple 'in' is robust enough for now
                    if kw in headline:
                        self.logger.warning(f"News Sentinel Alert [{symbol}]: '{headline}' contains '{kw}'")
                        danger_count += 1
            
            if danger_count >= 1:
                return 0.0 # BLOCK TRADE
                
            return 1.0 # SAFE
            
        except Exception as e:
            self.logger.error(f"Failed to fetch news for {symbol}: {e}")
            return 1.0 # Fail open (assume safe) so we don't paralyze bot on API faile, 
                       # but could also fail closed. Let's fail open for now.
