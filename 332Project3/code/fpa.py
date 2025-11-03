# fpa.py

class FPA:
    """
    First Price Auction (2 players only)
    Simple version that only returns allocation.
    """
    
    def __init__(self):
        pass
    
    def play_round(self, bid1, bid2):
        """
        Play one round of FPA.
        
        Args:
            bid1: bid from player 1
            bid2: bid from player 2
        
        Returns:
            (allocation1, allocation2): tuple of floats
                - allocation1: allocation for player 1 (1.0 if wins, 0.5 if tie, 0.0 if loses)
                - allocation2: allocation for player 2 (1.0 if wins, 0.5 if tie, 0.0 if loses)
        """
        if bid1 > bid2:
            return (1.0, 0.0)
        elif bid2 > bid1:
            return (0.0, 1.0)
        else:  # tie
            return (0.5, 0.5)