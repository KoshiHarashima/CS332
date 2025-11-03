# utility.py
def calculate_utility(value, allocation, bid):
    """
    Calculate utility: u = allocation * (value - bid)
    
    Args:
        value: float (v) - value of the item
        allocation: float (y) - allocation (1.0 if wins, 0.5 if tie, 0.0 if loses)
        bid: float (b) - bid amount
    
    Returns:
        utility: float (u) - utility = allocation * (value - bid)
    """
    return allocation * (value - bid)