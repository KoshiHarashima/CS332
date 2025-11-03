# utility.py
def calculate_utility(value, allocation, payment):
    """
    Calculate utility: u = v * y - p
    
    Args:
        value: float (v) - value of the item
        allocation: float (y) - allocation (1.0 if wins, 0.5 if tie, 0.0 if loses)
        payment: float (p) - payment made
    
    Returns:
        utility: float (u) - utility = value * allocation - payment
    """
    return value * allocation - payment