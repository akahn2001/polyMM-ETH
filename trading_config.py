"""
Trading configuration constants
Shared between trading.py and markouts.py to avoid circular imports
"""

# Book imbalance adjustment
USE_BOOK_IMBALANCE = True
BOOK_IMBALANCE_LEVELS = 4         # how many price levels to consider (0 for all)
MAX_IMBALANCE_ADJUSTMENT = 0.025  # Max price adjustment from book imbalance

# Momentum adjustment
MAX_MOMENTUM_ADJUSTMENT = 0.03    # Max price adjustment from momentum (caps at 3 cents)

# Z-score skew (continuous adjustment based on predicted RTDS movement)
MAX_Z_SCORE_SKEW = 0.035          # Cap z-score skew at ±3.5 cents

# Z-score confidence scaling (sigmoid function to filter noise)
# Scales z_skew based on z_score magnitude to avoid amplifying noise with high gamma
Z_SCORE_CONFIDENCE_MIDPOINT = 0.4   # Z-score value where confidence = 50%
Z_SCORE_CONFIDENCE_STEEPNESS = 5.0  # How sharp the sigmoid transition is

# Cap on total signal adjustments (book imbalance + z-score skew combined)
# Prevents crossing spread when both signals fire strongly in same direction
MAX_TOTAL_SIGNAL_ADJUSTMENT = 0.0275  # Cap combined adjustments at ±3.0¢ from mid
