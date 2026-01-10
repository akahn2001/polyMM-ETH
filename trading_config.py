"""
Trading configuration constants
Shared between trading.py and markouts.py to avoid circular imports
"""

# Book imbalance adjustment
USE_BOOK_IMBALANCE = True
BOOK_IMBALANCE_LEVELS = 4         # how many price levels to consider (0 for all)
MAX_IMBALANCE_ADJUSTMENT = 0.035  # Max price adjustment from book imbalance

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
MAX_TOTAL_SIGNAL_ADJUSTMENT = 0.035  # Cap combined adjustments at ±3.5¢ from mid

# Aggressive mode: increase cap when high conviction signals align
# Allows crossing spread to take liquidity when edge is high
AGGRESSIVE_MODE_ENABLED = True
AGGRESSIVE_Z_THRESHOLD = 1.5          # Minimum |z-score| to trigger aggressive mode
AGGRESSIVE_ZSKEW_THRESHOLD = 0.06     # Minimum |z_skew_residual| (edge remaining after market moved)
AGGRESSIVE_MAX_TOTAL_ADJUSTMENT = 0.06  # 4¢ cap when aggressive (crosses spread by 1+ tick)
AGGRESSIVE_MAX_Z_SCORE_SKEW = 0.06     # Allow 4.5¢ z_skew in aggressive mode (vs 3.5¢ normal)
