# =====================================================================
# AI Virtual Mouse Optimized Configuration
# =====================================================================

# Landmark Indexes
INDEX_FINGER_TIP = 8
INDEX_FINGER_MCP = 5
THUMB_TIP = 4
MIDDLE_FINGER_TIP = 12
MIDDLE_FINGER_MCP = 9
WRIST = 0

# Active Camera Region Box (Trims frame edges)
FRAME_MARGIN = 100

# Smoothing & Jitter Control
SMOOTHING_FACTOR = 0.18  # Lower = smoother cursor motion
DEADZONE_PIXELS = 3      # Ignores tiny micro-vibrations under 3 pixels

# Dynamic Normalized Pinch Ratios (Distance / Hand Reference Scale)
NORM_CLICK_RATIO = 0.18        # Left click threshold ratio
NORM_RIGHT_CLICK_RATIO = 0.18  # Right click threshold ratio
NORM_SCROLL_RATIO = 0.25       # Index-Middle pinch ratio for scroll mode

DRAG_DELAY = 0.35  # Latch delay for click & drag