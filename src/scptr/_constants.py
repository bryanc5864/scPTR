"""Layer name constants and defaults for scPTR."""

# Layer names
UNSPLICED = "unspliced"
SPLICED = "spliced"
SMOOTHED_UNSPLICED = "Mu"
SMOOTHED_SPLICED = "Ms"
GAMMA = "gamma"
PT_VELOCITY = "pt_velocity"
VELOCITY_S = "velocity_S"

# var columns
BETA = "beta"
TF_SCORE = "tf_score"
PTF_SCORE = "ptf_score"

# obs columns
PT_STATE = "pt_state"

# uns keys
UNS_KEY = "scptr"

# Defaults
DEFAULT_N_NEIGHBORS = 30
DEFAULT_BANDWIDTH = "adaptive"
DEFAULT_BETA_QUANTILE = 0.95
DEFAULT_CLIP_QUANTILE = 0.99
DEFAULT_GENE_BATCH_SIZE = 500
DEFAULT_LEIDEN_RESOLUTION = 1.0
DEFAULT_MIN_UNSPLICED_COUNTS = 10
DEFAULT_MIN_UNSPLICED_CELLS = 10
