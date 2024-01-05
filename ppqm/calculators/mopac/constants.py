MOPAC_METHOD = "PM6"
MOPAC_VALID_METHODS = ["PM3", "PM6", "PM7"]
MOPAC_CMD = "mopac"
MOPAC_ATOMLINE = "{atom:2s} {x} {opt_flag} {y} {opt_flag} {z} {opt_flag}"
MOPAC_INPUT_EXTENSION = "mop"
MOPAC_OUTPUT_EXTENSION = "out"
MOPAC_KEYWORD_CHARGE = "{charge}"
MOPAC_FILENAME = "_tmp_mopac." + MOPAC_INPUT_EXTENSION

MOPAC_DEFAULT_OPTIONS = {"precise": None, "mullik": None}
