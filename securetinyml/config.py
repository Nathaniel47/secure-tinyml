"""Dataset, seed and attack-strength choices shared by every experiment (see PLAN.md)."""

SEEDS = (0, 1, 2)

# name -> image side in pixels
DATASETS = {"cifar10": 32, "gtsrb": 32, "plantvillage": 64}
# Build order from PLAN.md: CIFAR-10 first, PlantVillage last and optional.
DATASET_ORDER = ("cifar10", "gtsrb", "plantvillage")

# L-infinity budgets in 1/255 pixel units
EPS_255 = (2, 4, 8)
# Patch area as a fraction of the image
PATCH_FRACTIONS = (0.04, 0.07, 0.10)
# Backdoor poison rates
POISON_RATES = (0.01, 0.02, 0.05)

# One quantized format for evaluation and the browser (PLAN.md)
QUANT_FORMAT = "onnx-int8-qdq"
