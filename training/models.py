"""Model architecture aliases for backward compatibility."""
from training.model import build_custom_cnn, build_residual_compact_cnn, get_model

# Aliases
build_baseline_cnn = build_custom_cnn
build_mini_xception = build_residual_compact_cnn

__all__ = ["build_custom_cnn", "build_residual_compact_cnn", "get_model", "build_baseline_cnn", "build_mini_xception"]
