"""Back-compat re-export. The implementation moved into the package
(`torch_pruning/utils/normnet_importance.py`) so the DDP harness
(`torch_pruning.utils.pruning_utils`) can import it without a benchmarks→package layering
violation. `prune_e2` and the tests keep importing from here unchanged.
"""
from torch_pruning.utils.normnet_importance import (  # noqa: F401
    NormalizedNetImportance,
    extract_input_channel_scores,
    extract_normnet_scores,
)

__all__ = [
    "NormalizedNetImportance",
    "extract_input_channel_scores",
    "extract_normnet_scores",
]
