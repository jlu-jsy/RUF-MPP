import argparse
import logging


def get_config():
    """Parse command-line arguments and print the experiment configuration."""
    parser = argparse.ArgumentParser(
        description="PyTorch implementation of graph neural network fine-tuning"
    )

    # Training hyperparameters
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--num_layer", type=int, default=3)
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--dropout_ratio", type=float, default=0.0)
    parser.add_argument(
        "--graph_pooling",
        type=str,
        default="mean",
        choices=["mean", "set2set", "attn"],
    )
    parser.add_argument(
        "--JK",
        type=str,
        default="last",
        help="How node features across layers are combined",
    )
    parser.add_argument("--gnn_type", type=str, default="gin")

    # Multi-task learning strategy
    parser.add_argument(
        "--mtl_loss",
        type=str,
        default="none",
        choices=["equal", "uncertainty", "uncertainty_v2", "dwa", "gradnorm", "pcgrad", "attn"],
        help="Multi-task learning loss strategy",
    )
    parser.add_argument(
        "--mtl_temp",
        type=float,
        default=2.0,
        help="Temperature parameter for DWA and uncertainty_v2",
    )
    parser.add_argument(
        "--mtl_alpha",
        type=float,
        default=1.5,
        help="Alpha parameter for GradNorm",
    )

    # Dataset and split settings
    parser.add_argument("--dataset", type=str, default="bbbp")
    parser.add_argument("--seed", type=int, default=-1, help="Seed for dataset splitting")
    parser.add_argument("--runseed", type=int, default=0, help="Seed for minibatch sampling and initialization")
    parser.add_argument(
        "--split",
        type=str,
        default="balanced_scaffold",
        choices=["scaffold", "random", "balanced_scaffold"],
    )
    parser.add_argument("--num_workers", type=int, default=16)

    # Data augmentation
    parser.add_argument(
        "--aug",
        type=str,
        default="none",
        choices=["none", "llm", "rule", "random"],
        help="Data augmentation type",
    )
    parser.add_argument(
        "--aug_ratio",
        type=float,
        default=0.0,
        help="Augmentation ratio relative to the original dataset; 0 disables augmentation",
    )
    parser.add_argument(
        "--rule_probs",
        type=str,
        default="0.45,0.45,0.10",
        help="Rule augmentation probabilities for atom_replace,bond_change,scaffold_hop",
    )

    # Repeated runs and learning-rate schedule
    parser.add_argument("--num_runs", type=int, default=5)
    parser.add_argument("--warmup_epochs", type=float, default=2.0)
    parser.add_argument("--max_lr", type=float, default=1e-4)
    parser.add_argument("--final_lr", type=float, default=1e-5)

    # Evaluation and early stopping
    parser.add_argument("--test_if", type=str, default="no")
    parser.add_argument("--analyze", type=str, default="yes")

    args = parser.parse_args()
    _log_config(args)
    return args


def _log_config(args):
    """Print parsed arguments in a deterministic order."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger(__name__)

    logger.info("=== Configuration ===")
    for key, value in sorted(vars(args).items()):
        logger.info(f"{key:<20}: {value}")
    logger.info("=" * 30)
