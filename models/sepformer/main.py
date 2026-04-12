from pathlib import Path

import torch
from loguru import logger

from .dataset import get_dataloaders
from .engine import Engine
from .model import Model
from sepformer_utils import util_implement
from sepformer_utils.decorators import logger_wraps

_SCRIPT_DIR = Path(__file__).resolve().parent

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None


@logger_wraps()
def main(args):
    config = args.config
    log_dir = Path(config["runtime"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "system_log.log", level="DEBUG", mode="w")

    wandb_run = None
    if hasattr(args, "no_wandb") and not args.no_wandb and HAS_WANDB:
        try:
            wandb_run = wandb.init(
                project=getattr(args, "wandb_project", None) or "audio-separation",
                name=getattr(args, "wandb_name", None) or f"sepformer_bs{config['dataloader']['batch_size']}",
                config=config,
                dir=config["runtime"]["wandb_dir"],
            )
            logger.info(f"[WandB] Initialized: {wandb_run.url}")
        except Exception as exc:
            logger.warning(f"[WandB] init failed: {exc}")

    dataloaders = get_dataloaders(args, config["dataset"], config["dataloader"])
    model = Model(**config["model"])

    gpuid = tuple(map(int, config["engine"]["gpuid"].split(",")))
    device = torch.device(f"cuda:{gpuid[0]}" if torch.cuda.is_available() else "cpu")

    criterions = util_implement.CriterionFactory(config["criterion"], device).get_criterions()
    optimizers = util_implement.OptimizerFactory(config["optimizer"], model.parameters()).get_optimizers()
    schedulers = util_implement.SchedulerFactory(config["scheduler"], optimizers).get_schedulers()

    engine = Engine(args, config, model, dataloaders, criterions, optimizers, schedulers, gpuid, device, wandb_run=wandb_run)
    engine.run()

    if wandb_run is not None:
        wandb_run.finish()
