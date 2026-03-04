import logging
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


def setup_run_logger(run_name: str, tz: str = "Europe/Berlin") -> tuple[Path, logging.Logger]:
    ts = datetime.now(ZoneInfo(tz)).strftime("%Y%m%d_%H%M%S")
    run_dir = Path("logs") / run_name / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)

    return run_dir, logger
