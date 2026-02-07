from src.model.emb_lr import run_emb_lr_gridsearch
from src.utils.run_logger import setup_run_logger


def main() -> None:
    run_dir, log = setup_run_logger("emb_lr")
    log.info("Run dir: %s", run_dir.as_posix())
    run_emb_lr_gridsearch(run_dir, log)


if __name__ == "__main__":
    main()
