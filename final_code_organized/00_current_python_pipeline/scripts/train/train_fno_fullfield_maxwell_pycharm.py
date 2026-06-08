import multiprocessing as mp
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train.train_fno_curvefield_hybrid import main


if __name__ == "__main__":
    mp.freeze_support()
    main()
