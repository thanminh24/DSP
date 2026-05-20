from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import openml
import seaborn as sns
import matplotlib

PROJECT_ROOT = Path(__file__).resolve().parent.parent

EXPECTED_SHAPES = {
    "pima.parquet": (768, 9),
    "credit-g.parquet": (1000, 21),
    "sick.parquet": (3772, 30),
}


def main() -> None:
    print(f"Python packages:")
    print(f"  scikit-learn {sklearn.__version__}")
    print(f"  numpy        {np.__version__}")
    print(f"  pandas       {pd.__version__}")
    print(f"  openml       {openml.__version__}")
    print(f"  seaborn      {sns.__version__}")
    print(f"  matplotlib   {matplotlib.__version__}")
    print()

    data_dir = PROJECT_ROOT / "data"
    for fname, expected_shape in EXPECTED_SHAPES.items():
        path = data_dir / fname
        assert path.exists(), f"Missing data file: {path}"
        df = pd.read_parquet(path)
        actual = df.shape
        assert actual == expected_shape, f"{fname}: expected {expected_shape}, got {actual}"
        print(f"  {fname}: shape={actual} OK")

    plots_dir = PROJECT_ROOT / "outputs" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n  outputs/plots/ exists OK")

    print("\nVALIDATION OK")


if __name__ == "__main__":
    main()
