"""Run the existing 3D training and prediction steps as one workflow."""

from workbase.current_scripts.benchmark_3d import main as run_benchmark
from workbase.current_scripts.predict_3d import main as run_prediction


def main() -> None:
    run_benchmark()
    run_prediction()


if __name__ == "__main__":
    main()
