import argparse

from .experiments import run_suite


def main():
    parser = argparse.ArgumentParser(description="Reproduce the mean-field execution research suite")
    parser.add_argument("--profile", choices=["quick", "full"], default="quick")
    parser.add_argument("--output", default="results/quick")
    parser.add_argument("--trials", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20261117)
    args = parser.parse_args()
    run_suite(args.output, args.profile, args.trials, args.seed)


if __name__ == "__main__":
    main()

