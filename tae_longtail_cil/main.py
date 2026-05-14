from tae_study.experiment import parse_args, run_experiment


def main() -> None:
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
