import argparse
from data_gen.generate_graphs import generate_dataset
from data_gen.generate_edits import generate_edits_for_graphs, augment_edits_for_target_r1


def main():
    parser = argparse.ArgumentParser(description="Data generation for RPI-DSATUR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Generate graphs
    gen_parser = subparsers.add_parser("generate", help="Generate synthetic graphs")
    gen_parser.add_argument("--sizes", nargs="+", type=int, default=[100, 200, 500])
    gen_parser.add_argument("--graphs-per-size", type=int, default=50)
    gen_parser.add_argument("--base-seed", type=int, default=0)
    gen_parser.add_argument("--out", type=str, default="data/raw/graphs")

    # Generate edits
    edit_parser = subparsers.add_parser("edits", help="Generate edit streams")
    edit_parser.add_argument("--graphs-dir", type=str, default="data/raw/graphs")
    edit_parser.add_argument("--num-edits", type=int, default=20)
    edit_parser.add_argument("--seed", type=int, default=0)
    edit_parser.add_argument("--out", type=str, default="data/raw/edits")

    r1_parser = subparsers.add_parser(
        "augment_r1",
        help="Append genuine r*=1 one-off edits to existing streams (does not rewrite `edits`)",
    )
    r1_parser.add_argument("--graphs-dir", type=str, default="data/raw/graphs")
    r1_parser.add_argument("--edits-dir", type=str, default="data/raw/edits")
    r1_parser.add_argument("--n-per-snapshot", type=int, default=4)
    r1_parser.add_argument("--max-attempts", type=int, default=100)
    r1_parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    if args.command == "generate":
        generate_dataset(args.sizes, args.graphs_per_size, args.base_seed, args.out)
    elif args.command == "edits":
        generate_edits_for_graphs(args.graphs_dir, args.num_edits, args.seed, args.out)
    elif args.command == "augment_r1":
        import logging
        logging.basicConfig(level=logging.INFO)
        augment_edits_for_target_r1(
            args.graphs_dir,
            args.edits_dir,
            n_per_snapshot=args.n_per_snapshot,
            max_attempts=args.max_attempts,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()