import argparse
from labels.build_labels import build_label_dataset


def main():
    parser = argparse.ArgumentParser(description="Build label dataset for RPI-DSATUR")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build labels from graphs + edits")
    build_parser.add_argument("--graphs-dir", type=str, default="data/raw/graphs")
    build_parser.add_argument("--edits-dir", type=str, default="data/raw/edits")
    build_parser.add_argument("--out", type=str, default="data/labels/labels.pt")

    args = parser.parse_args()
    if args.command == "build":
        build_label_dataset(args.graphs_dir, args.edits_dir, args.out)


if __name__ == "__main__":
    main()
