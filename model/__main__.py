import sys
from model.train import train


def main():
    config_path = sys.argv[1] if len(sys.argv) > 1 else "model/config.yaml"
    train(config_path)


if __name__ == "__main__":
    main()