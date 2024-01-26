import argparse


def get_track_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-n", "--n-frame", type=int, default=1, help="Frame number")
    arg_parser.add_argument("-g", "--gif", action="store_true", help="Create gif")
    arg_parser.add_argument(
        "-H", "--hungarian", action="store_true", help="Use Hungarian algorithm"
    )
    return arg_parser.parse_args()


def get_artifacts_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c", "--config", type=str, default="secretdir/config.yml", help="Config file"
    )

    return arg_parser.parse_args()
