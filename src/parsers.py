import argparse


def get_track_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-n", "--n-frame", type=int, default=1, help="Frame number")
    arg_parser.add_argument("-g", "--gif", action="store_true", help="Create gif")
    arg_parser.add_argument(
        "-H", "--hungarian", action="store_true", help="Use Hungarian algorithm"
    )
    arg_parser.add_argument("-a", "--all", action="store_true", help="Use all frames")
    return arg_parser.parse_args()


def get_git_manager_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-c", "--config", type=str, default="secretdir/config.yml", help="Config file"
    )
    # add subparsers
    subparsers = arg_parser.add_subparsers(dest="commands")
    artifacts_parser = subparsers.add_parser("artifacts")
    artifacts_parser.add_argument(
        "-l", "--link", type=str, default="", help="Artifact link"
    )

    tree_parser = subparsers.add_parser("tree")
    tree_parser.add_argument("-r", "--repo", type=str, default="", help="Repo name")
    tree_parser.add_argument("-u", "--user", type=str, default="", help="Tree url")

    tree_parser.add_argument(
        "-b", "--branch", type=str, default="main", help="Branch name"
    )

    return arg_parser.parse_args()
