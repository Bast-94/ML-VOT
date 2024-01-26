import argparse

def get_args():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-n", "--n-frame", type=int, default=1, help="Frame number")
    arg_parser.add_argument("-g", "--gif", action="store_true", help="Create gif")
    return arg_parser.parse_args()
