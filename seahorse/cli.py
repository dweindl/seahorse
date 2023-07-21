import argparse
from pathlib import Path

from .experiment import SeahorseExperiment


def vis_cli():
    """Entry-point for the seahorse-vis command line tool"""
    parser = argparse.ArgumentParser(description="Visualize Seahorse data")
    parser.add_argument("filepath", type=Path)
    args = parser.parse_args()

    if args.filepath.is_file():
        outpath = args.filepath.with_suffix("")
    else:
        outpath = args.filepath

    experiment = SeahorseExperiment(args.filepath)
    experiment.plot_to_dir(outpath)
