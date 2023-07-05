from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .C import CONFIGURATION, LOG, RATE, RAW


class SeahorseExperiment:
    """Class representing a Seahorse experiment / result file"""

    def __init__(self, file: Path | str):
        file = Path(file)

        # load from directory created by `ods_to_tsv`,
        # or directly from an .ods file (much slower)
        if file.is_dir():
            self.df_all = {}
            for f in file.glob("*.tsv"):
                self.df_all[f.stem] = pd.read_csv(f, sep="\t")
        else:
            self.df_all = pd.read_excel(file, sheet_name=None)

        self.df_raw = self.df_all[RAW]
        self.df_op = self.df_all[LOG]

        self._preprocess_operation_log()
        self._preprocess_raw()

    def _preprocess_operation_log(self):
        """Preprocess OperationLog sheet"""
        df_op = self.df_op
        df_op["start_dt"] = pd.to_datetime(df_op["Start Time"])
        df_op["end_dt"] = pd.to_datetime(df_op["End Time"])

        # TODO what is to be considered t=0? end of equilibration for now
        #  6s--18s off from measurement times
        #  t0 = df_op["start_dt"][df_op["Instruction Name"] == "Initialization"].values[0]
        t0 = df_op["end_dt"][df_op["Instruction Name"] == "Equilibrate"].values[
            0
        ]

        df_op["start_s"] = (df_op["start_dt"] - t0).dt.total_seconds()
        df_op["end_s"] = (df_op["end_dt"] - t0).dt.total_seconds()
        df_op["duration_s"] = df_op["end_s"] - df_op["start_s"]

    def _preprocess_raw(self):
        """Preprocess "Raw" sheet"""
        df_raw = self.df_raw
        assert "Measurement" in df_raw.columns.values[0]
        df_raw.columns.values[0] = "Measurement"

        df_raw["datetime"] = pd.to_datetime(df_raw.TimeStamp, format="%H:%M:%S")
        # TODO t0 unclear here
        t0 = df_raw["datetime"].min()
        df_raw["time_s"] = (df_raw["datetime"] - t0).dt.total_seconds()

    def plot_perturbations(self, time_unit="s", show=False):
        """Plot injection time span and label the perturbations."""
        assert time_unit in ["s", "min"]
        time_conv = 1 / 60 if time_unit == "min" else 1
        df_op = self.df_op
        injections = df_op[df_op["Command Name"] == "Inject"]
        for _, row in injections.iterrows():
            plt.axvspan(
                row["start_s"] * time_conv, row["end_s"] * time_conv, color="r"
            )
            plt.text(
                s=row["Instruction Name"],
                x=row["end_s"] * time_conv,
                y=plt.ylim()[0],
                color="r",
            )
        plt.xlabel(f"time [{time_unit}]")
        plt.xlim(0, df_op["end_s"].max() * time_conv)
        if show:
            plt.show()

    def config(self, key: str = None) -> dict:
        """Get assay configuration as dict, or a specific entry.

        For now, only the simple key, value pairs.

        To be extended.
        """
        df = self.df_all[CONFIGURATION]
        # TODO to proper data types
        config = {t._1: t._2 for t in df.itertuples() if not pd.isna(t._1)}

        return config[key] if key else config

    @property
    def rate(self):
        return self.df_all[RATE]

    @property
    def raw(self):
        return self.df_all[RAW]

    def small_multiples_rate(self):
        """Plot small multiples of all measurements"""
        # assumes 96 well plates
        assert 96 == len(self.rate["Well"].unique())
        mosaic = multi_well_mosaic_96()
        fig, axs = plt.subplot_mosaic(
            mosaic, sharex=True, sharey=True, figsize=(18, 10)
        )
        fig.suptitle(self.config("Project Name"))
        for (group,), group_df in self.rate.groupby(["Well"]):
            ax = axs[group]
            ax.set_title(group_df.Group.values[0])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.plot(group_df["Time"], group_df["OCR"], color="b")
            ax.plot(group_df["Time"], group_df["ECAR"], color="r")
            # TODO perturbations
            # TODO condition
        # TODO label rows and columns
        # TODO plot all trajectories with low alpha as background,
        plt.tight_layout()
        plt.subplots_adjust(hspace=1)
        # plt.show()

    def small_multiples_raw(self):
        """Plot small multiples of all raw measurements"""
        # assumes 96 well plates
        assert 96 == len(self.raw["Well"].unique())
        mosaic = multi_well_mosaic_96()
        fig, axs = plt.subplot_mosaic(
            mosaic, sharex=True, sharey=True, figsize=(18, 10)
        )
        fig.suptitle(self.config("Project Name"))
        for (group,), group_df in self.raw.groupby(["Well"]):
            ax = axs[group]
            ax.set_title(group_df.Group.values[0])
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines["left"].set_visible(False)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.plot(group_df["time_s"], group_df["O2 (mmHg)"], color="b")
            # TODO separate axis for pH
            ax.plot(group_df["time_s"], group_df["pH"], color="r")
            # TODO perturbations
            # TODO condition
        # TODO label rows and columns
        # TODO plot all trajectories with low alpha as background,
        plt.tight_layout()
        plt.subplots_adjust(hspace=1)
        # plt.show()

    def __repr__(self):
        return f"<{self.__class__.__name__}({self.config('Project Name')})>"


def multi_well_mosaic_96() -> list[list[str]]:
    """Create a mosaic for 96-well plate well labels for use with
    ``matplotlib.pyplot.subplot_mosaic``."""

    return multi_well_mosaic(nrows=8, ncols=12)


def multi_well_mosaic(nrows: int, ncols: int) -> list[list[str]]:
    """Create a mosaic for multi-well plate well labels for use with
    ``matplotlib.pyplot.subplot_mosaic``."""
    from math import floor, log10
    from string import ascii_uppercase

    assert nrows <= len(ascii_uppercase), "not enough letters"

    col_width = floor(log10(ncols)) + 1
    rows = ascii_uppercase[:nrows]
    cols = range(1, 1 + ncols)

    return [[f"{r}{str(c).zfill(col_width)}" for c in cols] for r in rows]
