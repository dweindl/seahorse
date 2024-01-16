from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .C import (
    SHEET_CONFIGURATION,
    SHEET_LOG,
    SHEET_NORMALIZED_RATE,
    SHEET_RATE,
    SHEET_RAW,
)
from .multi_well import MultiWellPlate


class SeahorseExperiment:
    """Class representing a Seahorse experiment / result file"""

    def __init__(self, file: Path | str, title: str = None):
        file = Path(file)

        # load from directory created by `spreadsheet_to_tsv`,
        # or directly from a spreadsheet (potentially slower)
        if file.is_dir():
            self._df_all = {}
            for f in file.glob("*.tsv"):
                self._df_all[f.stem] = pd.read_csv(f, sep="\t")
        else:
            self._df_all = pd.read_excel(
                file,
                sheet_name=None,
                engine="calamine",
            )

        self._preprocess_operation_log()
        self._preprocess_raw()
        self._title = title
        self._excluded_wells: set[str] = set()
        self._mwp = MultiWellPlate(nwells=96)

    def exclude_wells(self, wells: str):
        """Add a well to the exclusion list.

        E.g., for removing/labeling outliers.

        :param wells: Well label, e.g. "A1", or range of wells, e.g. "A1:A12",
            or comma-separated list of those.
        """
        for well_or_range in wells.split(","):
            if ":" in well_or_range:
                # range of wells
                for well in self._mwp.expand_range(well_or_range):
                    self._excluded_wells.add(well)
            else:
                # single well
                assert self._mwp.is_valid_well(
                    well_or_range
                ), f"Invalid well: {well_or_range}"
                self._excluded_wells.add(well_or_range)

    def _preprocess_operation_log(self):
        """Preprocess OperationLog sheet

        Add columns with relative timestamps, and duration of each operation.

        `Operation Log` has absolute timestamps, but `Raw` and `Rate*` have
        (different) relative timestamps.
        So, hat is to be considered t=0 in each sheet?

        For `Raw`, t=0 seems to be the end of `LoadProbes` (or the beginning of
        `Initialization`). For `Rate*`, t=0 seems to be the end of
        `Equilibrate`.
        """
        df_log = self.log

        # pd.to_datetime sometimes raises:
        #  UserWarning: Could not infer format, so each element will be parsed individually, falling back to `dateutil`.
        #  To ensure parsing is consistent and as-expected, please specify a format.
        #  Happens only if date format was messed up. Initially it corresponds to `2022-03-16 16:28:57`, which pandas
        #  handles well.
        df_log["start_dt"] = pd.to_datetime(df_log["Start Time"])
        df_log["end_dt"] = pd.to_datetime(df_log["End Time"])

        # t = 0: see docstring
        t0 = df_log["end_dt"][
            df_log["Instruction Name"] == "Equilibrate"
        ].values[0]

        df_log["start_s"] = (df_log["start_dt"] - t0).dt.total_seconds()
        df_log["end_s"] = (df_log["end_dt"] - t0).dt.total_seconds()
        df_log["duration_s"] = df_log["end_s"] - df_log["start_s"]

    def _preprocess_raw(self):
        """Preprocess "Raw" sheet"""
        df_raw = self.raw
        # Remove that funny comment, that pandas.read_excel doesn't want to ignore
        assert "Measurement" in df_raw.columns.values[0]
        df_raw.columns.values[0] = "Measurement"

        df_raw["datetime"] = pd.to_datetime(
            df_raw.TimeStamp, format="%H:%M:%S"
        )

        # t = 0: see docstring of `_preprocess_operation_log`
        t0 = self.log["end_dt"][
            self.log["Instruction Name"] == "Equilibrate"
        ].values[0]
        df_raw["time_s"] = (df_raw["datetime"] - t0).dt.total_seconds()

    def plot_perturbations(
        self,
        time_unit="s",
        show=False,
        ax: plt.Axes = None,
        label_axes=False,
        fontsize=None,
        staggered=True,
        color="grey",
    ) -> plt.Figure:
        """Plot injection time span and label the perturbations."""
        if fontsize is None:
            fontsize = plt.rcParams["font.size"]

        assert time_unit in ["s", "min"]
        time_conv = 1 / 60 if time_unit == "min" else 1
        df_log = self.log
        injections = self.injections

        if ax is None:
            ax = plt.gca()

        for i, (_, row) in enumerate(injections.iterrows()):
            ax.axvspan(
                row["start_s"] * time_conv,
                row["end_s"] * time_conv,
                color=color,
            )
            ax.annotate(
                text=row["Instruction Name"],
                xy=(row["start_s"] * time_conv, ax.get_ylim()[1]),
                xytext=(
                    0.5 * fontsize,
                    -(i + 1) * fontsize if staggered else -fontsize,
                ),
                textcoords="offset points",
                color=color,
                fontsize=fontsize,
            )
        if label_axes:
            ax.set_xlabel(f"time [{time_unit}]")
            ax.set_xlim(0, df_log["end_s"].max() * time_conv)
        fig = ax.figure
        if show:
            fig.show()
        return fig

    def config(self, key: str = None) -> dict | str:
        """Get assay configuration as dict, or a specific entry.

        For now, only the simple key, value pairs.

        To be extended.
        """
        df = self._df_all[SHEET_CONFIGURATION]
        # TODO to proper data types
        config = {t[1]: t[2] for t in df.itertuples() if not pd.isna(t[1])}
        config = {k.removesuffix(":").strip(): v for k, v in config.items()}
        return config[key] if key else config

    @property
    def log(self):
        return self._df_all[SHEET_LOG]

    @property
    def rate(self):
        return self._df_all[SHEET_RATE]

    @property
    def normalized_rate(self):
        return self._df_all[SHEET_NORMALIZED_RATE]

    @property
    def raw(self):
        return self._df_all[SHEET_RAW]

    @property
    def project_name(self) -> str:
        return self.config("Project Name")

    @property
    def assay_name(self) -> str:
        return self.config("Assay Name")

    @property
    def title(self) -> str:
        if self._title is not None:
            return self._title

        return self.assay_name or self.project_name

    @property
    def has_normalization(self):
        """Whether the experiment has normalized data."""
        return (
            SHEET_NORMALIZED_RATE in self._df_all
            and not self._df_all[SHEET_NORMALIZED_RATE].empty
        )

    @property
    def injections(self):
        df_log = self.log
        return df_log[df_log["Command Name"] == "Inject"]

    @property
    def groups(self):
        return self._df_all[SHEET_RAW]["Group"].unique().tolist()

    def small_multiples_rate(self, *, fig: plt.Figure = None) -> plt.Figure:
        """Plot small multiples of all measurements"""
        # assumes 96 well plates
        assert 96 == len(self.rate["Well"].unique())
        mosaic = multi_well_mosaic_96()
        if fig:
            axs = fig.subplot_mosaic(
                mosaic,
                sharex=True,
                sharey=True,
            )
        else:
            fig, axs = plt.subplot_mosaic(
                mosaic, sharex=True, sharey=True, figsize=(18, 10)
            )
        fig.suptitle(self.title)
        for (group,), group_df in self.rate.groupby(["Well"]):
            alpha = (
                0.2
                if self._mwp.remove_leading_zeroes(group)
                in self._excluded_wells
                else 1
            )
            ax = axs[group]
            ax.set_title(group_df.Group.values[0])
            _no_axes(ax)
            ax.plot(group_df["Time"], group_df["OCR"], color="b", alpha=alpha)
            # separate axis for ECAR
            ax2 = ax.twinx()
            _no_axes(ax2)
            ax2.plot(
                group_df["Time"], group_df["ECAR"], color="r", alpha=alpha
            )
            # TODO perturbations
        # TODO label rows and columns https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
        # TODO plot all trajectories with low alpha as background,
        plt.tight_layout()
        plt.subplots_adjust(hspace=1)

        return fig

    def small_multiples_raw(self) -> plt.Figure:
        """Plot small multiples of all raw measurements"""
        # assumes 96 well plates
        assert 96 == len(self.raw["Well"].unique())
        mosaic = multi_well_mosaic_96()
        fig, axs = plt.subplot_mosaic(
            mosaic, sharex=True, sharey=True, figsize=(18, 10)
        )

        for (group,), group_df in self.raw.groupby(["Well"]):
            ax = axs[group]
            ax.set_title(group_df.Group.values[0])
            _no_axes(ax)
            ax.plot(
                group_df["time_s"],
                group_df["O2 (mmHg)"],
                color="b",
                zorder=2,
                alpha=0.5,
            )
            # separate axis for pH
            ax2 = ax.twinx()
            _no_axes(ax2)
            ax2.plot(group_df["time_s"], group_df["pH"], color="r", alpha=0.5)
            # TODO perturbations
        # TODO label rows and columns https://stackoverflow.com/questions/25812255/row-and-column-headers-in-matplotlibs-subplots
        fig.suptitle(self.title)
        plt.tight_layout()
        plt.subplots_adjust(hspace=1)
        return fig

    def plot_temperature(self) -> plt.Figure:
        """Plot temperature over time."""
        df = self.raw
        df = df[["time_s", "Well Temperature", "Env. Temperature"]]
        df.groupby(["time_s", "Well Temperature", "Env. Temperature"]).agg(
            "mean"
        ).reset_index()

        fig = plt.figure()
        plt.plot(
            df["time_s"], df["Well Temperature"], label="Well Temperature"
        )
        plt.plot(
            df["time_s"], df["Env. Temperature"], label="Env. Temperature"
        )

        plt.xlabel("time [s]")
        plt.ylabel("temperature [°C]")
        plt.title(self.title)

        plt.legend()
        return fig

    def plot_summary_ocr(
        self, normalized=False, replicates=True, means=True, errorbars=True
    ) -> plt.Figure:
        """Plot experiment summary.

        Plots the mean and standard deviation of the OCR for each timepoint.
        """
        df = self.normalized_rate if normalized else self.rate
        df_grouped = df.groupby(["Measurement", "Time", "Group"]).agg(
            {"OCR": ["count", "mean", "std"], "ECAR": ["count", "mean", "std"]}
        )
        df_grouped = df_grouped.OCR.reset_index()

        import plotnine as p9

        gg = (
            p9.ggplot(df_grouped)
            + p9.aes("Time", "mean", color="factor(Group)")
            + p9.scale_x_continuous(name="time [min]")
            + p9.scale_y_continuous(
                name=f"{'normalized ' if normalized else ''}OCR [pmol/min]"
            )
            + p9.labs(colour="")
            + p9.theme_light()
            + p9.theme(
                # panel_border=element_blank(),
                panel_grid_major=p9.element_blank(),
                panel_grid_minor=p9.element_blank(),
                axis_line=p9.element_line(colour="black"),
                legend_key=p9.element_blank(),
                figure_size=(12, 6),
                text=p9.element_text(size=18),
            )
            + p9.ggtitle(self.title)
        )
        if means:
            gg += p9.geom_line(size=2)

        if errorbars:
            gg += p9.geom_errorbar(
                p9.aes(ymin="mean-std", ymax="mean+std"), width=2, size=1
            )

        if replicates:
            gg += p9.geom_line(
                p9.aes(x="Time", y="OCR", group="Well", color="factor(Group)"),
                alpha=0.3,
                data=df[
                    ~df.Well.apply(self._mwp.remove_leading_zeroes).isin(
                        self._excluded_wells
                    )
                ],
            )
            gg += p9.geom_line(
                p9.aes(x="Time", y="OCR", group="Well", color="factor(Group)"),
                alpha=0.1,
                data=df[
                    df.Well.apply(self._mwp.remove_leading_zeroes).isin(
                        self._excluded_wells
                    )
                ],
            )

        fig = gg.draw()
        self.plot_perturbations(ax=fig.axes[0], time_unit="min")
        return fig

    def plot_summary_ecar(
        self, normalized=False, replicates=True, means=True, errorbars=True
    ) -> plt.Figure:
        """Plot experiment summary.

        Plots the mean and standard deviation of the ECAR for each timepoint.
        """
        df = self.normalized_rate if normalized else self.rate
        df_grouped = df.groupby(["Measurement", "Time", "Group"]).agg(
            {"OCR": ["count", "mean", "std"], "ECAR": ["count", "mean", "std"]}
        )
        df_grouped = df_grouped.ECAR.reset_index()

        import plotnine as p9

        gg = (
            p9.ggplot(df_grouped)
            + p9.aes("Time", "mean", color="factor(Group)")
            + p9.scale_x_continuous(name="time [min]")
            + p9.scale_y_continuous(
                name=f"{'normalized ' if normalized else ''}ECAR [mpH/min]"
            )
            + p9.labs(colour="")
            + p9.theme_light()
            + p9.theme(
                # panel_border=element_blank(),
                panel_grid_major=p9.element_blank(),
                panel_grid_minor=p9.element_blank(),
                axis_line=p9.element_line(colour="black"),
                legend_key=p9.element_blank(),
                figure_size=(12, 6),
                text=p9.element_text(size=18),
            )
            + p9.ggtitle(self.title)
        )
        if means:
            gg += p9.geom_line(size=2)

        if errorbars:
            gg += p9.geom_errorbar(
                p9.aes(ymin="mean-std", ymax="mean+std"), width=2, size=1
            )

        if replicates:
            gg += p9.geom_line(
                p9.aes(
                    x="Time", y="ECAR", group="Well", color="factor(Group)"
                ),
                alpha=0.3,
                data=df[
                    ~df.Well.apply(self._mwp.remove_leading_zeroes).isin(
                        self._excluded_wells
                    )
                ],
            )
            gg += p9.geom_line(
                p9.aes(
                    x="Time", y="ECAR", group="Well", color="factor(Group)"
                ),
                alpha=0.1,
                data=df[
                    df.Well.apply(self._mwp.remove_leading_zeroes).isin(
                        self._excluded_wells
                    )
                ],
            )

        fig = gg.draw()
        self.plot_perturbations(ax=fig.axes[0], time_unit="min")
        return fig

    def plot_summary_ph(self, replicates=True, groups=None) -> plt.Figure:
        """Plot experiment summary.

        Plots the mean and standard deviation of the pH for each timepoint.
        """
        df = self.raw
        if groups:
            df = df.loc[df["Group"].isin(groups), :]
        df.loc[:, "time_min"] = df.time_s / 60
        df_grouped = (
            df.groupby(["Measurement", "time_min", "Group"])
            .agg(
                count=("Measurement", "count"),
                O2_mean=("O2 (mmHg)", "mean"),
                O2_std=("O2 (mmHg)", "std"),
                pH_mean=("pH", "mean"),
                pH_std=("pH", "std"),
            )
            .reset_index()
        )

        import plotnine as p9

        gg = (
            p9.ggplot(df_grouped)
            + p9.aes("time_min", "pH_mean", color="factor(Group)")
            + p9.geom_line(size=2)
            + p9.scale_x_continuous(name="time [min]")
            + p9.scale_y_continuous(name="pH")
            + p9.geom_errorbar(
                p9.aes(ymin="pH_mean-pH_std", ymax="pH_mean+pH_std"),
                width=1,
                size=1,
            )
            + p9.labs(colour="")
            + p9.theme_light()
            + p9.theme(
                # panel_border=element_blank(),
                panel_grid_major=p9.element_blank(),
                panel_grid_minor=p9.element_blank(),
                axis_line=p9.element_line(colour="black"),
                legend_key=p9.element_blank(),
                figure_size=(12, 6),
                text=p9.element_text(size=18),
            )
            + p9.ggtitle(self.title)
        )

        if replicates:
            gg += p9.geom_line(
                p9.aes(
                    x="time_min", y="pH", group="Well", color="factor(Group)"
                ),
                alpha=0.3,
                data=df[
                    ~df.Well.apply(self._mwp.remove_leading_zeroes).isin(
                        self._excluded_wells
                    )
                ],
            )
            gg += p9.geom_line(
                p9.aes(
                    x="time_min",
                    y="pH",
                    group="Well",
                    color="factor(Group)",
                ),
                # TODO dashed?
                alpha=0.1,
                data=df[
                    df.Well.apply(self._mwp.remove_leading_zeroes).isin(
                        self._excluded_wells
                    )
                ],
            )

        fig = gg.draw()
        self.plot_perturbations(ax=fig.axes[0], time_unit="min")
        return fig

    def plot_summary_o2(self, replicates=True) -> plt.Figure:
        """Plot experiment summary.

        Plots the mean and standard deviation of the O2 for each timepoint.
        """
        df = self.raw
        df["time_min"] = df.time_s / 60
        df_grouped = (
            df.groupby(["Measurement", "time_min", "Group"])
            .agg(
                count=("Measurement", "count"),
                O2_mean=("O2 (mmHg)", "mean"),
                O2_std=("O2 (mmHg)", "std"),
                pH_mean=("pH", "mean"),
                pH_std=("pH", "std"),
            )
            .reset_index()
        )

        import plotnine as p9

        gg = (
            p9.ggplot(df_grouped)
            + p9.aes("time_min", "O2_mean", color="factor(Group)")
            + p9.geom_line(size=2)
            + p9.scale_x_continuous(name="time [min]")
            + p9.scale_y_continuous(name="O2 (mmHg)")
            + p9.geom_errorbar(
                p9.aes(ymin="O2_mean-O2_std", ymax="O2_mean+O2_std"),
                width=1,
                size=1,
            )
            + p9.labs(colour="")
            + p9.theme_light()
            + p9.theme(
                # panel_border=element_blank(),
                panel_grid_major=p9.element_blank(),
                panel_grid_minor=p9.element_blank(),
                axis_line=p9.element_line(colour="black"),
                legend_key=p9.element_blank(),
                figure_size=(12, 6),
                text=p9.element_text(size=18),
            )
            + p9.ggtitle(self.title)
        )

        if replicates:
            gg += p9.geom_line(
                p9.aes(
                    x="time_min",
                    y="O2 (mmHg)",
                    group="Well",
                    color="factor(Group)",
                ),
                alpha=0.3,
                data=df[
                    ~df.Well.apply(self._mwp.remove_leading_zeroes).isin(
                        self._excluded_wells
                    )
                ],
            )
            gg += p9.geom_line(
                p9.aes(
                    x="time_min",
                    y="O2 (mmHg)",
                    group="Well",
                    color="factor(Group)",
                ),
                alpha=0.1,
                data=df[
                    df.Well.apply(self._mwp.remove_leading_zeroes).isin(
                        self._excluded_wells
                    )
                ],
            )

        fig = gg.draw()
        self.plot_perturbations(ax=fig.axes[0], time_unit="min")
        return fig

    def plot_to_dir(
        self, dirpath: str | Path, prefix: str = "", suffix: str = ".png"
    ):
        """Save plots to directory.

        Parameters
        ----------
        dirpath:
            Path to directory to save plots to. Will be created if it doesn't exist.
        prefix:
            Optional prefix to add to each filename.
        suffix:
            Suffix to append to each filename, including the file extension.
            Allows choosing a different file format, e.g. ".svg".
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)

        fig = self.plot_summary_ph()
        fig.savefig(dirpath / f"{prefix}summary_ph{suffix}")
        plt.close(fig)

        fig = self.plot_summary_o2()
        fig.savefig(dirpath / f"{prefix}summary_o2{suffix}")
        plt.close(fig)

        fig = self.plot_summary_ocr(normalized=False)
        fig.savefig(dirpath / f"{prefix}summary_ocr{suffix}")
        plt.close(fig)

        fig = self.plot_summary_ecar(normalized=False)
        fig.savefig(dirpath / f"{prefix}summary_ecar{suffix}")
        plt.close(fig)

        if self.has_normalization:
            fig = self.plot_summary_ecar(normalized=True)
            fig.savefig(dirpath / f"{prefix}summary_ecar_normalized{suffix}")
            plt.close(fig)

            fig = self.plot_summary_ocr(normalized=True)
            fig.savefig(dirpath / f"{prefix}summary_ocr_normalized{suffix}")
            plt.close(fig)

        self.small_multiples_rate()
        plt.savefig(dirpath / f"{prefix}small_multiples_rate{suffix}")
        plt.close(fig)

        self.small_multiples_raw()
        plt.savefig(dirpath / f"{prefix}small_multiples_raw{suffix}")
        plt.close(fig)

        self.plot_temperature()
        plt.savefig(dirpath / f"{prefix}temperature{suffix}")
        plt.close(fig)

    def aggregated_rates(self, normalized=False) -> pd.DataFrame:
        """Aggregate normalized OCR + ECAR data.

        For now, report mean+sd.
        """
        df = self.normalized_rate if normalized else self.rate
        df = df.loc[~df["Well"].isin(self._excluded_wells), :]
        df = df.groupby(["Measurement", "Time", "Group"]).agg(
            count=("Measurement", "count"),
            OCR_mean=("OCR", "mean"),
            OCR_std=("OCR", "std"),
            ECAR_mean=("ECAR", "mean"),
            ECAR_std=("ECAR", "std"),
        )
        return df.reset_index()

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


def _no_axes(ax):
    """Remove all axes from a matplotlib axes object."""
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
