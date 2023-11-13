from pathlib import Path

import pandas as pd

from .C import *
from .experiment import SeahorseExperiment


def ods_to_tsv(file: Path | str, outdir: Path | str):
    """
    Convert spreadsheet file to tsv files for each sheet.

    Loading ``.ods`` files with pandas is, at least in some cases, super slow.
    """
    outdir = Path(outdir)
    file = Path(file)

    outdir.mkdir(exist_ok=True)

    from importlib.metadata import version

    from pandas.io.excel._base import inspect_excel_format

    # because openpyxl doesn't some files properly with read_only=True
    # (requires https://github.com/pandas-dev/pandas/pull/55807 - remove once pandas 2.2.0 is released)
    if inspect_excel_format(file) == "xlsx" and tuple(
        map(int, version("pandas").split(".")[:3])
    ) >= (2, 2, 0):
        df_all = pd.read_excel(
            file,
            sheet_name=None,
            engine="openpyxl",
            engine_kwargs={"read_only": False},
        )
    else:
        df_all = pd.read_excel(file, sheet_name=None)

    for sheet_name, df in df_all.items():
        # skip useless sheets
        if sheet_name == "styles":
            continue
        # e.g. if there is no normalization, normalized data sheets are still present, but empty and hidden
        if df.empty:
            continue

        df.to_csv(outdir / f"{sheet_name}.tsv", sep="\t", index=False)
