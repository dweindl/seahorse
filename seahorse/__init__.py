from pathlib import Path

import pandas as pd

from .C import *  # noqa 403
from .experiment import SeahorseExperiment as SeahorseExperiment


def spreadsheet_to_tsv(file: Path | str, outdir: Path | str):
    """
    Convert spreadsheet file to tsv files for each sheet.
    """
    outdir = Path(outdir)
    file = Path(file)

    outdir.mkdir(exist_ok=True)

    df_all = pd.read_excel(
        file,
        sheet_name=None,
        engine="calamine",
    )

    for sheet_name, df in df_all.items():
        # skip useless sheets
        if sheet_name == "styles":
            continue
        # e.g., if there is no normalization,
        #  normalized data sheets are still present, but empty and hidden
        if df.empty:
            continue

        df.to_csv(outdir / f"{sheet_name}.tsv", sep="\t", index=False)
