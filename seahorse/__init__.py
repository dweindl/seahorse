from pathlib import Path

import pandas as pd

from .C import *
from .experiment import SeahorseExperiment


def ods_to_tsv(file: Path | str, outdir: Path | str):
    """
    Convert ods file to tsv files for each sheet.

    Loading ``.ods`` files with pandas is pretty slow...
    """
    outdir = Path(outdir)
    file = Path(file)

    outdir.mkdir(exist_ok=True)

    df_all = pd.read_excel(file, sheet_name=None)

    for sheet_name, df in df_all.items():
        # skip useless sheets
        if sheet_name == "styles":
            continue

        df.to_csv(outdir / f"{sheet_name}.tsv", sep="\t", index=False)
