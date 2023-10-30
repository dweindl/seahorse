"""Seahorse measurements to PEtab measurement file format

https://petab.readthedocs.io/
"""
import pandas as pd


def to_petab(
    df: pd.DataFrame,
    ocr_observable_id: str | None = "OCR",
    ecar_observable_id: str | None = "ECAR",
    exclude_groups=None,
) -> pd.DataFrame:
    """Convert aggregated Seahorse data to PEtab format.

    Parameters
    ----------
    df:
        DataFrame with rates or normalized rates (-> ``Experiment.aggregated_rates``)
    ocr_observable_id:
        PEtab observable ID to be used for OCR measurements. ``None`` to exclude OCR measurements.
    ecar_observable_id:
        PEtab observable ID to be used for ECAR measurements. ``None`` to exclude ECAR measurements.
    exclude_groups:
        Sequence of experiment group names to exclude from the output, e.g. ``['Background']``.

    Returns
    -------
    DataFrame in PEtab measurement table format.
    Measurement units depend on the input data, no changes are made.
    Time unit is the same as for ``Time`` in the input data (minutes).
    """
    if exclude_groups is None:
        exclude_groups = []

    from petab.C import (
        MEASUREMENT,
        NOISE_PARAMETERS,
        OBSERVABLE_ID,
        SIMULATION_CONDITION_ID,
        TIME,
    )

    # wide to long format
    if ocr_observable_id is None:
        df_ocr = pd.DataFrame()
    else:
        df_ocr = df[["Time", "Group", "OCR_mean", "OCR_std"]][
            ~df.Group.isin(exclude_groups)
        ].rename(
            columns={"OCR_mean": MEASUREMENT, "OCR_std": NOISE_PARAMETERS}
        )
        df_ocr[OBSERVABLE_ID] = ocr_observable_id

    if ecar_observable_id is None:
        df_ecar = pd.DataFrame()
    else:
        df_ecar = df[["Time", "Group", "ECAR_mean", "ECAR_std"]][
            ~df.Group.isin(exclude_groups)
        ].rename(
            columns={"ECAR_mean": MEASUREMENT, "ECAR_std": NOISE_PARAMETERS}
        )
        df_ecar[OBSERVABLE_ID] = ecar_observable_id

    df_petab = pd.concat([df_ocr, df_ecar])
    df_petab = df_petab.rename(
        columns={"Time": TIME, "Group": SIMULATION_CONDITION_ID}
    )
    # reorder columns for readability
    df_petab = df_petab[
        [
            OBSERVABLE_ID,
            SIMULATION_CONDITION_ID,
            TIME,
            MEASUREMENT,
            NOISE_PARAMETERS,
        ]
    ]
    return df_petab
