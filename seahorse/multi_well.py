"""Multi-well plate related functions."""

import re
from string import ascii_uppercase


class MultiWellPlate:
    """A multi-well plate."""

    nwells_to_dim = {
        6: (2, 3),
        12: (3, 4),
        24: (4, 6),
        48: (6, 8),
        96: (8, 12),
        384: (16, 24),
        1536: (32, 48),
    }

    def __init__(
        self, nrows: int = None, ncols: int = None, nwells: int = None
    ):
        """Create a multi-well plate."""
        if nwells is not None:
            if nrows is not None or ncols is not None:
                raise ValueError("Either specify nwells or nrows and ncols")
            if nwells not in self.nwells_to_dim:
                raise ValueError(f"Invalid number of wells: {nwells}")
            self.nrows, self.ncols = self.nwells_to_dim[nwells]
        else:
            if nrows is None or ncols is None:
                raise ValueError("Either specify nwells or nrows and ncols")
            self.nrows = nrows
            self.ncols = ncols

    @property
    def nwells(self):
        """Number of wells."""
        return self.nrows * self.ncols

    def split_well(self, well: str) -> tuple[str, int]:
        """Split well into row and column"""
        match = re.match(r"([A-Z]+)([0-9]+)", well)
        if not match:
            raise ValueError(f"Invalid well: {well}")
        row = match[1]
        col = int(match[2])
        return row, col

    def is_valid_well(self, well: str, nrows=8, ncols=12) -> bool:
        row, col = self.split_well(well)
        return row in ascii_uppercase[:nrows] and 0 < col <= ncols

    def expand_range(self, well_range: str) -> str:
        """Expand a range of wells, e.g. A1:A12"""
        start_well, end_well = well_range.split(":")
        row_start, col_start = self.split_well(start_well)
        row_end, col_end = self.split_well(end_well)
        for row in ascii_uppercase[
            ascii_uppercase.index(row_start) : ascii_uppercase.index(row_end)
            + 1
        ]:
            for col in range(col_start, col_end + 1):
                yield f"{row}{col}"

    def iter_rows(self):
        """Iterate over rows."""
        yield from ascii_uppercase[: self.nrows]

    def iter_cols(self):
        """Iterate over columns."""
        yield from range(1, self.ncols + 1)

    def remove_leading_zeroes(self, well: str) -> str:
        """Remove leading zeroes from well."""
        row, col = self.split_well(well)
        return f"{row}{col}"
