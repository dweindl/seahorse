from seahorse.multi_well import MultiWellPlate


def test_mwp():
    mwp = MultiWellPlate(nwells=96)
    assert mwp.nrows == 8
    assert mwp.ncols == 12


def test_split_well():
    mwp = MultiWellPlate(nwells=96)
    assert mwp.split_well("A1") == ("A", 1)
    assert mwp.split_well("A01") == ("A", 1)
    assert mwp.split_well("H12") == ("H", 12)


def test_is_valid():
    mwp = MultiWellPlate(nwells=96)
    assert mwp.is_valid_well("A1")
    assert mwp.is_valid_well("H12")


def test_well_range():
    mwp = MultiWellPlate(nwells=96)
    assert list(mwp.expand_range("A1:A12")) == [
        "A1",
        "A2",
        "A3",
        "A4",
        "A5",
        "A6",
        "A7",
        "A8",
        "A9",
        "A10",
        "A11",
        "A12",
    ]
    assert list(mwp.expand_range("A1:C1")) == ["A1", "B1", "C1"]
    assert list(mwp.expand_range("A1:C2")) == [
        "A1",
        "A2",
        "B1",
        "B2",
        "C1",
        "C2",
    ]


def test_iter_rows():
    mwp = MultiWellPlate(nwells=96)
    assert list(mwp.iter_rows()) == ["A", "B", "C", "D", "E", "F", "G", "H"]


def test_iter_cols():
    mwp = MultiWellPlate(nwells=96)
    assert list(mwp.iter_cols()) == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]


def test_remove_leading_zeroes():
    mwp = MultiWellPlate(nwells=96)
    assert mwp.remove_leading_zeroes("A01") == "A1"
    assert mwp.remove_leading_zeroes("A12") == "A12"
    assert mwp.remove_leading_zeroes("A001") == "A1"
    assert mwp.remove_leading_zeroes("A012") == "A12"
    assert mwp.remove_leading_zeroes("A0001") == "A1"
    assert mwp.remove_leading_zeroes("A0012") == "A12"
    assert mwp.remove_leading_zeroes("A00001") == "A1"
    assert mwp.remove_leading_zeroes("A00012") == "A12"
