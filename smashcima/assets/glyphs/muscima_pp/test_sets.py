"""MUSCIMA++ v2.0 test sets (specifications/testset-*.txt). Glyphs from these
pages are left out of the symbol repository, so that models trained on
Smashcima output can be evaluated on them."""

WRITER_INDEPENDENT_TEST_SET = frozenset([
    "CVC-MUSCIMA_W-12_N-04_D-ideal",
    "CVC-MUSCIMA_W-12_N-11_D-ideal",
    "CVC-MUSCIMA_W-12_N-19_D-ideal",
    "CVC-MUSCIMA_W-13_N-02_D-ideal",
    "CVC-MUSCIMA_W-13_N-03_D-ideal",
    "CVC-MUSCIMA_W-13_N-16_D-ideal",
    "CVC-MUSCIMA_W-15_N-10_D-ideal",
    "CVC-MUSCIMA_W-15_N-14_D-ideal",
    "CVC-MUSCIMA_W-15_N-15_D-ideal",
    "CVC-MUSCIMA_W-28_N-05_D-ideal",
    "CVC-MUSCIMA_W-28_N-08_D-ideal",
    "CVC-MUSCIMA_W-28_N-09_D-ideal",
    "CVC-MUSCIMA_W-30_N-06_D-ideal",
    "CVC-MUSCIMA_W-30_N-13_D-ideal",
    "CVC-MUSCIMA_W-30_N-17_D-ideal",
    "CVC-MUSCIMA_W-31_N-01_D-ideal",
    "CVC-MUSCIMA_W-31_N-07_D-ideal",
    "CVC-MUSCIMA_W-31_N-18_D-ideal",
    "CVC-MUSCIMA_W-39_N-12_D-ideal",
    "CVC-MUSCIMA_W-39_N-20_D-ideal",
])

WRITER_DEPENDENT_TEST_SET = frozenset([
    "CVC-MUSCIMA_W-03_N-01_D-ideal",
    "CVC-MUSCIMA_W-04_N-20_D-ideal",
    "CVC-MUSCIMA_W-05_N-11_D-ideal",
    "CVC-MUSCIMA_W-09_N-06_D-ideal",
    "CVC-MUSCIMA_W-19_N-19_D-ideal",
    "CVC-MUSCIMA_W-20_N-03_D-ideal",
    "CVC-MUSCIMA_W-23_N-13_D-ideal",
    "CVC-MUSCIMA_W-24_N-07_D-ideal",
    "CVC-MUSCIMA_W-26_N-04_D-ideal",
    "CVC-MUSCIMA_W-29_N-14_D-ideal",
    "CVC-MUSCIMA_W-32_N-09_D-ideal",
    "CVC-MUSCIMA_W-34_N-16_D-ideal",
    "CVC-MUSCIMA_W-35_N-08_D-ideal",
    "CVC-MUSCIMA_W-37_N-17_D-ideal",
    "CVC-MUSCIMA_W-41_N-02_D-ideal",
    "CVC-MUSCIMA_W-43_N-10_D-ideal",
    "CVC-MUSCIMA_W-45_N-18_D-ideal",
    "CVC-MUSCIMA_W-47_N-12_D-ideal",
    "CVC-MUSCIMA_W-49_N-05_D-ideal",
    "CVC-MUSCIMA_W-50_N-15_D-ideal",
])

TEST_DOCUMENTS = WRITER_INDEPENDENT_TEST_SET | WRITER_DEPENDENT_TEST_SET
