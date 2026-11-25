import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stack_test


class LineShapeTests(unittest.TestCase):
    def test_numbers_become_hashes(self):
        self.assertEqual(
            stack_test.line_shape("O    \t  -5.9  7.6e+12  -nan 12"), "O # # # #"
        )

    def test_versions_become_vx(self):
        self.assertEqual(
            stack_test.line_shape("# PQ energy file v1.1.0+faff93f4"),
            "# PQ energy file vX",
        )

    def test_labels_with_digits_stay(self):
        self.assertEqual(
            stack_test.line_shape("|  E(TOT)  -7854.37 kcal/mol |"),
            "| E(TOT) # kcal/mol |",
        )


class OutputFormatTests(unittest.TestCase):
    def test_skips_logs_and_keeps_suffixes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "md.en").write_text("# head\n1 2.0\n2 3.0\n")
            (run / "md.log").write_text("took 3 s\n")
            (run / "md.hessian.info").write_text("modes 6\n")
            formats = stack_test.output_formats(run, "md")
        self.assertEqual(
            formats, {".en": ["# #", "# head"], ".hessian.info": ["modes #"]}
        )


class CompareFormatsTests(unittest.TestCase):
    def test_same_formats_have_no_differences(self):
        self.assertEqual(
            stack_test.compare_formats({".en": ["# #"]}, {".en": ["# #"]}), []
        )

    def test_reports_missing_new_and_changed(self):
        expected = {".en": ["# #", "# head"], ".vel": ["O # # #"]}
        actual = {".en": ["# # #", "# head"], ".mom": ["# #"]}
        self.assertEqual(
            stack_test.compare_formats(expected, actual),
            [
                ".vel: file is no longer written",
                ".mom: new file",
                ".en: line gone: # #",
                ".en: new line: # # #",
            ],
        )


if __name__ == "__main__":
    unittest.main()
