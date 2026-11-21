"""integration_tests/diagnostics.py: messages of failing integration tests."""

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "integration_tests"))

import diagnostics  # noqa: E402


def frame(box, atoms):
    lines = [f"{len(atoms)}  {box}", ""]
    lines += ["\t".join([name] + values) for name, values in atoms]
    return "\n".join(lines) + "\n"


BOX = "10 10 10  90 90 90"


class FormatProcessFailureTest(unittest.TestCase):
    def test_the_status_command_and_both_streams_are_in_the_message(self):
        message = diagnostics.format_process_failure(
            ["build/apps/PQ", "run-01.in"], 1, b"info\nError: no file\n",
            b"warning\n")

        self.assertIn("exit status 1: build/apps/PQ run-01.in", message)
        self.assertIn("Error: no file", message)
        self.assertIn("warning", message)

    def test_only_the_end_of_a_long_stdout_is_shown(self):
        stdout = "\n".join(f"line {i}" for i in range(100)).encode()

        message = diagnostics.format_process_failure(
            ["pq"], 2, stdout, b"", stdout_lines=5)

        self.assertIn("line 99", message)
        self.assertNotIn("line 94\n", message)
        self.assertIn("95 earlier lines left out", message)

    def test_blank_lines_do_not_use_up_the_lines_shown(self):
        stdout = b"Error: the real reason\n" + b"\n" * 100

        message = diagnostics.format_process_failure(
            ["pq"], 1, stdout, b"", stdout_lines=5)

        self.assertIn("Error: the real reason", message)
        self.assertNotIn("left out", message)

    def test_an_empty_stream_is_marked_not_blank(self):
        message = diagnostics.format_process_failure(["pq"], 1, b"", b"")

        self.assertEqual(2, message.count("(empty)"))

    def test_undecodable_bytes_do_not_hide_the_message(self):
        message = diagnostics.format_process_failure(
            ["pq"], 1, b"Error \xff\xfe here", b"")

        self.assertIn("here", message)


class ReadFramesTest(unittest.TestCase):
    def test_frames_atoms_and_headers_are_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "a.xyz"
            path.write_text(
                frame(BOX, [("H", ["1.0", "2.0", "3.0"]),
                            ("O", ["4.0", "5.0", "6.0"])])
                + frame(BOX, [("H", ["7.0", "8.0", "9.0"]),
                              ("O", ["1.5", "2.5", "3.5"])]))

            frames = diagnostics.read_frames(path)

        self.assertEqual(2, len(frames))
        self.assertEqual(["2", "10", "10", "10", "90", "90", "90"], frames[0][0])
        self.assertEqual(("O", [1.5, 2.5, 3.5]), frames[1][1][1])


class DescribeTrajectoryDifferenceTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def describe(self, new, reference, rtol=1e-5, atol=0.0):
        (self.root / "new").write_text(new)
        (self.root / "ref").write_text(reference)

        return diagnostics.describe_trajectory_difference(
            self.root / "new", self.root / "ref", rtol, atol)

    def test_the_worst_value_is_located(self):
        reference = frame(BOX, [("H", ["1.0", "2.0", "3.0"]),
                                ("O", ["4.0", "5.0", "6.0"])])
        new = frame(BOX, [("H", ["1.0", "2.0", "3.0"]),
                          ("O", ["4.0", "5.5", "6.0"])])

        message = self.describe(new, reference)

        self.assertIn("1 of 6 values are out of tolerance", message)
        self.assertIn("frame 0, atom 1 (O), column y", message)
        self.assertIn("5.5 vs reference 5.0", message)
        self.assertIn("difference 0.5", message)

    def test_the_absolute_tolerance_is_part_of_the_comparison(self):
        reference = frame(BOX, [("H", ["-0.19247710", "0.0", "0.0"])])
        new = frame(BOX, [("H", ["-0.19247936", "0.0", "0.0"])])

        failing = self.describe(new, reference, rtol=1e-5, atol=0.0)
        passing = self.describe(new, reference, rtol=1e-5, atol=1e-4)

        self.assertIn("1 of 3 values are out of tolerance", failing)
        self.assertIn("0 of 3 values are out of tolerance", passing)
        self.assertNotIn("worst", passing)

    def test_a_reference_of_zero_needs_an_exact_value(self):
        reference = frame(BOX, [("H", ["0.0", "0.0", "0.0"])])
        new = frame(BOX, [("H", ["0.0", "1e-9", "0.0"])])

        message = self.describe(new, reference)

        self.assertIn("1 of 3 values are out of tolerance", message)
        self.assertIn("relative n/a", message)

    def test_a_box_that_differs_is_reported_with_both_headers(self):
        atoms = [("H", ["1.0", "2.0", "3.0"])]
        reference = frame("27.4421 27.4421 27.4421  90 90 90", atoms)
        new = frame("27.44205774 27.44205774 27.44205774  90 90 90", atoms)

        message = self.describe(new, reference)

        self.assertIn("0 of 3 values are out of tolerance", message)
        self.assertIn("header of frame 0: '1 27.44205774 27.44205774", message)
        self.assertIn("vs reference '1 27.4421 27.4421", message)

    def test_only_a_few_differing_headers_are_listed(self):
        atoms = [("H", ["1.0", "2.0", "3.0"])]
        reference = "".join(frame(f"10 10 10  90 90 {i}", atoms) for i in range(5))
        new = "".join(frame("10 10 10  90 90 99", atoms) for _ in range(5))

        message = self.describe(new, reference)

        self.assertEqual(3, message.count("header of frame"))
        self.assertIn("and 2 more headers differ", message)

    def test_a_different_number_of_frames_is_reported(self):
        atoms = [("H", ["1.0", "2.0", "3.0"])]

        message = self.describe(frame(BOX, atoms), frame(BOX, atoms) * 2)

        self.assertIn("number of frames: 1, reference has 2", message)

    def test_different_atoms_are_reported(self):
        reference = frame(BOX, [("H", ["1.0", "2.0", "3.0"])])
        new = frame(BOX, [("O", ["1.0", "2.0", "3.0"])])

        message = self.describe(new, reference)

        self.assertIn("frame 0, atom 0: 'O'", message)
        self.assertIn("reference 'H'", message)

    def test_a_charge_file_with_one_value_per_atom_names_the_column_by_number(self):
        reference = frame(BOX, [("K", ["0.5"])])
        new = frame(BOX, [("K", ["0.6"])])

        message = self.describe(new, reference)

        self.assertIn("column 0", message)


class DescribeTableDifferenceTest(unittest.TestCase):
    def test_the_worst_value_of_a_table_is_located(self):
        rows = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.3]]
        reference = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]

        message = diagnostics.describe_table_difference(
            "run.en", rows, reference, 1e-5, 1e-8)

        self.assertIn("run.en differs from the reference", message)
        self.assertIn("1 of 6 values are out of tolerance", message)
        self.assertIn("row 1, column 2: 3.3 vs reference 3.0", message)

    def test_a_different_shape_is_reported(self):
        self.assertIn(
            "number of rows: 1, reference has 2",
            diagnostics.describe_table_difference(
                "run.en", [[1.0]], [[1.0], [2.0]], 1e-5, 1e-8))
        self.assertIn(
            "row 0: 1 columns, reference has 2",
            diagnostics.describe_table_difference(
                "run.en", [[1.0]], [[1.0, 2.0]], 1e-5, 1e-8))


if __name__ == "__main__":
    unittest.main()
