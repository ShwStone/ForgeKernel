import csv
import tempfile
from pathlib import Path
import unittest

from probe_utils import single_task_profile_summary


class ProfileSummaryTests(unittest.TestCase):
    def summarize(self, durations, calls, names=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'kernel_details.csv'
            with path.open('w', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=['Name', 'Type', 'Block Num', 'Duration(us)'])
                writer.writeheader()
                for i, duration in enumerate(durations):
                    writer.writerow({'Name': names[i] if names else 'probe', 'Type': 'N/A',
                                     'Block Num': 28, 'Duration(us)': duration})
            return single_task_profile_summary(path, calls)

    def test_one_task_per_call(self):
        result = self.summarize([3, 1, 2], 3)
        self.assertEqual(result['median_us'], 2)
        self.assertEqual(result['single_task_duration_us'], [3, 1, 2])

    def test_partial_capture_rejected(self):
        with self.assertRaises(ValueError):
            self.summarize([1, 2], 3)

    def test_multiple_tasks_not_silently_summed(self):
        with self.assertRaises(ValueError):
            self.summarize([1, 2, 3, 4], 2)

    def test_role_change_rejected(self):
        with self.assertRaises(ValueError):
            self.summarize([1, 2], 2, ['probe', 'other'])

    def test_empty_capture_rejected(self):
        with self.assertRaises(ValueError):
            self.summarize([], 0)

    def test_nonpositive_duration_rejected(self):
        for duration in (0, -1):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.summarize([duration], 1)

    def test_nonfinite_duration_rejected(self):
        for duration in ('nan', 'inf', '-inf'):
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                self.summarize([duration], 1)


if __name__ == '__main__':
    unittest.main()
