from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from calc_catch import _write_excel_output


class EmptyROIExportTests(unittest.TestCase):
    def test_empty_roi_export_retains_headers_and_time_axis(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'empty.xlsx'
            times = np.arange(6)/2
            _write_excel_output(path, [], np.empty((6, 0)), times)
            rows = pd.read_excel(path, sheet_name='ROI_Data')
            self.assertEqual(list(rows.columns), [
                'x_Centre', 'y_Centre', 'Mean_Intensity', 'Std_Intensity'])
            self.assertEqual(len(rows), 0)
            traces = pd.read_excel(path, sheet_name='Time_Series')
            self.assertEqual(list(traces.columns), ['Time'])
            np.testing.assert_array_equal(traces['Time'], times)


if __name__ == '__main__':
    unittest.main()
