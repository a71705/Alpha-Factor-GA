import unittest
from unittest.mock import patch, mock_open, MagicMock
import pandas as pd
import os
import json

# Add project root to sys.path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import utils
from config import PLATFORM_ALPHA_URL, SIMULATION_RESULTS_PATH, PNL_DATA_PATH, YEARLY_STATS_PATH

class TestUtilsFormatting(unittest.TestCase):
    def test_make_clickable_valid_id(self):
        """测试make_clickable_alpha_id能否为有效ID生成正确的HTML链接。"""
        alpha_id = "test_alpha_123"
        expected_html = f'<a href="{PLATFORM_ALPHA_URL}{alpha_id}">{alpha_id}</a>'
        self.assertEqual(utils.make_clickable_alpha_id(alpha_id), expected_html)

class TestUtilsResultProcessing(unittest.TestCase):
    def _create_mock_simulation_output(self, alpha_id, expression, fitness, book_pnl=0.1, turnover=0.2, drawdown=0.05, sharpe=1.5, returns=0.1,
                                       is_tests_data=None, pnl_data_df=None, yearly_stats_df=None):
        """Helper to create a single simulation output dictionary."""
        mock_is_stats = pd.DataFrame([{
            "alpha_id": alpha_id, "fitness": fitness, "book_pnl": book_pnl,
            "turnover": turnover, "drawdown": drawdown, "sharpe": sharpe, "returns": returns
        }])

        mock_simulate_data = {"regular": expression}

        if is_tests_data is None:
            mock_is_tests = pd.DataFrame([
                {"alpha_id": alpha_id, "name": "IS_SHARPE", "result": "PASS", "value": sharpe, "limit": 0.5},
                {"alpha_id": alpha_id, "name": "IS_FITNESS", "result": "PASS", "value": fitness, "limit": 1.0},
            ])
        else:
            mock_is_tests = pd.DataFrame(is_tests_data)
            if "alpha_id" not in mock_is_tests.columns and alpha_id:
                mock_is_tests["alpha_id"] = alpha_id


        return {
            "alpha_id": alpha_id,
            "simulate_data": mock_simulate_data,
            "is_stats": mock_is_stats,
            "is_tests": mock_is_tests,
            "pnl": pnl_data_df if pnl_data_df is not None else pd.DataFrame({'Pnl': [0.1,0.2], 'alpha_id': [alpha_id, alpha_id]}), # Ensure pnl_data_df has alpha_id
            "stats": yearly_stats_df if yearly_stats_df is not None else pd.DataFrame({'Sharpe': [1.5], 'alpha_id': [alpha_id]}) # Ensure yearly_stats_df has alpha_id
        }

    def test_prettify_basic_valid_data(self):
        """测试prettify_result处理基本有效数据。"""
        results = [
            self._create_mock_simulation_output("alpha1", "close - open", 1.5),
            self._create_mock_simulation_output("alpha2", "rank(vwap)", 2.0)
        ]
        df = utils.prettify_result(results)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertIn("alpha_id", df.columns)
        self.assertIn("expression", df.columns)
        self.assertIn("is_sharpe", df.columns) # Test pivot
        self.assertEqual(df.iloc[0]['alpha_id'], "alpha2") # Should be sorted by fitness

    def test_prettify_empty_result(self):
        """测试prettify_result处理空输入列表。"""
        df = utils.prettify_result([])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

    def test_prettify_missing_data_for_alpha(self):
        """测试prettify_result处理部分数据缺失的情况。"""
        results = [
            self._create_mock_simulation_output("alpha1", "close - open", 1.5),
            {"alpha_id": "alpha2", "simulate_data": {"regular":"rank(vwap)"}, "is_stats": None, "is_tests": None} # Missing stats and tests
        ]
        df = utils.prettify_result(results)
        self.assertEqual(len(df), 2) # Should still include alpha2, possibly with NaNs
        self.assertTrue(pd.isna(df[df['alpha_id'] == 'alpha2']['is_sharpe'].iloc[0]))


    def test_concat_pnl_basic(self):
        """测试concat_pnl的基本功能。"""
        pnl_df1 = pd.DataFrame({'Pnl': [0.1, 0.2], 'Date': pd.to_datetime(['2023-01-01', '2023-01-02'])})
        pnl_df2 = pd.DataFrame({'Pnl': [0.3, 0.4], 'Date': pd.to_datetime(['2023-01-01', '2023-01-02'])})
        results = [
            self._create_mock_simulation_output("alpha1", "expr1", 1.0, pnl_data_df=pnl_df1.copy()),
            self._create_mock_simulation_output("alpha2", "expr2", 2.0, pnl_data_df=pnl_df2.copy())
        ]
        # Ensure pnl_data_df in _create_mock_simulation_output adds alpha_id if not present
        # For this test, let's assume it's added or the function handles it.
        # The default mock pnl_data_df in helper already has alpha_id.

        concatenated_pnl = utils.concat_pnl(results)
        self.assertIsInstance(concatenated_pnl, pd.DataFrame)
        self.assertEqual(len(concatenated_pnl), 4) # 2 rows from each alpha
        self.assertIn("alpha_id", concatenated_pnl.columns)
        self.assertEqual(len(concatenated_pnl[concatenated_pnl['alpha_id'] == 'alpha1']), 2)
        self.assertEqual(len(concatenated_pnl[concatenated_pnl['alpha_id'] == 'alpha2']), 2)


    def test_concat_is_tests_basic(self):
        """测试concat_is_tests的基本功能。"""
        tests1 = [{"name": "IS_SHARPE", "result": "PASS"}, {"name": "IS_TURNOVER", "result": "FAIL"}]
        tests2 = [{"name": "IS_SHARPE", "result": "FAIL"}, {"name": "IS_FITNESS", "result": "PASS"}]
        results = [
            self._create_mock_simulation_output("alpha1", "expr1", 1.0, is_tests_data=[{"alpha_id": "alpha1", **t} for t in tests1]),
            self._create_mock_simulation_output("alpha2", "expr2", 2.0, is_tests_data=[{"alpha_id": "alpha2", **t} for t in tests2])
        ]
        concatenated_tests = utils.concat_is_tests(results)
        self.assertIsInstance(concatenated_tests, pd.DataFrame)
        self.assertEqual(len(concatenated_tests), 4)
        self.assertIn("alpha_id", concatenated_tests.columns)


class TestUtilsFileSaving(unittest.TestCase):
    @patch("os.makedirs")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_save_simulation_result(self, mock_json_dump, mock_file_open, mock_os_makedirs):
        """测试save_simulation_result是否正确调用文件操作。"""
        result_data = {"id": "alpha_test", "settings": {"region": "USA"}, "data": "test_data"}
        utils.save_simulation_result(result_data)

        mock_os_makedirs.assert_called_once_with(SIMULATION_RESULTS_PATH, exist_ok=True)
        expected_filepath = os.path.join(SIMULATION_RESULTS_PATH, "alpha_test_USA.json")
        mock_file_open.assert_called_once_with(expected_filepath, "w", encoding="utf-8")
        mock_json_dump.assert_called_once_with(result_data, mock_file_open(), indent=4, ensure_ascii=False)

    @patch("os.makedirs")
    @patch.object(pd.DataFrame, "to_csv")
    def test_save_pnl_calls_to_csv(self, mock_to_csv, mock_os_makedirs):
        """测试save_pnl是否正确调用DataFrame.to_csv。"""
        pnl_df = pd.DataFrame({"Pnl": [0.1, 0.2]})
        alpha_id = "alpha_test_pnl"
        region = "USA"

        utils.save_pnl(pnl_df, alpha_id, region)

        mock_os_makedirs.assert_called_once_with(PNL_DATA_PATH, exist_ok=True)
        expected_filepath = os.path.join(PNL_DATA_PATH, f"{alpha_id}_{region}.csv")
        mock_to_csv.assert_called_once_with(expected_filepath, index=False)

    @patch("os.makedirs")
    @patch.object(pd.DataFrame, "to_csv")
    def test_save_yearly_stats_calls_to_csv(self, mock_to_csv, mock_os_makedirs):
        """测试save_yearly_stats是否正确调用DataFrame.to_csv。"""
        stats_df = pd.DataFrame({"Sharpe": [1.5, 1.8]})
        alpha_id = "alpha_test_stats"
        region = "EUR"

        utils.save_yearly_stats(stats_df, alpha_id, region)

        mock_os_makedirs.assert_called_once_with(YEARLY_STATS_PATH, exist_ok=True)
        expected_filepath = os.path.join(YEARLY_STATS_PATH, f"{alpha_id}_{region}.csv")
        mock_to_csv.assert_called_once_with(expected_filepath, index=False)

    def test_save_pnl_handles_none_or_empty_df(self):
        """测试save_pnl对None或空DataFrame的处理。"""
        with patch("os.makedirs") as mock_os_makedirs, \
             patch.object(pd.DataFrame, "to_csv") as mock_to_csv:

            utils.save_pnl(None, "alpha_none", "USA")
            mock_os_makedirs.assert_not_called()
            mock_to_csv.assert_not_called()

            utils.save_pnl(pd.DataFrame(), "alpha_empty", "USA")
            mock_os_makedirs.assert_not_called()
            mock_to_csv.assert_not_called()


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

# TODO: Add more tests for prettify_result edge cases (e.g. missing 'is_stats', 'is_tests' keys entirely)
# TODO: Test file saving functions for error handling (e.g., if os.makedirs fails, or write fails) - requires more complex mock setup
# TODO: Test type validation in save_pnl, save_yearly_stats, save_simulation_result more directly.
