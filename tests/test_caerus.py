"""
Test suite for the caerus package.

This module contains comprehensive tests for all major functionalities
of the caerus package including initialization, fitting, gridsearch,
plotting, and error handling.
"""

import matplotlib
# non-GUI backend for testing. Import for pytest!
matplotlib.use("Agg")

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from unittest.mock import patch, MagicMock
import os
import tempfile
import shutil

# Import the package to test
from caerus import caerus
try:
    import caerus.helper as helper
except:
    import helper


class TestCaerusInitialization:
    """Test caerus class initialization and parameter setting."""
    
    def test_default_initialization(self):
        """Test caerus initialization with default parameters."""
        cs = caerus()
        assert cs.window == 50
        assert cs.minperc == 3
        assert cs.nlargest == 10
        assert cs.threshold == 0.25
        assert cs.extb == 0
        assert cs.extf == 10
    
    def test_custom_initialization(self):
        """Test caerus initialization with custom parameters."""
        cs = caerus(window=100, minperc=5, nlargest=15, threshold=0.5, extb=2, extf=20)
        assert cs.window == 100
        assert cs.minperc == 5
        assert cs.nlargest == 15
        assert cs.threshold == 0.5
        assert cs.extb == 2
        assert cs.extf == 20
    
    def test_parameter_types(self):
        """Test that parameters are properly typed."""
        cs = caerus()
        assert isinstance(cs.window, int)
        assert isinstance(cs.minperc, (int, float))
        assert isinstance(cs.nlargest, (int, float))
        assert isinstance(cs.threshold, float)
        assert isinstance(cs.extb, int)
        assert isinstance(cs.extf, int)


class TestCaerusInputValidation:
    """Test input validation and error handling."""
    
    def test_fit_with_valid_1d_array(self):
        """Test fit method with valid 1D numpy array."""
        cs = caerus()
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        result = cs.fit(X)
        assert isinstance(result, dict)
        assert 'X' in result
        assert 'simmat' in result
        assert 'loc_start' in result
        assert 'loc_stop' in result
        assert 'loc_start_best' in result
        assert 'loc_stop_best' in result
        assert 'agg' in result
        assert 'df' in result
    
    def test_fit_with_valid_list(self):
        """Test fit method with valid list."""
        cs = caerus()
        X = [1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0]
        result = cs.fit(X)
        assert isinstance(result, dict)
    
    def test_fit_with_valid_series(self):
        """Test fit method with valid pandas Series."""
        cs = caerus()
        X = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        result = cs.fit(X)
        assert isinstance(result, dict)
    
    def test_fit_with_invalid_dataframe(self):
        """Test fit method with invalid 2D dataframe."""
        cs = caerus()
        X = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        with pytest.raises(Exception):
            cs.fit(X)
    
    def test_fit_with_empty_array(self):
        """Test fit method with empty array."""
        cs = caerus()
        X = np.array([])
        cs.fit(X)
        
    def test_window_larger_than_data(self):
        """Test behavior when window size is larger than data length."""
        cs = caerus(window=1000)
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        # Should not raise error, should adjust window size
        result = cs.fit(X)
        assert isinstance(result, dict)


class TestCaerusCoreFunctionality:
    """Test core functionality of the caerus package."""
    
    def test_fit_with_simple_data(self):
        """Test fit method with simple synthetic data."""
        cs = caerus(window=3, minperc=1, threshold=0.1)
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        result = cs.fit(X)
        
        assert 'X' in result
        assert 'simmat' in result
        assert 'loc_start' in result
        assert 'loc_stop' in result
        assert 'agg' in result
        assert 'df' in result
        
        # Check that results are stored in the object
        assert hasattr(cs, 'results')
        assert cs.results == result
    
    def test_fit_with_parameter_override(self):
        """Test fit method with parameter overrides."""
        cs = caerus(window=50, minperc=3, threshold=0.25)
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        
        # Override parameters during fit
        result = cs.fit(X, window=5, minperc=1, threshold=0.1)
        
        # Check that parameters were overridden
        assert cs.window == 5
        assert cs.minperc == 1
        assert cs.threshold == 0.1
    
    def test_fit_return_as_dict(self):
        """Test fit method with return_as_dict=False."""
        cs = caerus()
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        
        # Should return None when return_as_dict=False
        result = cs.fit(X, return_as_dict=False)
        assert result is None
        
        # But results should still be stored in the object
        assert hasattr(cs, 'results')
        assert isinstance(cs.results, dict)
    
    def test_logger_status_and_verbose(self):
        """Test logger status and verbose functionality."""
        import logging
        
        # Test 1: Default logger status
        cs1 = caerus()
        assert cs1.verbose == 'info'
        # Check that the logger level is set correctly
        current_level = helper.get_logger()
        # 'info' should correspond to logging.INFO (20)
        assert current_level == logging.INFO
        
        # Test 2: Silent logger status
        cs2 = caerus(verbose='silent')
        assert cs2.verbose == 'silent'
        # Check that the logger level is set correctly
        current_level = helper.get_logger()
        # 'silent' should correspond to logging.CRITICAL + 10 (50)
        assert current_level == logging.CRITICAL + 10
        
        # Test 3: Warning logger status
        cs3 = caerus(verbose='warning')
        assert cs3.verbose == 'warning'
        # Check that the logger level is set correctly
        current_level = helper.get_logger()
        # 'warning' should correspond to logging.WARNING (30)
        assert current_level == logging.WARNING
        
        # Test 4: Debug logger status
        cs4 = caerus(verbose='debug')
        assert cs4.verbose == 'debug'
        # Check that the logger level is set correctly
        current_level = helper.get_logger()
        # 'debug' should correspond to logging.DEBUG (10)
        assert current_level == logging.DEBUG
        
        # Test 5: Changing logger status after creation
        cs5 = caerus(verbose='info')
        initial_level = helper.get_logger()
        assert initial_level == logging.INFO
        
        # Change to silent
        cs5.set_verbose('silent')
        new_level = helper.get_logger()
        assert new_level == logging.CRITICAL + 10
        
        # Change back to info
        cs5.set_verbose('info')
        final_level = helper.get_logger()
        assert final_level == logging.INFO
        
        # Test 6: Logger level consistency across instances
        cs6 = caerus(verbose='critical')
        cs7 = caerus(verbose='error')
        
        # Each instance should maintain its own logger level
        assert cs6.verbose == 'critical'
        assert cs7.verbose == 'error'
        
        # The global logger should reflect the last instance created
        current_level = helper.get_logger()
        assert current_level == logging.ERROR  # 'error' corresponds to logging.ERROR (40)
    
    def test_fit_method_logger_behavior(self):
        """Test that the fit method properly handles logger status changes."""
        import logging
        
        # Create instance with info level
        cs = caerus(verbose='info')
        initial_level = helper.get_logger()
        assert initial_level == logging.INFO
        
        # Test data
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        
        # Test 1: Fit without verbose override (should use instance verbose)
        result = cs.fit(X)
        assert isinstance(result, dict)
        # Logger should still be at info level after fit
        current_level = helper.get_logger()
        assert current_level == logging.INFO
        
        # Test 2: Fit with verbose override to silent
        result = cs.fit(X, verbose='silent')
        assert isinstance(result, dict)
        # Logger should still be at info level after fit (not permanently changed)
        current_level = helper.get_logger()
        assert current_level == logging.INFO
        
        # Test 3: Fit with verbose override to debug
        result = cs.fit(X, verbose='debug')
        assert isinstance(result, dict)
        # Logger should still be at info level after fit (not permanently changed)
        current_level = helper.get_logger()
        assert current_level == logging.INFO
        
        # Test 4: Change instance verbose and verify fit uses it
        cs.set_verbose('warning')
        assert cs.verbose == 'warning'
        current_level = helper.get_logger()
        assert current_level == logging.WARNING
        
        # Fit should now use warning level
        result = cs.fit(X)
        assert isinstance(result, dict)
        # Logger should still be at warning level after fit
        current_level = helper.get_logger()
        assert current_level == logging.WARNING
    

class TestCaerusGridsearch:
    """Test gridsearch functionality."""
    
    def test_gridsearch_basic(self):
        """Test basic gridsearch functionality."""
        cs = caerus()
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0])
        
        # Test with default parameters
        result = cs.gridsearch(X)
        assert result is None  # Should return None when return_as_dict=False
        
        # Check that results are stored in the object
        assert hasattr(cs, 'results_gridsearch')
        assert isinstance(cs.results_gridsearch, dict)
        assert 'balances' in cs.results_gridsearch
        assert 'trades' in cs.results_gridsearch
        assert 'window' in cs.results_gridsearch
        assert 'minperc' in cs.results_gridsearch
    
    def test_gridsearch_custom_parameters(self):
        """Test gridsearch with custom parameters."""
        cs = caerus()
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0])
        
        windows = np.array([3, 5])
        minpercs = np.array([1, 2])
        
        result = cs.gridsearch(X, window=windows, minperc=minpercs)
        
        # Check that results have correct shapes
        assert cs.results_gridsearch['balances'].shape == (len(minpercs), len(windows))
        assert cs.results_gridsearch['trades'].shape == (len(minpercs), len(windows))
        assert np.array_equal(cs.results_gridsearch['window'], windows)
        assert np.array_equal(cs.results_gridsearch['minperc'], minpercs)
    
    def test_gridsearch_return_as_dict(self):
        """Test gridsearch with return_as_dict=True."""
        cs = caerus()
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0])
        
        result = cs.gridsearch(X, return_as_dict=True)
        assert isinstance(result, dict)
        assert 'balances' in result
        assert 'trades' in result
        assert 'window' in result
        assert 'minperc' in result
    

class TestCaerusPlotting:
    """Test plotting functionality."""
    
    def test_plot_with_results(self):
        """Test plot method with fit results."""
        cs = caerus()
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        cs.fit(X)
        # Should not raise error
        cs.plot()
        # Clean up
        plt.close('all')
    
    def test_plot_with_gridsearch_results(self):
        """Test plot method with gridsearch results."""
        cs = caerus()
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0])
        cs.gridsearch(X)
        cs.plot()
        # Clean up
        plt.close('all')
    
    def test_plot_with_custom_figsize(self):
        """Test plot method with custom figure size."""
        cs = caerus()
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        cs.fit(X)
        # Test with custom figsize
        cs.plot(figsize=(10, 8))
        # Clean up
        plt.close('all')
    
    def test_plot_without_results(self):
        """Test plot method without any results."""
        cs = caerus()
        # Should not raise error, just do nothing
        cs.plot()
        # Clean up
        plt.close('all')

class TestCaerusDataDownload:
    """Test data download functionality."""
    
    @patch('caerus.caerus.requests.get')
    @patch('caerus.caerus.os.path.isfile')
    @patch('caerus.caerus.pd.read_csv')
    def test_download_example_btc(self, mock_read_csv, mock_isfile, mock_get):
        """Test downloading BTC example dataset."""
        # Mock the file doesn't exist initially
        mock_isfile.return_value = False
        
        # Mock the download response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'fake_data']
        mock_get.return_value = mock_response
        
        # Mock the CSV reading
        mock_df = pd.DataFrame({'Close': [100, 200, 300, 250, 200, 150, 200, 250, 300, 350]})
        mock_read_csv.return_value = mock_df
        
        cs = caerus()
        X = cs.download_example(name='btc')
        
        # Check that the result is a numpy array
        assert isinstance(X, np.ndarray)
        assert len(X) == 10
        assert X[0] == 100
        assert X[-1] == 350
    
    @patch('caerus.caerus.requests.get')
    @patch('caerus.caerus.os.path.isfile')
    @patch('caerus.caerus.pd.read_csv')
    def test_download_example_facebook(self, mock_read_csv, mock_isfile, mock_get):
        """Test downloading Facebook example dataset."""
        # Mock the file doesn't exist initially
        mock_isfile.return_value = False
        
        # Mock the download response
        mock_response = MagicMock()
        mock_response.iter_content.return_value = [b'fake_data']
        mock_get.return_value = mock_response
        
        # Mock the CSV reading
        mock_df = pd.DataFrame({'close': [50, 60, 70, 65, 60, 55, 60, 65, 70, 75]})
        mock_read_csv.return_value = mock_df
        
        cs = caerus()
        X = cs.download_example(name='facebook')
        
        # Check that the result is a numpy array
        assert isinstance(X, np.ndarray)
        assert len(X) == 10
        assert X[0] == 50
        assert X[-1] == 75
    
    @patch('caerus.caerus.os.path.isfile')
    @patch('caerus.caerus.pd.read_csv')
    def test_download_example_file_exists(self, mock_read_csv, mock_isfile):
        """Test downloading example dataset when file already exists."""
        # Mock the file exists
        mock_isfile.return_value = True
        
        # Mock the CSV reading
        mock_df = pd.DataFrame({'Close': [100, 200, 300, 250, 200, 150, 200, 250, 300, 350]})
        mock_read_csv.return_value = mock_df
        
        cs = caerus()
        X = cs.download_example(name='btc')
        
        # Check that the result is a numpy array
        assert isinstance(X, np.ndarray)
        assert len(X) == 10


class TestCaerusHelperFunctions:
    """Test helper functions."""
    
    def test_check_input_valid_types(self):
        """Test _check_input with valid input types."""
        # Test with numpy array
        X_np = np.array([1, 2, 3, 4, 5])
        result_np = helper._check_input(X_np)
        assert isinstance(result_np, pd.Series)
        assert len(result_np) == 5
        
        # Test with list
        X_list = [1, 2, 3, 4, 5]
        result_list = helper._check_input(X_list)
        assert isinstance(result_list, pd.Series)
        assert len(result_list) == 5
        
        # Test with pandas Series
        X_series = pd.Series([1, 2, 3, 4, 5])
        result_series = helper._check_input(X_series)
        assert isinstance(result_series, pd.Series)
        assert len(result_series) == 5
    
    def test_check_input_invalid_dataframe(self):
        """Test _check_input with invalid 2D dataframe."""
        X_df = pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
        with pytest.raises(Exception):
            helper._check_input(X_df)
    
    def test_ones2region(self):
        """Test ones2region function."""
        data = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0])
        result = helper.ones2region(data)
        
        # Should return list of tuples with start-stop indices
        assert isinstance(result, list)
        assert len(result) == 3  # Three regions of ones
        
        # Check first region
        assert result[0] == (1, 2)  # indices 1-2
        # Check second region
        assert result[1] == (4, 4)  # index 4
        # Check third region
        assert result[2] == (6, 8)  # indices 6-8
    
    def test_region2ones(self):
        """Test region2ones function."""
        regions = [(1, 3), (5, 7), (9, 9)]
        result = helper.region2ones(regions)
        
        # Should return array with ones in specified regions
        assert isinstance(result, np.ndarray)
        assert result[1:4].sum() == 3  # indices 1-3 should be ones
        assert result[5:8].sum() == 3  # indices 5-7 should be ones
        assert result[9] == 1  # index 9 should be one
        assert result[0] == 0  # index 0 should be zero
        assert result[4] == 0  # index 4 should be zero
        assert result[8] == 0  # index 8 should be zero
    
    def test_idx2region(self):
        """Test idx2region function."""
        indices = [1, 2, 3, 5, 6, 7, 9]
        result = helper.idx2region(indices)
        
        # Should return list of tuples with start-stop indices
        assert isinstance(result, list)
        assert len(result) == 3  # Three regions
        
        # Check regions
        assert result[0] == (1, 3)  # indices 1-3
        assert result[1] == (5, 7)  # indices 5-7
        assert result[2] == (9, 9)  # index 9


class TestCaerusEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_fit_with_no_detected_regions(self):
        """Test fit method when no regions are detected."""
        cs = caerus(window=5, minperc=50, threshold=0.9)  # Very strict parameters
        X = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])  # Flat data
        
        result = cs.fit(X)
        
        # Should still return a result dict
        assert isinstance(result, dict)
        assert result['loc_start'] is None
        assert result['loc_stop'] is None
        assert result['loc_start_best'] is None
        assert result['loc_stop_best'] is None
    
    def test_fit_with_very_small_data(self):
        """Test fit method with very small dataset."""
        cs = caerus(window=2, minperc=1, threshold=0.1)
        X = np.array([1.0, 2.0, 1.0])
        result = cs.fit(X)
        assert isinstance(result, dict)
    
    def test_gridsearch_with_single_values(self):
        """Test gridsearch with single window and minperc values."""
        cs = caerus()
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 2.0])
        
        result = cs.gridsearch(X, window=np.array([5]), minperc=np.array([1]))
        
        # Should work with single values
        assert hasattr(cs, 'results_gridsearch')
        assert cs.results_gridsearch['balances'].shape == (1, 1)
        assert cs.results_gridsearch['trades'].shape == (1, 1)
    
    def test_parameter_validation(self):
        """Test parameter validation and bounds."""
        # Test with extreme values
        cs = caerus(window=1, minperc=0.1, threshold=0.01, nlargest=1)
        
        # Should not raise errors during initialization
        assert cs.window == 1
        assert cs.minperc == 0.1
        assert cs.threshold == 0.01
        assert cs.nlargest == 1


class TestCaerusIntegration:
    """Test integration scenarios and real-world usage patterns."""
    
    def test_complete_workflow(self):
        """Test complete workflow from initialization to plotting."""
        # Initialize
        cs = caerus(window=10, minperc=2, threshold=0.2)
        
        # Create synthetic data with clear patterns
        np.random.seed(42)
        X = np.cumsum(np.random.randn(100)) + 100
        
        # Fit the model
        result = cs.fit(X)
        assert isinstance(result, dict)
        assert hasattr(cs, 'results')
        
        # Run gridsearch
        cs.gridsearch(X, window=np.array([5, 10]), minperc=np.array([1, 2]))
        assert hasattr(cs, 'results_gridsearch')
        
        # Plot results
        cs.plot()
        plt.close('all')
    
    def test_multiple_fits_same_instance(self):
        """Test multiple fits on the same caerus instance."""
        cs = caerus()
        X1 = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        X2 = np.array([2.0, 3.0, 4.0, 3.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0])
        
        # First fit
        result1 = cs.fit(X1)
        assert isinstance(result1, dict)
        assert hasattr(cs, 'results')
        
        # Second fit (should overwrite previous results)
        result2 = cs.fit(X2)
        assert isinstance(result2, dict)
        assert hasattr(cs, 'results')
        
        # Results should be different
        assert not np.array_equal(result1['X'], result2['X'])
    
    def test_parameter_persistence(self):
        """Test that parameters persist between fits."""
        cs = caerus(window=20, minperc=5, threshold=0.3)
        
        X = np.array([1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0])
        
        # Fit with default parameters
        results = cs.fit(X)
        assert cs.window == 10
        assert cs.minperc == 5
        assert cs.threshold == 0.3
        
        # Fit with overridden parameters
        results = cs.fit(X, window=20, minperc=2, threshold=0.1)
        assert cs.window == 10
        assert cs.minperc == 2
        assert cs.threshold == 0.1

        # Fit with overridden parameters
        results = cs.fit(X, window=3, minperc=2, threshold=0.1)
        assert cs.window == 3
        assert cs.minperc == 2
        assert cs.threshold == 0.1


if __name__ == "__main__":
    pytest.main([__file__])
