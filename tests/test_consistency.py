import unittest
import pandas as pd
import numpy as np
import os
import sys

# Add parent directory to path for module imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from audit import consistency

class TestConsistency(unittest.TestCase):
    """
    Unit tests for the consistency module.
    """

    def setUp(self):
        """
        Set up common test data for consistency checks.
        """
        self.df_perfect_agreement = pd.DataFrame({
            'task_id': [1, 1, 2, 2, 3, 3],
            'annotator_id': ['A', 'B', 'A', 'B', 'A', 'B'],
            'label': ['cat', 'cat', 'dog', 'dog', 'bird', 'bird']
        })

        self.df_some_disagreement = pd.DataFrame({
            'task_id': [1, 1, 2, 2, 3, 3],
            'annotator_id': ['A', 'B', 'A', 'B', 'A', 'B'],
            'label': ['cat', 'dog', 'dog', 'dog', 'bird', 'fish']
        })

        self.df_multiple_annotators = pd.DataFrame({
            'task_id': [1, 1, 1, 2, 2, 2, 3, 3, 3],
            'annotator_id': ['A', 'B', 'C', 'A', 'B', 'C', 'A', 'B', 'C'],
            'label': ['pos', 'pos', 'neg', 'neg', 'neg', 'pos', 'neu', 'neu', 'neu']
        })

        self.df_insufficient_annotators = pd.DataFrame({
            'task_id': [1, 2, 3],
            'annotator_id': ['A', 'A', 'A'],
            'label': ['cat', 'dog', 'bird']
        })
        
        self.df_empty = pd.DataFrame(columns=['task_id', 'annotator_id', 'label'])

    def test_perfect_agreement(self):
        """
        Test case for perfect inter-annotator agreement.
        """
        scores = consistency.compute_agreement(
            self.df_perfect_agreement, 'task_id', 'annotator_id', 'label'
        )
        self.assertAlmostEqual(scores['cohen_kappa'], 1.0, places=5)
        self.assertAlmostEqual(scores['krippendorff_alpha'], 1.0, places=5)

    def test_some_disagreement(self):
        """
        Test case for some inter-annotator disagreement.
        Cohen's Kappa and Krippendorff's Alpha should be less than 1.0.
        Exact values can be tricky due to pairwise averaging and simplified alpha.
        """
        scores = consistency.compute_agreement(
            self.df_some_disagreement, 'task_id', 'annotator_id', 'label'
        )
        self.assertIsNotNone(scores['cohen_kappa'])
        self.assertLess(scores['cohen_kappa'], 1.0)
        self.assertIsNotNone(scores['krippendorff_alpha'])
        self.assertLess(scores['krippendorff_alpha'], 1.0)
        # Check for non-negative values, as agreement scores are typically >= -1
        self.assertGreaterEqual(scores['cohen_kappa'], -1.0)
        self.assertGreaterEqual(scores['krippendorff_alpha'], -1.0)


    def test_multiple_annotators(self):
        """
        Test case for multiple annotators with mixed agreement.
        """
        scores = consistency.compute_agreement(
            self.df_multiple_annotators, 'task_id', 'annotator_id', 'label'
        )
        self.assertIsNotNone(scores['cohen_kappa'])
        self.assertIsNotNone(scores['krippendorff_alpha'])
        self.assertLess(scores['cohen_kappa'], 1.0)
        self.assertLess(scores['krippendorff_alpha'], 1.0)
        self.assertGreaterEqual(scores['cohen_kappa'], -1.0)
        self.assertGreaterEqual(scores['krippendorff_alpha'], -1.0)

    def test_insufficient_annotators(self):
        """
        Test case for insufficient annotators (less than 2).
        Scores should be NaN.
        """
        scores = consistency.compute_agreement(
            self.df_insufficient_annotators, 'task_id', 'annotator_id', 'label'
        )
        self.assertTrue(np.isnan(scores['cohen_kappa']))
        self.assertTrue(np.isnan(scores['krippendorff_alpha']))

    def test_empty_dataframe(self):
        """
        Test case for an empty input DataFrame.
        Scores should be NaN.
        """
        scores = consistency.compute_agreement(
            self.df_empty, 'task_id', 'annotator_id', 'label'
        )
        self.assertTrue(np.isnan(scores['cohen_kappa']))
        self.assertTrue(np.isnan(scores['krippendorff_alpha']))

    def test_missing_columns(self):
        """
        Test case for missing required columns.
        Should raise a KeyError from pandas pivot_table or other operations.
        """
        df_missing_col = self.df_perfect_agreement.drop(columns=['label'])
        with self.assertRaises(KeyError): # Or ValueError depending on exact failure point
            consistency.compute_agreement(
                df_missing_col, 'task_id', 'annotator_id', 'label'
            )

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
