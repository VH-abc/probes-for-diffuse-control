"""
Unit tests for lib.activations module.
"""

import unittest
import sys
import os
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.activations import get_probe_positions


class TestActivationsModule(unittest.TestCase):
    """Test cases for activation extraction."""

    def test_get_probe_positions_last(self):
        """Test getting last non-special token position."""
        input_ids = torch.tensor([1, 2, 3, 4, 5])  # Tokens
        special_token_ids = {1, 5}  # 1 and 5 are special

        positions = get_probe_positions(input_ids, special_token_ids, "last")
        self.assertEqual(positions, [3])  # Index 3 (token 4) is last non-special

    def test_get_probe_positions_first(self):
        """Test getting first non-special token position."""
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        special_token_ids = {1, 5}

        positions = get_probe_positions(input_ids, special_token_ids, "first")
        self.assertEqual(positions, [1])  # Index 1 (token 2) is first non-special

    def test_get_probe_positions_middle(self):
        """Test getting middle non-special token position."""
        input_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7])
        special_token_ids = {1, 7}

        positions = get_probe_positions(input_ids, special_token_ids, "middle")
        # Non-special positions: [1, 2, 3, 4, 5], middle is index 3
        self.assertEqual(positions, [3])

    def test_get_probe_positions_all(self):
        """Test getting all non-special token positions."""
        input_ids = torch.tensor([1, 2, 3, 4, 5])
        special_token_ids = {1, 5}

        positions = get_probe_positions(input_ids, special_token_ids, "all")
        self.assertEqual(positions, [1, 2, 3])  # All non-special positions

    def test_get_probe_positions_empty(self):
        """Test fallback when all tokens are special."""
        input_ids = torch.tensor([1, 2, 3])
        special_token_ids = {1, 2, 3}  # All are special

        positions = get_probe_positions(input_ids, special_token_ids, "last")
        # Should fallback to last position
        self.assertEqual(positions, [2])

    def test_get_probe_positions_invalid(self):
        """Test invalid token position raises error."""
        input_ids = torch.tensor([1, 2, 3])
        special_token_ids = set()

        with self.assertRaises(ValueError):
            get_probe_positions(input_ids, special_token_ids, "invalid")


if __name__ == '__main__':
    unittest.main()

