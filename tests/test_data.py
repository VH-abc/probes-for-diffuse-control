"""
Unit tests for lib.data module.
"""

import unittest
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.data import (
    format_mmlu_question,
    extract_answer_from_generation,
    compute_correctness_labels
)


class TestDataModule(unittest.TestCase):
    """Test cases for data loading and formatting."""

    def test_format_mmlu_question(self):
        """Test MMLU question formatting."""
        question_dict = {
            'question': 'What is 2+2?',
            'choices': ['3', '4', '5', '6'],
            'answer': 1,
            'subject': 'math'
        }

        formatted = format_mmlu_question(question_dict)

        self.assertIn('What is 2+2?', formatted)
        self.assertIn('A. 3', formatted)
        self.assertIn('B. 4', formatted)
        self.assertIn('C. 5', formatted)
        self.assertIn('D. 6', formatted)
        self.assertIn('\\box', formatted)

    def test_extract_answer_box_format(self):
        """Test extracting answer in \\box{} format."""
        text = "The answer is \\box{B} because..."
        self.assertEqual(extract_answer_from_generation(text), 'B')

        text = "After careful consideration, \\box{D}."
        self.assertEqual(extract_answer_from_generation(text), 'D')

    def test_extract_answer_plain_letter(self):
        """Test extracting plain letter answers."""
        text = "The answer is C."
        self.assertEqual(extract_answer_from_generation(text), 'C')

        text = "I believe the correct answer is A"
        self.assertEqual(extract_answer_from_generation(text), 'A')

    def test_extract_answer_case_insensitive(self):
        """Test case insensitivity."""
        text = "The answer is \\box{b}"
        self.assertEqual(extract_answer_from_generation(text), 'B')

        text = "The answer is d"
        self.assertEqual(extract_answer_from_generation(text), 'D')

    def test_extract_answer_last_occurrence(self):
        """Test that last occurrence is used."""
        text = "A is wrong. B is also wrong. The answer is C."
        self.assertEqual(extract_answer_from_generation(text), 'C')

    def test_extract_answer_none(self):
        """Test when no answer can be extracted."""
        text = "I don't know the answer to this question."
        self.assertIsNone(extract_answer_from_generation(text))

    def test_compute_correctness_labels(self):
        """Test correctness label computation."""
        questions = [
            {'answer': 0, 'subject': 'math'},  # Correct answer is A
            {'answer': 1, 'subject': 'science'},  # Correct answer is B
            {'answer': 2, 'subject': 'history'},  # Correct answer is C
        ]

        generated_texts = [
            "The answer is \\box{A}",  # Correct
            "The answer is \\box{C}",  # Incorrect
            "The answer is \\box{C}",  # Correct
        ]

        labels, predicted, correct_indices = compute_correctness_labels(
            questions, generated_texts
        )

        self.assertEqual(labels, [1, 0, 1])  # Correct, Incorrect, Correct
        self.assertEqual(predicted, ['A', 'C', 'C'])
        self.assertEqual(correct_indices, [0, 1, 2])

    def test_compute_correctness_unparseable(self):
        """Test that unparseable answers count as incorrect."""
        questions = [
            {'answer': 0, 'subject': 'math'},
        ]

        generated_texts = [
            "I don't know",  # Unparseable
        ]

        labels, predicted, _ = compute_correctness_labels(
            questions, generated_texts
        )

        self.assertEqual(labels, [0])  # Incorrect
        self.assertIsNone(predicted[0])


if __name__ == '__main__':
    unittest.main()

