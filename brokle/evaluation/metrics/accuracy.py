"""
Accuracy evaluator for comparing predictions with reference answers.

Provides multiple accuracy measurement strategies including exact match,
fuzzy matching, and semantic similarity.
"""

import re
import logging
from typing import Any, Dict, Optional, Union, List
from difflib import SequenceMatcher

from ..base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class AccuracyEvaluator(BaseEvaluator):
    """
    Evaluates accuracy of predictions against reference answers.

    Supports multiple accuracy strategies:
    - exact: Exact string matching
    - fuzzy: Fuzzy string matching with configurable threshold
    - normalized: Normalized string comparison (lowercase, stripped)
    - semantic: Semantic similarity (requires embeddings)
    - custom: Custom comparison function
    """

    def __init__(
        self,
        strategy: str = "normalized",
        threshold: float = 0.8,
        case_sensitive: bool = False,
        ignore_whitespace: bool = True,
        ignore_punctuation: bool = False,
        custom_comparator: Optional[callable] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize accuracy evaluator.

        Args:
            strategy: Accuracy strategy ('exact', 'fuzzy', 'normalized', 'semantic', 'custom')
            threshold: Threshold for fuzzy and semantic matching (0.0-1.0)
            case_sensitive: Whether to consider case in comparisons
            ignore_whitespace: Whether to ignore whitespace differences
            ignore_punctuation: Whether to ignore punctuation differences
            custom_comparator: Custom comparison function for 'custom' strategy
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to BaseEvaluator
        """
        super().__init__(
            name=name or f"accuracy_{strategy}",
            description=f"Accuracy evaluator using {strategy} strategy",
            **kwargs
        )

        self.strategy = strategy
        self.threshold = threshold
        self.case_sensitive = case_sensitive
        self.ignore_whitespace = ignore_whitespace
        self.ignore_punctuation = ignore_punctuation
        self.custom_comparator = custom_comparator

        # Validate strategy
        valid_strategies = ["exact", "fuzzy", "normalized", "semantic", "custom"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")

        if strategy == "custom" and custom_comparator is None:
            raise ValueError("custom_comparator must be provided when using 'custom' strategy")

    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate accuracy of prediction against reference.

        Args:
            prediction: The model's prediction
            reference: The expected/reference answer
            input_data: The input that generated the prediction (unused)
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with accuracy score (0.0-1.0)
        """
        if reference is None:
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment="No reference provided for accuracy evaluation",
                metadata={"error": "missing_reference"}
            )

        try:
            # Convert to strings
            pred_str = self._to_string(prediction)
            ref_str = self._to_string(reference)

            # Apply strategy
            if self.strategy == "exact":
                score = self._exact_match(pred_str, ref_str)
            elif self.strategy == "fuzzy":
                score = self._fuzzy_match(pred_str, ref_str)
            elif self.strategy == "normalized":
                score = self._normalized_match(pred_str, ref_str)
            elif self.strategy == "semantic":
                score = self._semantic_match(pred_str, ref_str)
            elif self.strategy == "custom":
                score = self._custom_match(pred_str, ref_str)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            # Prepare metadata
            metadata = {
                "strategy": self.strategy,
                "threshold": self.threshold,
                "prediction_length": len(pred_str),
                "reference_length": len(ref_str),
                "case_sensitive": self.case_sensitive,
                "ignore_whitespace": self.ignore_whitespace,
                "ignore_punctuation": self.ignore_punctuation
            }

            # Add strategy-specific metadata
            if self.strategy == "fuzzy":
                similarity = SequenceMatcher(None, pred_str, ref_str).ratio()
                metadata["raw_similarity"] = similarity

            comment = self._generate_comment(score, pred_str, ref_str)

            return EvaluationResult(
                key=self.name,
                score=score,
                value=score,
                comment=comment,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Accuracy evaluation failed: {e}")
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment=f"Evaluation error: {str(e)}",
                metadata={"error": str(e), "strategy": self.strategy}
            )

    def _to_string(self, value: Any) -> str:
        """Convert value to string for comparison."""
        if value is None:
            return ""
        elif isinstance(value, str):
            return value
        elif isinstance(value, (list, dict)):
            # For structured data, convert to string representation
            return str(value)
        else:
            return str(value)

    def _normalize_string(self, text: str) -> str:
        """Normalize string based on configuration."""
        if not self.case_sensitive:
            text = text.lower()

        if self.ignore_whitespace:
            text = re.sub(r'\s+', ' ', text.strip())

        if self.ignore_punctuation:
            text = re.sub(r'[^\w\s]', '', text)

        return text

    def _exact_match(self, prediction: str, reference: str) -> float:
        """Exact string matching."""
        pred_norm = self._normalize_string(prediction)
        ref_norm = self._normalize_string(reference)
        return 1.0 if pred_norm == ref_norm else 0.0

    def _fuzzy_match(self, prediction: str, reference: str) -> float:
        """Fuzzy string matching using SequenceMatcher."""
        pred_norm = self._normalize_string(prediction)
        ref_norm = self._normalize_string(reference)

        similarity = SequenceMatcher(None, pred_norm, ref_norm).ratio()
        return 1.0 if similarity >= self.threshold else similarity

    def _normalized_match(self, prediction: str, reference: str) -> float:
        """Normalized string comparison (default strategy)."""
        pred_norm = self._normalize_string(prediction)
        ref_norm = self._normalize_string(reference)

        if pred_norm == ref_norm:
            return 1.0

        # Fall back to fuzzy matching for partial credit
        similarity = SequenceMatcher(None, pred_norm, ref_norm).ratio()
        return similarity if similarity >= 0.5 else 0.0

    def _semantic_match(self, prediction: str, reference: str) -> float:
        """
        Semantic similarity matching.

        Note: This is a placeholder implementation. In production,
        this would use embeddings and vector similarity.
        """
        # TODO: Implement semantic similarity using embeddings
        # For now, fall back to fuzzy matching
        logger.warning("Semantic matching not implemented, falling back to fuzzy matching")
        return self._fuzzy_match(prediction, reference)

    def _custom_match(self, prediction: str, reference: str) -> float:
        """Custom comparison using user-provided function."""
        try:
            result = self.custom_comparator(prediction, reference)
            # Ensure result is between 0.0 and 1.0
            return max(0.0, min(1.0, float(result)))
        except Exception as e:
            logger.error(f"Custom comparator failed: {e}")
            return 0.0

    def _generate_comment(self, score: float, prediction: str, reference: str) -> str:
        """Generate human-readable comment about the accuracy."""
        if score == 1.0:
            return "Perfect match"
        elif score >= 0.8:
            return "High accuracy"
        elif score >= 0.6:
            return "Moderate accuracy"
        elif score >= 0.3:
            return "Low accuracy"
        else:
            return "Poor accuracy"


class MultiChoiceAccuracyEvaluator(AccuracyEvaluator):
    """
    Specialized accuracy evaluator for multiple choice questions.

    Handles common multiple choice formats and extracts answers intelligently.
    """

    def __init__(
        self,
        choices: Optional[List[str]] = None,
        extract_letter: bool = True,
        extract_content: bool = True,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize multiple choice accuracy evaluator.

        Args:
            choices: List of valid choice labels (e.g., ['A', 'B', 'C', 'D'])
            extract_letter: Whether to extract choice letters (A, B, C, D)
            extract_content: Whether to extract choice content
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to AccuracyEvaluator
        """
        super().__init__(
            name=name or "accuracy_multiple_choice",
            strategy="custom",
            custom_comparator=self._compare_choices,
            **kwargs
        )

        self.choices = choices or ['A', 'B', 'C', 'D']
        self.extract_letter = extract_letter
        self.extract_content = extract_content

    def _compare_choices(self, prediction: str, reference: str) -> float:
        """Compare multiple choice answers."""
        pred_choice = self._extract_choice(prediction)
        ref_choice = self._extract_choice(reference)

        if pred_choice and ref_choice:
            return 1.0 if pred_choice.upper() == ref_choice.upper() else 0.0

        # Fall back to exact string matching
        return 1.0 if prediction.strip().upper() == reference.strip().upper() else 0.0

    def _extract_choice(self, text: str) -> Optional[str]:
        """Extract choice letter from text."""
        if not text:
            return None

        text = text.strip().upper()

        # Direct choice letter
        if len(text) == 1 and text in self.choices:
            return text

        # Choice with parentheses or periods
        for choice in self.choices:
            patterns = [
                f"\\({choice}\\)",  # (A)
                f"{choice}\\.",     # A.
                f"{choice}\\)",     # A)
                f"^{choice}\\b",    # A at start
                f"\\b{choice}$",    # A at end
            ]

            for pattern in patterns:
                if re.search(pattern, text):
                    return choice

        # Extract first letter if it's a valid choice
        first_char = text[0] if text else ""
        if first_char in self.choices:
            return first_char

        return None


class NumericAccuracyEvaluator(AccuracyEvaluator):
    """
    Specialized accuracy evaluator for numeric answers.

    Handles floating point precision, unit conversion, and range-based accuracy.
    """

    def __init__(
        self,
        tolerance: float = 0.01,
        relative_tolerance: bool = True,
        extract_numbers: bool = True,
        units_matter: bool = False,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize numeric accuracy evaluator.

        Args:
            tolerance: Tolerance for numeric comparison
            relative_tolerance: Whether tolerance is relative (percentage) or absolute
            extract_numbers: Whether to extract numbers from text
            units_matter: Whether units must match exactly
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to AccuracyEvaluator
        """
        super().__init__(
            name=name or "accuracy_numeric",
            strategy="custom",
            custom_comparator=self._compare_numbers,
            **kwargs
        )

        self.tolerance = tolerance
        self.relative_tolerance = relative_tolerance
        self.extract_numbers = extract_numbers
        self.units_matter = units_matter

    def _compare_numbers(self, prediction: str, reference: str) -> float:
        """Compare numeric answers with tolerance."""
        try:
            pred_num = self._extract_number(prediction)
            ref_num = self._extract_number(reference)

            if pred_num is None or ref_num is None:
                # Fall back to string comparison
                return 1.0 if prediction.strip() == reference.strip() else 0.0

            # Calculate difference
            if self.relative_tolerance and ref_num != 0:
                diff = abs(pred_num - ref_num) / abs(ref_num)
            else:
                diff = abs(pred_num - ref_num)

            # Return 1.0 if within tolerance, otherwise scale based on difference
            if diff <= self.tolerance:
                return 1.0
            else:
                # Gradual scoring based on how far off the answer is
                return max(0.0, 1.0 - (diff / self.tolerance))

        except Exception as e:
            logger.error(f"Numeric comparison failed: {e}")
            return 0.0

    def _extract_number(self, text: str) -> Optional[float]:
        """Extract number from text."""
        if not text:
            return None

        # Remove common formatting
        text = text.replace(',', '').replace('$', '').replace('%', '')

        # Try to extract number with regex
        number_pattern = r'-?\d+\.?\d*'
        matches = re.findall(number_pattern, text)

        if matches:
            try:
                return float(matches[0])
            except ValueError:
                pass

        # Try direct conversion
        try:
            return float(text.strip())
        except ValueError:
            return None