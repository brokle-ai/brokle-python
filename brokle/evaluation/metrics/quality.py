"""
Quality evaluator for comprehensive AI response quality assessment.

Combines multiple quality dimensions into a unified quality score,
providing holistic evaluation of AI response quality.
"""

import logging
from typing import Any, Dict, Optional, List, Union
from dataclasses import dataclass

from ..base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class QualityDimension:
    """
    Represents a single quality dimension with weight and evaluator.
    """
    name: str
    evaluator: BaseEvaluator
    weight: float
    required: bool = False
    min_threshold: Optional[float] = None


class QualityEvaluator(BaseEvaluator):
    """
    Comprehensive quality evaluator that combines multiple quality dimensions.

    Provides holistic quality assessment by combining scores from multiple
    specialized evaluators like accuracy, relevance, coherence, etc.
    """

    def __init__(
        self,
        dimensions: Optional[List[QualityDimension]] = None,
        accuracy_weight: float = 0.3,
        relevance_weight: float = 0.25,
        coherence_weight: float = 0.2,
        completeness_weight: float = 0.15,
        clarity_weight: float = 0.1,
        min_overall_threshold: float = 0.6,
        require_all_dimensions: bool = False,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize quality evaluator.

        Args:
            dimensions: List of QualityDimension objects for custom evaluation
            accuracy_weight: Weight for accuracy dimension
            relevance_weight: Weight for relevance dimension
            coherence_weight: Weight for coherence dimension
            completeness_weight: Weight for completeness dimension
            clarity_weight: Weight for clarity dimension
            min_overall_threshold: Minimum threshold for overall quality
            require_all_dimensions: Whether all dimensions must pass their thresholds
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to BaseEvaluator
        """
        super().__init__(
            name=name or "quality_comprehensive",
            description="Comprehensive quality evaluator combining multiple dimensions",
            **kwargs
        )

        if dimensions:
            self.dimensions = dimensions
        else:
            # Create default dimensions
            self.dimensions = self._create_default_dimensions(
                accuracy_weight, relevance_weight, coherence_weight,
                completeness_weight, clarity_weight
            )

        self.min_overall_threshold = min_overall_threshold
        self.require_all_dimensions = require_all_dimensions

        # Validate weights sum to 1.0
        total_weight = sum(d.weight for d in self.dimensions)
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Dimension weights sum to {total_weight:.3f}, not 1.0")

    def _create_default_dimensions(
        self,
        accuracy_weight: float,
        relevance_weight: float,
        coherence_weight: float,
        completeness_weight: float,
        clarity_weight: float
    ) -> List[QualityDimension]:
        """Create default quality dimensions."""

        from .accuracy import AccuracyEvaluator
        from .relevance import RelevanceEvaluator

        dimensions = []

        # Accuracy dimension
        if accuracy_weight > 0:
            dimensions.append(QualityDimension(
                name="accuracy",
                evaluator=AccuracyEvaluator(strategy="normalized"),
                weight=accuracy_weight,
                required=True,
                min_threshold=0.5
            ))

        # Relevance dimension
        if relevance_weight > 0:
            dimensions.append(QualityDimension(
                name="relevance",
                evaluator=RelevanceEvaluator(strategy="keyword"),
                weight=relevance_weight,
                required=True,
                min_threshold=0.4
            ))

        # Coherence dimension (placeholder - would need specialized evaluator)
        if coherence_weight > 0:
            dimensions.append(QualityDimension(
                name="coherence",
                evaluator=CoherenceEvaluator(),
                weight=coherence_weight,
                min_threshold=0.5
            ))

        # Completeness dimension
        if completeness_weight > 0:
            dimensions.append(QualityDimension(
                name="completeness",
                evaluator=CompletenessEvaluator(),
                weight=completeness_weight,
                min_threshold=0.5
            ))

        # Clarity dimension
        if clarity_weight > 0:
            dimensions.append(QualityDimension(
                name="clarity",
                evaluator=ClarityEvaluator(),
                weight=clarity_weight,
                min_threshold=0.4
            ))

        return dimensions

    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate comprehensive quality of AI response.

        Args:
            prediction: The model's response
            reference: Expected response (optional)
            input_data: The input query/prompt (optional)
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with comprehensive quality score (0.0-1.0)
        """
        try:
            dimension_results = {}
            dimension_scores = {}
            failed_requirements = []

            # Evaluate each dimension
            for dimension in self.dimensions:
                try:
                    result = dimension.evaluator.evaluate(
                        prediction=prediction,
                        reference=reference,
                        input_data=input_data,
                        **kwargs
                    )

                    dimension_results[dimension.name] = result
                    dimension_scores[dimension.name] = result.score

                    # Check minimum threshold
                    if dimension.min_threshold and result.score < dimension.min_threshold:
                        if dimension.required:
                            failed_requirements.append(dimension.name)

                except Exception as e:
                    logger.error(f"Dimension {dimension.name} evaluation failed: {e}")
                    dimension_results[dimension.name] = EvaluationResult(
                        key=dimension.name,
                        score=0.0,
                        comment=f"Evaluation failed: {str(e)}",
                        metadata={"error": str(e)}
                    )
                    dimension_scores[dimension.name] = 0.0

                    if dimension.required:
                        failed_requirements.append(dimension.name)

            # Calculate weighted overall score
            overall_score = sum(
                dimension_scores.get(d.name, 0.0) * d.weight
                for d in self.dimensions
            )

            # Apply requirement failures
            if self.require_all_dimensions and failed_requirements:
                overall_score *= 0.5  # Significant penalty for failing requirements

            # Check overall threshold
            meets_threshold = overall_score >= self.min_overall_threshold

            # Prepare metadata
            metadata = {
                "dimension_scores": dimension_scores,
                "dimension_weights": {d.name: d.weight for d in self.dimensions},
                "failed_requirements": failed_requirements,
                "meets_threshold": meets_threshold,
                "min_overall_threshold": self.min_overall_threshold,
                "require_all_dimensions": self.require_all_dimensions,
                "dimension_count": len(self.dimensions)
            }

            # Add detailed dimension results
            dimension_details = {}
            for name, result in dimension_results.items():
                dimension_details[name] = {
                    "score": result.score,
                    "comment": result.comment,
                    "metadata": result.metadata
                }
            metadata["dimension_details"] = dimension_details

            # Generate comment
            comment = self._generate_comment(overall_score, failed_requirements, meets_threshold)

            return EvaluationResult(
                key=self.name,
                score=overall_score,
                value=overall_score,
                comment=comment,
                metadata=metadata,
                quality_score=overall_score
            )

        except Exception as e:
            logger.error(f"Quality evaluation failed: {e}")
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment=f"Evaluation error: {str(e)}",
                metadata={"error": str(e)}
            )

    def _generate_comment(
        self,
        score: float,
        failed_requirements: List[str],
        meets_threshold: bool
    ) -> str:
        """Generate human-readable comment about quality."""

        if score >= 0.9:
            quality_level = "Excellent"
        elif score >= 0.8:
            quality_level = "High"
        elif score >= 0.7:
            quality_level = "Good"
        elif score >= 0.6:
            quality_level = "Acceptable"
        elif score >= 0.4:
            quality_level = "Poor"
        else:
            quality_level = "Very Poor"

        comment = f"{quality_level} quality: {score:.2f}"

        if failed_requirements:
            comment += f" (Failed: {', '.join(failed_requirements)})"

        if not meets_threshold:
            comment += f" (Below threshold: {self.min_overall_threshold})"

        return comment

    def get_dimension_breakdown(self, evaluation_result: EvaluationResult) -> Dict[str, Any]:
        """Get detailed breakdown of dimension scores."""

        if "dimension_details" not in evaluation_result.metadata:
            return {}

        breakdown = {}
        dimension_details = evaluation_result.metadata["dimension_details"]
        dimension_weights = evaluation_result.metadata["dimension_weights"]

        for name, details in dimension_details.items():
            weight = dimension_weights.get(name, 0.0)
            weighted_score = details["score"] * weight

            breakdown[name] = {
                "raw_score": details["score"],
                "weight": weight,
                "weighted_score": weighted_score,
                "comment": details["comment"],
                "contribution_percentage": (weighted_score / evaluation_result.score * 100) if evaluation_result.score > 0 else 0
            }

        return breakdown


class CoherenceEvaluator(BaseEvaluator):
    """
    Evaluates logical coherence and consistency of AI responses.

    Checks for logical flow, consistent reasoning, and absence of contradictions.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(
            name=name or "coherence",
            description="Evaluates logical coherence and consistency",
            **kwargs
        )

    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate coherence of response.

        This is a simplified implementation. In production, this would use
        more sophisticated NLP techniques to detect contradictions and
        logical inconsistencies.
        """
        try:
            pred_str = str(prediction) if prediction else ""

            # Basic coherence checks
            coherence_score = 0.8  # Default good score

            # Check for obvious contradictions
            contradiction_patterns = [
                ("yes", "no"),
                ("true", "false"),
                ("always", "never"),
                ("all", "none"),
                ("increase", "decrease")
            ]

            pred_lower = pred_str.lower()
            for pos, neg in contradiction_patterns:
                if pos in pred_lower and neg in pred_lower:
                    coherence_score *= 0.7  # Penalty for potential contradiction

            # Check sentence structure (very basic)
            sentences = pred_str.split('.')
            if len(sentences) > 1:
                # Bonus for structured response
                coherence_score = min(1.0, coherence_score * 1.1)

            # Check for excessive repetition
            words = pred_str.split()
            if len(words) > 10:
                unique_words = set(words)
                repetition_ratio = len(unique_words) / len(words)
                if repetition_ratio < 0.5:
                    coherence_score *= 0.8  # Penalty for excessive repetition

            return EvaluationResult(
                key=self.name,
                score=coherence_score,
                value=coherence_score,
                comment=f"Coherence score: {coherence_score:.2f}",
                metadata={
                    "word_count": len(words) if words else 0,
                    "sentence_count": len(sentences),
                    "unique_word_ratio": len(set(words)) / len(words) if words else 0
                }
            )

        except Exception as e:
            logger.error(f"Coherence evaluation failed: {e}")
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment=f"Evaluation error: {str(e)}",
                metadata={"error": str(e)}
            )


class CompletenessEvaluator(BaseEvaluator):
    """
    Evaluates completeness of AI responses.

    Checks whether the response adequately addresses all aspects of the input query.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(
            name=name or "completeness",
            description="Evaluates response completeness",
            **kwargs
        )

    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate completeness of response.

        This is a simplified implementation. In production, this would analyze
        the input query structure and check coverage of all query components.
        """
        try:
            pred_str = str(prediction) if prediction else ""
            input_str = str(input_data) if input_data else ""

            # Basic completeness assessment
            if not pred_str.strip():
                return EvaluationResult(
                    key=self.name,
                    score=0.0,
                    comment="Empty response",
                    metadata={"empty_response": True}
                )

            # Length-based completeness (very basic)
            if len(pred_str) < 10:
                completeness_score = 0.3
            elif len(pred_str) < 50:
                completeness_score = 0.6
            elif len(pred_str) < 100:
                completeness_score = 0.8
            else:
                completeness_score = 1.0

            # Check if response addresses the input (keyword overlap)
            if input_str:
                input_words = set(input_str.lower().split())
                pred_words = set(pred_str.lower().split())
                overlap = len(input_words.intersection(pred_words))
                overlap_ratio = overlap / len(input_words) if input_words else 0

                # Adjust score based on overlap
                completeness_score = (completeness_score + overlap_ratio) / 2

            return EvaluationResult(
                key=self.name,
                score=completeness_score,
                value=completeness_score,
                comment=f"Completeness score: {completeness_score:.2f}",
                metadata={
                    "response_length": len(pred_str),
                    "word_count": len(pred_str.split()),
                    "input_overlap_ratio": overlap_ratio if input_str else None
                }
            )

        except Exception as e:
            logger.error(f"Completeness evaluation failed: {e}")
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment=f"Evaluation error: {str(e)}",
                metadata={"error": str(e)}
            )


class ClarityEvaluator(BaseEvaluator):
    """
    Evaluates clarity and readability of AI responses.

    Assesses factors like sentence structure, vocabulary complexity, and overall readability.
    """

    def __init__(self, name: Optional[str] = None, **kwargs):
        super().__init__(
            name=name or "clarity",
            description="Evaluates response clarity and readability",
            **kwargs
        )

    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate clarity of response.

        This is a simplified implementation. In production, this would use
        readability metrics like Flesch Reading Ease, sentence complexity analysis, etc.
        """
        try:
            pred_str = str(prediction) if prediction else ""

            if not pred_str.strip():
                return EvaluationResult(
                    key=self.name,
                    score=0.0,
                    comment="Empty response",
                    metadata={"empty_response": True}
                )

            clarity_score = 1.0

            # Basic clarity metrics
            sentences = [s.strip() for s in pred_str.split('.') if s.strip()]
            words = pred_str.split()

            avg_sentence_length = len(words) / len(sentences) if sentences else 0
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0

            # Penalties for complexity
            if avg_sentence_length > 25:  # Very long sentences
                clarity_score *= 0.8
            elif avg_sentence_length > 15:  # Moderately long sentences
                clarity_score *= 0.9

            if avg_word_length > 7:  # Very long words
                clarity_score *= 0.9
            elif avg_word_length > 5:  # Moderately long words
                clarity_score *= 0.95

            # Check for excessive use of complex punctuation
            complex_punct_count = pred_str.count(';') + pred_str.count(':') + pred_str.count('(')
            if complex_punct_count > len(sentences):
                clarity_score *= 0.9

            return EvaluationResult(
                key=self.name,
                score=clarity_score,
                value=clarity_score,
                comment=f"Clarity score: {clarity_score:.2f}",
                metadata={
                    "avg_sentence_length": avg_sentence_length,
                    "avg_word_length": avg_word_length,
                    "sentence_count": len(sentences),
                    "word_count": len(words),
                    "complex_punctuation_count": complex_punct_count
                }
            )

        except Exception as e:
            logger.error(f"Clarity evaluation failed: {e}")
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment=f"Evaluation error: {str(e)}",
                metadata={"error": str(e)}
            )