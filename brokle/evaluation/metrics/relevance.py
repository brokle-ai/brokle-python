"""
Relevance evaluator for measuring response relevance to input queries.

Provides semantic relevance scoring using various strategies including
keyword matching, semantic similarity, and LLM-based evaluation.
"""

import re
import logging
from typing import Any, Dict, Optional, List, Set
from collections import Counter

from ..base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class RelevanceEvaluator(BaseEvaluator):
    """
    Evaluates relevance of responses to input queries.

    Supports multiple relevance strategies:
    - keyword: Keyword overlap and TF-IDF scoring
    - semantic: Semantic similarity using embeddings
    - llm: LLM-based relevance evaluation
    - hybrid: Combination of keyword and semantic approaches
    """

    def __init__(
        self,
        strategy: str = "keyword",
        min_keywords: int = 2,
        keyword_weight: float = 0.6,
        semantic_weight: float = 0.4,
        ignore_stopwords: bool = True,
        case_sensitive: bool = False,
        llm_evaluator: Optional[callable] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize relevance evaluator.

        Args:
            strategy: Relevance strategy ('keyword', 'semantic', 'llm', 'hybrid')
            min_keywords: Minimum keywords required for relevance
            keyword_weight: Weight for keyword-based scoring in hybrid mode
            semantic_weight: Weight for semantic scoring in hybrid mode
            ignore_stopwords: Whether to ignore common stopwords
            case_sensitive: Whether to consider case in keyword matching
            llm_evaluator: Custom LLM function for LLM-based evaluation
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to BaseEvaluator
        """
        super().__init__(
            name=name or f"relevance_{strategy}",
            description=f"Relevance evaluator using {strategy} strategy",
            **kwargs
        )

        self.strategy = strategy
        self.min_keywords = min_keywords
        self.keyword_weight = keyword_weight
        self.semantic_weight = semantic_weight
        self.ignore_stopwords = ignore_stopwords
        self.case_sensitive = case_sensitive
        self.llm_evaluator = llm_evaluator

        # Validate strategy
        valid_strategies = ["keyword", "semantic", "llm", "hybrid"]
        if strategy not in valid_strategies:
            raise ValueError(f"Invalid strategy: {strategy}. Must be one of {valid_strategies}")

        if strategy == "llm" and llm_evaluator is None:
            raise ValueError("llm_evaluator must be provided when using 'llm' strategy")

        # Common English stopwords
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'would', 'you', 'your', 'have', 'had',
            'this', 'they', 'we', 'can', 'could', 'should', 'do', 'did', 'does'
        } if ignore_stopwords else set()

    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate relevance of prediction to input query.

        Args:
            prediction: The model's response
            reference: Expected response (optional, used for comparison)
            input_data: The input query/prompt
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with relevance score (0.0-1.0)
        """
        if input_data is None:
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment="No input provided for relevance evaluation",
                metadata={"error": "missing_input"}
            )

        try:
            # Convert to strings
            pred_str = self._to_string(prediction)
            input_str = self._to_string(input_data)
            ref_str = self._to_string(reference) if reference else None

            # Apply strategy
            if self.strategy == "keyword":
                score = self._keyword_relevance(pred_str, input_str)
            elif self.strategy == "semantic":
                score = self._semantic_relevance(pred_str, input_str)
            elif self.strategy == "llm":
                score = self._llm_relevance(pred_str, input_str, ref_str)
            elif self.strategy == "hybrid":
                score = self._hybrid_relevance(pred_str, input_str)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            # Prepare metadata
            metadata = {
                "strategy": self.strategy,
                "input_length": len(input_str),
                "prediction_length": len(pred_str),
                "min_keywords": self.min_keywords,
                "ignore_stopwords": self.ignore_stopwords
            }

            # Add strategy-specific metadata
            if self.strategy in ["keyword", "hybrid"]:
                input_keywords = self._extract_keywords(input_str)
                pred_keywords = self._extract_keywords(pred_str)
                common_keywords = input_keywords.intersection(pred_keywords)

                metadata.update({
                    "input_keywords": len(input_keywords),
                    "prediction_keywords": len(pred_keywords),
                    "common_keywords": len(common_keywords),
                    "keyword_overlap_ratio": (
                        len(common_keywords) / len(input_keywords)
                        if input_keywords else 0.0
                    )
                })

            comment = self._generate_comment(score, metadata)

            return EvaluationResult(
                key=self.name,
                score=score,
                value=score,
                comment=comment,
                metadata=metadata
            )

        except Exception as e:
            logger.error(f"Relevance evaluation failed: {e}")
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment=f"Evaluation error: {str(e)}",
                metadata={"error": str(e), "strategy": self.strategy}
            )

    def _to_string(self, value: Any) -> str:
        """Convert value to string for analysis."""
        if value is None:
            return ""
        elif isinstance(value, str):
            return value
        elif isinstance(value, dict):
            # Extract text from common dict fields
            if "content" in value:
                return str(value["content"])
            elif "text" in value:
                return str(value["text"])
            elif "message" in value:
                return str(value["message"])
            else:
                return str(value)
        else:
            return str(value)

    def _extract_keywords(self, text: str) -> Set[str]:
        """Extract keywords from text."""
        if not text:
            return set()

        # Convert to lowercase if not case sensitive
        if not self.case_sensitive:
            text = text.lower()

        # Extract words (alphanumeric sequences)
        words = re.findall(r'\b\w+\b', text)

        # Filter out stopwords and short words
        keywords = {
            word for word in words
            if len(word) >= 2 and word not in self.stopwords
        }

        return keywords

    def _keyword_relevance(self, prediction: str, input_text: str) -> float:
        """Calculate relevance based on keyword overlap."""
        input_keywords = self._extract_keywords(input_text)
        pred_keywords = self._extract_keywords(prediction)

        if not input_keywords:
            return 0.5  # Neutral score if no keywords in input

        # Calculate overlap
        common_keywords = input_keywords.intersection(pred_keywords)
        overlap_ratio = len(common_keywords) / len(input_keywords)

        # Bonus for having minimum keywords
        if len(common_keywords) >= self.min_keywords:
            overlap_ratio *= 1.1  # 10% bonus

        # Calculate TF-IDF style scoring
        tf_score = self._calculate_tf_score(prediction, input_keywords)

        # Combine overlap and TF scores
        final_score = (overlap_ratio * 0.7) + (tf_score * 0.3)

        return min(1.0, final_score)

    def _calculate_tf_score(self, text: str, keywords: Set[str]) -> float:
        """Calculate term frequency score for keywords in text."""
        if not keywords or not text:
            return 0.0

        text_lower = text.lower() if not self.case_sensitive else text
        word_counts = Counter(re.findall(r'\b\w+\b', text_lower))

        total_words = sum(word_counts.values())
        if total_words == 0:
            return 0.0

        # Calculate average TF for keywords present in text
        keyword_tfs = []
        for keyword in keywords:
            if keyword in word_counts:
                tf = word_counts[keyword] / total_words
                keyword_tfs.append(tf)

        if not keyword_tfs:
            return 0.0

        return sum(keyword_tfs) / len(keywords)

    def _semantic_relevance(self, prediction: str, input_text: str) -> float:
        """
        Calculate semantic relevance using embeddings.

        Note: This is a placeholder implementation. In production,
        this would use sentence embeddings and cosine similarity.
        """
        # TODO: Implement semantic similarity using embeddings
        # For now, fall back to keyword-based relevance
        logger.warning("Semantic relevance not implemented, falling back to keyword relevance")
        return self._keyword_relevance(prediction, input_text)

    def _llm_relevance(self, prediction: str, input_text: str, reference: Optional[str] = None) -> float:
        """Calculate relevance using LLM-based evaluation."""
        try:
            result = self.llm_evaluator(
                prediction=prediction,
                input_text=input_text,
                reference=reference
            )
            # Ensure result is between 0.0 and 1.0
            return max(0.0, min(1.0, float(result)))
        except Exception as e:
            logger.error(f"LLM relevance evaluation failed: {e}")
            # Fall back to keyword relevance
            return self._keyword_relevance(prediction, input_text)

    def _hybrid_relevance(self, prediction: str, input_text: str) -> float:
        """Calculate relevance using hybrid approach."""
        keyword_score = self._keyword_relevance(prediction, input_text)
        semantic_score = self._semantic_relevance(prediction, input_text)

        # Weighted combination
        final_score = (
            keyword_score * self.keyword_weight +
            semantic_score * self.semantic_weight
        )

        return min(1.0, final_score)

    def _generate_comment(self, score: float, metadata: Dict[str, Any]) -> str:
        """Generate human-readable comment about relevance."""
        if score >= 0.9:
            return "Highly relevant response"
        elif score >= 0.7:
            return "Good relevance to input"
        elif score >= 0.5:
            return "Moderate relevance"
        elif score >= 0.3:
            return "Low relevance"
        else:
            return "Poor relevance to input"


class ContextualRelevanceEvaluator(RelevanceEvaluator):
    """
    Specialized relevance evaluator that considers conversation context.

    Evaluates relevance not just to the immediate input but to the broader
    conversation context and user intent.
    """

    def __init__(
        self,
        context_window: int = 3,
        context_weight: float = 0.3,
        immediate_weight: float = 0.7,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize contextual relevance evaluator.

        Args:
            context_window: Number of previous exchanges to consider
            context_weight: Weight for contextual relevance
            immediate_weight: Weight for immediate input relevance
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to RelevanceEvaluator
        """
        super().__init__(
            name=name or "relevance_contextual",
            **kwargs
        )

        self.context_window = context_window
        self.context_weight = context_weight
        self.immediate_weight = immediate_weight

    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        context: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate contextual relevance.

        Args:
            prediction: The model's response
            reference: Expected response (optional)
            input_data: The immediate input query/prompt
            context: List of previous conversation turns
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with contextual relevance score
        """
        # Get immediate relevance
        immediate_result = super().evaluate(
            prediction=prediction,
            reference=reference,
            input_data=input_data,
            **kwargs
        )

        # Calculate context relevance if context is provided
        context_score = 0.5  # Default neutral score
        if context and len(context) > 0:
            context_score = self._evaluate_context_relevance(prediction, context)

        # Combine scores
        final_score = (
            immediate_result.score * self.immediate_weight +
            context_score * self.context_weight
        )

        # Update metadata
        metadata = immediate_result.metadata.copy()
        metadata.update({
            "context_window": self.context_window,
            "context_turns_used": min(len(context) if context else 0, self.context_window),
            "context_score": context_score,
            "immediate_score": immediate_result.score,
            "context_weight": self.context_weight,
            "immediate_weight": self.immediate_weight
        })

        return EvaluationResult(
            key=self.name,
            score=final_score,
            value=final_score,
            comment=f"Contextual relevance: {final_score:.2f} (immediate: {immediate_result.score:.2f}, context: {context_score:.2f})",
            metadata=metadata
        )

    def _evaluate_context_relevance(self, prediction: str, context: List[Dict[str, Any]]) -> float:
        """Evaluate relevance to conversation context."""
        if not context:
            return 0.5

        pred_str = self._to_string(prediction)
        pred_keywords = self._extract_keywords(pred_str)

        # Extract keywords from recent context
        context_keywords = set()
        recent_context = context[-self.context_window:] if context else []

        for turn in recent_context:
            # Extract keywords from both user input and assistant response
            user_input = turn.get("input", turn.get("user", ""))
            assistant_response = turn.get("output", turn.get("assistant", ""))

            context_keywords.update(self._extract_keywords(user_input))
            context_keywords.update(self._extract_keywords(assistant_response))

        if not context_keywords:
            return 0.5

        # Calculate overlap with context
        common_keywords = pred_keywords.intersection(context_keywords)
        overlap_ratio = len(common_keywords) / len(context_keywords)

        return min(1.0, overlap_ratio * 1.2)  # Small bonus for context awareness


class TaskRelevanceEvaluator(RelevanceEvaluator):
    """
    Specialized relevance evaluator for specific task types.

    Evaluates whether the response appropriately addresses the specific
    type of task (QA, summarization, code generation, etc.).
    """

    def __init__(
        self,
        task_type: str = "general",
        task_keywords: Optional[Dict[str, List[str]]] = None,
        format_requirements: Optional[Dict[str, Any]] = None,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize task-specific relevance evaluator.

        Args:
            task_type: Type of task ('qa', 'summarization', 'code', 'analysis', 'general')
            task_keywords: Expected keywords for different task types
            format_requirements: Format requirements for the task
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to RelevanceEvaluator
        """
        super().__init__(
            name=name or f"relevance_task_{task_type}",
            **kwargs
        )

        self.task_type = task_type
        self.format_requirements = format_requirements or {}

        # Default task keywords
        self.task_keywords = task_keywords or {
            "qa": ["answer", "question", "because", "therefore", "explanation"],
            "summarization": ["summary", "key", "main", "important", "overview"],
            "code": ["function", "class", "import", "return", "def", "var"],
            "analysis": ["analysis", "result", "conclusion", "finding", "trend"],
            "general": []
        }

    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate task-specific relevance.

        Args:
            prediction: The model's response
            reference: Expected response (optional)
            input_data: The input query/prompt
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with task-specific relevance score
        """
        # Get base relevance score
        base_result = super().evaluate(
            prediction=prediction,
            reference=reference,
            input_data=input_data,
            **kwargs
        )

        # Calculate task-specific score
        task_score = self._evaluate_task_relevance(prediction)

        # Format compliance score
        format_score = self._evaluate_format_compliance(prediction)

        # Combine scores
        final_score = (
            base_result.score * 0.5 +
            task_score * 0.3 +
            format_score * 0.2
        )

        # Update metadata
        metadata = base_result.metadata.copy()
        metadata.update({
            "task_type": self.task_type,
            "task_score": task_score,
            "format_score": format_score,
            "base_relevance": base_result.score
        })

        return EvaluationResult(
            key=self.name,
            score=final_score,
            value=final_score,
            comment=f"Task relevance ({self.task_type}): {final_score:.2f}",
            metadata=metadata
        )

    def _evaluate_task_relevance(self, prediction: str) -> float:
        """Evaluate relevance to specific task type."""
        expected_keywords = self.task_keywords.get(self.task_type, [])
        if not expected_keywords:
            return 1.0  # No specific requirements

        pred_keywords = self._extract_keywords(prediction)
        task_keyword_set = set(expected_keywords)

        # Check for presence of task-specific keywords
        common_task_keywords = pred_keywords.intersection(task_keyword_set)
        task_relevance = len(common_task_keywords) / len(task_keyword_set)

        return min(1.0, task_relevance * 1.5)  # Bonus for task alignment

    def _evaluate_format_compliance(self, prediction: str) -> float:
        """Evaluate compliance with format requirements."""
        if not self.format_requirements:
            return 1.0

        score = 1.0

        # Check length requirements
        if "min_length" in self.format_requirements:
            if len(prediction) < self.format_requirements["min_length"]:
                score *= 0.5

        if "max_length" in self.format_requirements:
            if len(prediction) > self.format_requirements["max_length"]:
                score *= 0.8

        # Check structure requirements
        if "requires_list" in self.format_requirements:
            if self.format_requirements["requires_list"]:
                # Look for list indicators
                if not re.search(r'[•\-\*\d+\.]\s', prediction):
                    score *= 0.7

        if "requires_code" in self.format_requirements:
            if self.format_requirements["requires_code"]:
                # Look for code indicators
                if not re.search(r'```|`[^`]+`', prediction):
                    score *= 0.6

        return score