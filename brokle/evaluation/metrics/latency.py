"""
Latency evaluator for measuring AI response time performance.

Evaluates response latency against acceptable thresholds and
provides performance insights for optimization.
"""

import logging
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta

from ..base import BaseEvaluator, EvaluationResult

logger = logging.getLogger(__name__)


class LatencyEvaluator(BaseEvaluator):
    """
    Evaluates latency performance of AI responses.

    Considers multiple factors:
    - Absolute response time
    - Relative to content length
    - Percentile performance
    - SLA compliance
    """

    def __init__(
        self,
        target_latency_ms: float = 2000.0,
        max_acceptable_latency_ms: float = 10000.0,
        per_token_baseline_ms: float = 50.0,
        sla_threshold_ms: Optional[float] = None,
        percentile_target: float = 95.0,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize latency evaluator.

        Args:
            target_latency_ms: Target latency in milliseconds
            max_acceptable_latency_ms: Maximum acceptable latency
            per_token_baseline_ms: Expected time per output token
            sla_threshold_ms: SLA threshold (defaults to target_latency_ms)
            percentile_target: Target percentile for performance (e.g., 95th percentile)
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to BaseEvaluator
        """
        super().__init__(
            name=name or "latency_performance",
            description="Latency performance evaluator for AI responses",
            **kwargs
        )

        self.target_latency_ms = target_latency_ms
        self.max_acceptable_latency_ms = max_acceptable_latency_ms
        self.per_token_baseline_ms = per_token_baseline_ms
        self.sla_threshold_ms = sla_threshold_ms or target_latency_ms
        self.percentile_target = percentile_target

        # Track historical performance for percentile calculations
        self._latency_history: List[float] = []

    def evaluate(
        self,
        prediction: Any,
        reference: Any = None,
        input_data: Any = None,
        latency_ms: Optional[float] = None,
        output_tokens: Optional[int] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate latency performance.

        Args:
            prediction: The model's response
            reference: Expected response (optional)
            input_data: The input query/prompt (optional)
            latency_ms: Response latency in milliseconds
            output_tokens: Number of output tokens generated
            start_time: Request start time
            end_time: Request end time
            **kwargs: Additional evaluation parameters

        Returns:
            EvaluationResult with latency score (0.0-1.0)
        """
        try:
            # Calculate latency from different sources
            actual_latency = self._calculate_latency(
                latency_ms=latency_ms,
                start_time=start_time,
                end_time=end_time
            )

            if actual_latency is None:
                return EvaluationResult(
                    key=self.name,
                    score=0.0,
                    comment="No latency data provided",
                    metadata={"error": "missing_latency_data"}
                )

            # Add to history for percentile tracking
            self._latency_history.append(actual_latency)

            # Keep history manageable (last 1000 measurements)
            if len(self._latency_history) > 1000:
                self._latency_history = self._latency_history[-1000:]

            # Calculate performance scores
            absolute_score = self._calculate_absolute_score(actual_latency)
            relative_score = self._calculate_relative_score(actual_latency, output_tokens)
            sla_score = self._calculate_sla_score(actual_latency)
            percentile_score = self._calculate_percentile_score(actual_latency)

            # Weighted combination
            final_score = (
                absolute_score * 0.4 +
                relative_score * 0.3 +
                sla_score * 0.2 +
                percentile_score * 0.1
            )

            # Prepare metadata
            metadata = {
                "latency_ms": actual_latency,
                "target_latency_ms": self.target_latency_ms,
                "max_acceptable_latency_ms": self.max_acceptable_latency_ms,
                "sla_threshold_ms": self.sla_threshold_ms,
                "output_tokens": output_tokens,
                "absolute_score": absolute_score,
                "relative_score": relative_score,
                "sla_score": sla_score,
                "percentile_score": percentile_score,
                "sla_compliant": actual_latency <= self.sla_threshold_ms,
                "performance_category": self._categorize_performance(actual_latency)
            }

            # Add token-based metrics
            if output_tokens:
                metadata.update({
                    "ms_per_token": actual_latency / output_tokens,
                    "tokens_per_second": output_tokens / (actual_latency / 1000.0),
                    "expected_latency_ms": output_tokens * self.per_token_baseline_ms
                })

            # Add percentile information
            if len(self._latency_history) >= 10:
                percentiles = self._calculate_percentiles(self._latency_history)
                metadata["percentiles"] = percentiles
                metadata["current_percentile"] = self._find_percentile(actual_latency, self._latency_history)

            comment = self._generate_comment(actual_latency, final_score, metadata)

            return EvaluationResult(
                key=self.name,
                score=final_score,
                value=actual_latency,
                comment=comment,
                metadata=metadata,
                latency_impact=actual_latency
            )

        except Exception as e:
            logger.error(f"Latency evaluation failed: {e}")
            return EvaluationResult(
                key=self.name,
                score=0.0,
                comment=f"Evaluation error: {str(e)}",
                metadata={"error": str(e)}
            )

    def _calculate_latency(
        self,
        latency_ms: Optional[float] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Optional[float]:
        """Calculate latency from available data."""

        if latency_ms is not None:
            return latency_ms

        if start_time and end_time:
            delta = end_time - start_time
            return delta.total_seconds() * 1000.0

        return None

    def _calculate_absolute_score(self, latency_ms: float) -> float:
        """Calculate score based on absolute latency."""

        if latency_ms <= self.target_latency_ms:
            return 1.0
        elif latency_ms >= self.max_acceptable_latency_ms:
            return 0.0
        else:
            # Linear scale between target and max acceptable
            range_ms = self.max_acceptable_latency_ms - self.target_latency_ms
            excess_ms = latency_ms - self.target_latency_ms
            return 1.0 - (excess_ms / range_ms)

    def _calculate_relative_score(self, latency_ms: float, output_tokens: Optional[int]) -> float:
        """Calculate score relative to output size."""

        if not output_tokens:
            return 0.5  # Neutral score if no token info

        expected_latency = output_tokens * self.per_token_baseline_ms

        if expected_latency == 0:
            return 1.0 if latency_ms <= self.target_latency_ms else 0.0

        efficiency_ratio = expected_latency / latency_ms

        # Score based on efficiency (1.0 = meeting baseline, >1.0 = better than baseline)
        if efficiency_ratio >= 1.0:
            return min(1.0, efficiency_ratio * 0.5)  # Cap bonus at 1.0
        else:
            return efficiency_ratio

    def _calculate_sla_score(self, latency_ms: float) -> float:
        """Calculate SLA compliance score."""

        if latency_ms <= self.sla_threshold_ms:
            return 1.0
        else:
            # Rapid degradation after SLA breach
            excess_ratio = latency_ms / self.sla_threshold_ms
            return max(0.0, 1.0 - (excess_ratio - 1.0) * 2.0)

    def _calculate_percentile_score(self, latency_ms: float) -> float:
        """Calculate score based on percentile performance."""

        if len(self._latency_history) < 10:
            return 0.5  # Neutral score with insufficient data

        current_percentile = self._find_percentile(latency_ms, self._latency_history)

        # Score based on how close to target percentile
        target_percentile = self.percentile_target

        if current_percentile <= target_percentile:
            return 1.0
        else:
            # Degrade score for worse than target percentile
            excess = current_percentile - target_percentile
            return max(0.0, 1.0 - (excess / (100.0 - target_percentile)))

    def _categorize_performance(self, latency_ms: float) -> str:
        """Categorize performance level."""

        if latency_ms <= self.target_latency_ms * 0.5:
            return "excellent"
        elif latency_ms <= self.target_latency_ms:
            return "good"
        elif latency_ms <= self.target_latency_ms * 2:
            return "acceptable"
        elif latency_ms <= self.max_acceptable_latency_ms:
            return "poor"
        else:
            return "unacceptable"

    def _calculate_percentiles(self, latencies: List[float]) -> Dict[str, float]:
        """Calculate common percentiles from latency history."""

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        percentiles = {}
        for p in [50, 75, 90, 95, 99]:
            index = int((p / 100.0) * (n - 1))
            percentiles[f"p{p}"] = sorted_latencies[index]

        return percentiles

    def _find_percentile(self, value: float, data: List[float]) -> float:
        """Find what percentile a value represents in the dataset."""

        sorted_data = sorted(data)
        n = len(sorted_data)

        # Find position of value
        count_below = sum(1 for x in sorted_data if x < value)
        count_equal = sum(1 for x in sorted_data if x == value)

        # Calculate percentile (midpoint of range)
        percentile = ((count_below + count_equal / 2) / n) * 100

        return percentile

    def _generate_comment(self, latency_ms: float, score: float, metadata: Dict[str, Any]) -> str:
        """Generate human-readable comment about latency performance."""

        category = metadata.get("performance_category", "unknown")
        sla_compliant = metadata.get("sla_compliant", False)

        if category == "excellent":
            return f"Excellent latency: {latency_ms:.0f}ms (well below target)"
        elif category == "good":
            return f"Good latency: {latency_ms:.0f}ms (within target)"
        elif category == "acceptable":
            return f"Acceptable latency: {latency_ms:.0f}ms (above target but manageable)"
        elif category == "poor":
            return f"Poor latency: {latency_ms:.0f}ms (significantly above target)"
        else:
            return f"Unacceptable latency: {latency_ms:.0f}ms (exceeds maximum threshold)"

    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of historical performance."""

        if not self._latency_history:
            return {"message": "No latency data available"}

        latencies = self._latency_history
        percentiles = self._calculate_percentiles(latencies)

        sla_violations = sum(1 for l in latencies if l > self.sla_threshold_ms)
        sla_compliance_rate = 1.0 - (sla_violations / len(latencies))

        return {
            "total_measurements": len(latencies),
            "average_latency_ms": sum(latencies) / len(latencies),
            "median_latency_ms": percentiles.get("p50", 0),
            "percentiles": percentiles,
            "sla_compliance_rate": sla_compliance_rate,
            "sla_violations": sla_violations,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies)
        }


class ThroughputEvaluator(LatencyEvaluator):
    """
    Specialized evaluator for throughput performance.

    Focuses on tokens/requests per second rather than individual latency.
    """

    def __init__(
        self,
        target_throughput_tps: float = 20.0,  # tokens per second
        target_throughput_rps: float = 1.0,   # requests per second
        measurement_window_seconds: float = 60.0,
        name: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize throughput evaluator.

        Args:
            target_throughput_tps: Target tokens per second
            target_throughput_rps: Target requests per second
            measurement_window_seconds: Window for throughput measurement
            name: Custom name for the evaluator
            **kwargs: Additional parameters passed to LatencyEvaluator
        """
        super().__init__(
            name=name or "throughput_performance",
            **kwargs
        )

        self.target_throughput_tps = target_throughput_tps
        self.target_throughput_rps = target_throughput_rps
        self.measurement_window_seconds = measurement_window_seconds

        # Track throughput measurements
        self._throughput_history: List[Dict[str, Any]] = []

    def evaluate(
        self,
        output_tokens: Optional[int] = None,
        concurrent_requests: Optional[int] = None,
        **kwargs
    ) -> EvaluationResult:
        """
        Evaluate throughput performance.

        Args:
            output_tokens: Number of output tokens
            concurrent_requests: Number of concurrent requests
            **kwargs: Additional parameters passed to base evaluator

        Returns:
            EvaluationResult with throughput score
        """
        # Get base latency evaluation
        base_result = super().evaluate(output_tokens=output_tokens, **kwargs)

        # Calculate throughput metrics
        latency_ms = base_result.metadata.get("latency_ms", 0)

        if latency_ms > 0 and output_tokens:
            tokens_per_second = output_tokens / (latency_ms / 1000.0)
            requests_per_second = 1.0 / (latency_ms / 1000.0)
        else:
            tokens_per_second = 0.0
            requests_per_second = 0.0

        # Record measurement
        measurement = {
            "timestamp": datetime.utcnow(),
            "tokens_per_second": tokens_per_second,
            "requests_per_second": requests_per_second,
            "output_tokens": output_tokens,
            "latency_ms": latency_ms,
            "concurrent_requests": concurrent_requests
        }
        self._throughput_history.append(measurement)

        # Clean old measurements
        cutoff_time = datetime.utcnow() - timedelta(seconds=self.measurement_window_seconds)
        self._throughput_history = [
            m for m in self._throughput_history
            if m["timestamp"] > cutoff_time
        ]

        # Calculate throughput scores
        tps_score = min(1.0, tokens_per_second / self.target_throughput_tps)
        rps_score = min(1.0, requests_per_second / self.target_throughput_rps)

        # Combined throughput score
        throughput_score = (tps_score + rps_score) / 2.0

        # Update metadata
        metadata = base_result.metadata.copy()
        metadata.update({
            "tokens_per_second": tokens_per_second,
            "requests_per_second": requests_per_second,
            "target_throughput_tps": self.target_throughput_tps,
            "target_throughput_rps": self.target_throughput_rps,
            "tps_score": tps_score,
            "rps_score": rps_score,
            "throughput_score": throughput_score,
            "concurrent_requests": concurrent_requests,
            "measurement_window_seconds": self.measurement_window_seconds
        })

        # Add window statistics if we have enough data
        if len(self._throughput_history) > 1:
            window_stats = self._calculate_window_stats()
            metadata["window_stats"] = window_stats

        comment = f"Throughput: {tokens_per_second:.1f} TPS, {requests_per_second:.2f} RPS"

        return EvaluationResult(
            key=self.name,
            score=throughput_score,
            value=tokens_per_second,
            comment=comment,
            metadata=metadata,
            latency_impact=latency_ms
        )

    def _calculate_window_stats(self) -> Dict[str, Any]:
        """Calculate statistics for the current measurement window."""

        if not self._throughput_history:
            return {}

        tps_values = [m["tokens_per_second"] for m in self._throughput_history]
        rps_values = [m["requests_per_second"] for m in self._throughput_history]

        return {
            "avg_tokens_per_second": sum(tps_values) / len(tps_values),
            "avg_requests_per_second": sum(rps_values) / len(rps_values),
            "max_tokens_per_second": max(tps_values),
            "max_requests_per_second": max(rps_values),
            "min_tokens_per_second": min(tps_values),
            "min_requests_per_second": min(rps_values),
            "measurement_count": len(self._throughput_history)
        }