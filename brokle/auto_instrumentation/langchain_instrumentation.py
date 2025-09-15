"""
LangChain auto-instrumentation for Brokle observability.

This module automatically instruments LangChain components to capture
comprehensive observability data including chains, agents, and LLM calls.
"""

import functools
import logging
import time
from typing import Any, Dict, Optional, Union, List
from datetime import datetime, timezone

try:
    import langchain
    from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager
    from langchain.schema import LLMResult, Generation
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)


class BrokleCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler for Brokle observability."""

    def __init__(self, brokle_client=None):
        super().__init__()
        self._config = None
        self._client = brokle_client
        self._traces = {}
        self._observations = {}

    @property
    def config(self):
        """Get or create Brokle config with lazy loading."""
        if self._config is None:
            try:
                from ..config import get_config
                self._config = get_config()
            except Exception as e:
                logger.warning(f"Failed to load Brokle config: {e}")
                self._config = None
        return self._config

    @property
    def client(self):
        """Get Brokle client for observability."""
        if self._client is None:
            try:
                from ..client import get_client
                self._client = get_client()
            except Exception as e:
                logger.warning(f"Failed to initialize Brokle client: {e}")
                self._client = None
        return self._client

    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts running."""
        try:
            run_id = kwargs.get("run_id")
            parent_run_id = kwargs.get("parent_run_id")

            # Create trace if this is a top-level run
            if not parent_run_id:
                trace = self.client.observability.create_trace_sync(
                    name=f"langchain_{serialized.get('_type', 'llm')}",
                    metadata={
                        "library": "langchain",
                        "component_type": serialized.get("_type"),
                        "component_name": serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
                    }
                )
                self._traces[run_id] = trace.id
                trace_id = trace.id
            else:
                # Use parent's trace
                trace_id = self._traces.get(parent_run_id)
                if trace_id:
                    self._traces[run_id] = trace_id

            if not trace_id:
                logger.warning(f"No trace found for run {run_id}")
                return

            # Create observation for LLM call
            observation = self.client.observability.create_observation_sync(
                trace_id=trace_id,
                name=f"LangChain LLM: {serialized.get('id', ['unknown'])[-1] if serialized.get('id') else 'unknown'}",
                observation_type="llm",
                provider=self._extract_provider(serialized),
                model=self._extract_model(serialized),
                input_data={
                    "prompts": prompts,
                    "serialized": serialized,
                    "kwargs": self._clean_kwargs(kwargs)
                },
                start_time=datetime.now(timezone.utc)
            )

            self._observations[run_id] = observation.id

        except Exception as e:
            logger.error(f"Failed to start LLM observation: {e}")

    def on_llm_end(self, response: LLMResult, **kwargs) -> None:
        """Called when LLM ends running."""
        try:
            run_id = kwargs.get("run_id")
            observation_id = self._observations.get(run_id)

            if not observation_id:
                logger.warning(f"No observation found for run {run_id}")
                return

            # Extract response data
            response_data = self._extract_llm_response(response)

            # Complete observation
            self.client.observability.complete_observation_sync(
                observation_id,
                end_time=datetime.now(timezone.utc),
                output_data=response_data,
                prompt_tokens=response_data.get("prompt_tokens"),
                completion_tokens=response_data.get("completion_tokens"),
                total_tokens=response_data.get("total_tokens"),
                total_cost=response_data.get("total_cost"),
                status_message="success"
            )

            # Clean up
            self._observations.pop(run_id, None)

        except Exception as e:
            logger.error(f"Failed to end LLM observation: {e}")

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when LLM errors."""
        try:
            run_id = kwargs.get("run_id")
            observation_id = self._observations.get(run_id)

            if observation_id:
                self.client.observability.complete_observation_sync(
                    observation_id,
                    end_time=datetime.now(timezone.utc),
                    status_message=f"error: {str(error)}"
                )

            # Clean up
            self._observations.pop(run_id, None)

        except Exception as e:
            logger.error(f"Failed to handle LLM error: {e}")

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain starts running."""
        try:
            run_id = kwargs.get("run_id")
            parent_run_id = kwargs.get("parent_run_id")

            # Create trace if this is a top-level run
            if not parent_run_id:
                trace = self.client.observability.create_trace_sync(
                    name=f"langchain_{serialized.get('_type', 'chain')}",
                    metadata={
                        "library": "langchain",
                        "component_type": serialized.get("_type"),
                        "component_name": serialized.get("id", ["unknown"])[-1] if serialized.get("id") else "unknown"
                    }
                )
                self._traces[run_id] = trace.id
                trace_id = trace.id
            else:
                # Use parent's trace
                trace_id = self._traces.get(parent_run_id)
                if trace_id:
                    self._traces[run_id] = trace_id

            if not trace_id:
                logger.warning(f"No trace found for run {run_id}")
                return

            # Create observation for chain
            observation = self.client.observability.create_observation_sync(
                trace_id=trace_id,
                name=f"LangChain Chain: {serialized.get('id', ['unknown'])[-1] if serialized.get('id') else 'unknown'}",
                observation_type="span",
                input_data={
                    "inputs": inputs,
                    "serialized": serialized,
                    "kwargs": self._clean_kwargs(kwargs)
                },
                start_time=datetime.now(timezone.utc)
            )

            self._observations[run_id] = observation.id

        except Exception as e:
            logger.error(f"Failed to start chain observation: {e}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs) -> None:
        """Called when chain ends running."""
        try:
            run_id = kwargs.get("run_id")
            observation_id = self._observations.get(run_id)

            if not observation_id:
                logger.warning(f"No observation found for run {run_id}")
                return

            # Complete observation
            self.client.observability.complete_observation_sync(
                observation_id,
                end_time=datetime.now(timezone.utc),
                output_data={"outputs": outputs},
                status_message="success"
            )

            # Clean up
            self._observations.pop(run_id, None)

        except Exception as e:
            logger.error(f"Failed to end chain observation: {e}")

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when chain errors."""
        try:
            run_id = kwargs.get("run_id")
            observation_id = self._observations.get(run_id)

            if observation_id:
                self.client.observability.complete_observation_sync(
                    observation_id,
                    end_time=datetime.now(timezone.utc),
                    status_message=f"error: {str(error)}"
                )

            # Clean up
            self._observations.pop(run_id, None)

        except Exception as e:
            logger.error(f"Failed to handle chain error: {e}")

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts running."""
        try:
            run_id = kwargs.get("run_id")
            parent_run_id = kwargs.get("parent_run_id")

            # Use parent's trace
            trace_id = self._traces.get(parent_run_id)
            if trace_id:
                self._traces[run_id] = trace_id

            if not trace_id:
                logger.warning(f"No trace found for tool run {run_id}")
                return

            # Create observation for tool
            observation = self.client.observability.create_observation_sync(
                trace_id=trace_id,
                name=f"LangChain Tool: {serialized.get('name', 'unknown')}",
                observation_type="span",
                input_data={
                    "input_str": input_str,
                    "serialized": serialized,
                    "kwargs": self._clean_kwargs(kwargs)
                },
                start_time=datetime.now(timezone.utc)
            )

            self._observations[run_id] = observation.id

        except Exception as e:
            logger.error(f"Failed to start tool observation: {e}")

    def on_tool_end(self, output: str, **kwargs) -> None:
        """Called when tool ends running."""
        try:
            run_id = kwargs.get("run_id")
            observation_id = self._observations.get(run_id)

            if not observation_id:
                logger.warning(f"No observation found for tool run {run_id}")
                return

            # Complete observation
            self.client.observability.complete_observation_sync(
                observation_id,
                end_time=datetime.now(timezone.utc),
                output_data={"output": output},
                status_message="success"
            )

            # Clean up
            self._observations.pop(run_id, None)

        except Exception as e:
            logger.error(f"Failed to end tool observation: {e}")

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], **kwargs) -> None:
        """Called when tool errors."""
        try:
            run_id = kwargs.get("run_id")
            observation_id = self._observations.get(run_id)

            if observation_id:
                self.client.observability.complete_observation_sync(
                    observation_id,
                    end_time=datetime.now(timezone.utc),
                    status_message=f"error: {str(error)}"
                )

            # Clean up
            self._observations.pop(run_id, None)

        except Exception as e:
            logger.error(f"Failed to handle tool error: {e}")

    def on_agent_action(self, action, **kwargs) -> None:
        """Called when agent takes an action."""
        try:
            run_id = kwargs.get("run_id")
            parent_run_id = kwargs.get("parent_run_id")

            # Use parent's trace
            trace_id = self._traces.get(parent_run_id)
            if not trace_id:
                return

            # Create observation for agent action
            observation = self.client.observability.create_observation_sync(
                trace_id=trace_id,
                name=f"Agent Action: {action.tool}",
                observation_type="event",
                input_data={
                    "tool": action.tool,
                    "tool_input": action.tool_input,
                    "log": action.log
                },
                start_time=datetime.now(timezone.utc)
            )

            # Complete immediately since actions are atomic
            self.client.observability.complete_observation_sync(
                observation.id,
                end_time=datetime.now(timezone.utc),
                status_message="success"
            )

        except Exception as e:
            logger.error(f"Failed to record agent action: {e}")

    def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when agent finishes."""
        try:
            run_id = kwargs.get("run_id")
            parent_run_id = kwargs.get("parent_run_id")

            # Use parent's trace
            trace_id = self._traces.get(parent_run_id)
            if not trace_id:
                return

            # Create observation for agent finish
            observation = self.client.observability.create_observation_sync(
                trace_id=trace_id,
                name="Agent Finish",
                observation_type="event",
                input_data={
                    "return_values": finish.return_values,
                    "log": finish.log
                },
                start_time=datetime.now(timezone.utc)
            )

            # Complete immediately
            self.client.observability.complete_observation_sync(
                observation.id,
                end_time=datetime.now(timezone.utc),
                status_message="success"
            )

        except Exception as e:
            logger.error(f"Failed to record agent finish: {e}")

    def _extract_provider(self, serialized: Dict[str, Any]) -> Optional[str]:
        """Extract provider from serialized LLM."""
        try:
            llm_type = serialized.get("_type", "").lower()

            if "openai" in llm_type:
                return "openai"
            elif "anthropic" in llm_type or "claude" in llm_type:
                return "anthropic"
            elif "google" in llm_type or "gemini" in llm_type:
                return "google"
            elif "cohere" in llm_type:
                return "cohere"
            elif "huggingface" in llm_type:
                return "huggingface"
            else:
                return llm_type

        except:
            return None

    def _extract_model(self, serialized: Dict[str, Any]) -> Optional[str]:
        """Extract model from serialized LLM."""
        try:
            # Try common model field names
            for field in ["model_name", "model", "engine"]:
                if field in serialized:
                    return serialized[field]

            return None

        except:
            return None

    def _extract_llm_response(self, response: LLMResult) -> Dict[str, Any]:
        """Extract data from LLM response."""
        try:
            response_data = {
                "generations": [],
                "llm_output": response.llm_output or {}
            }

            # Extract generations
            for generation_list in response.generations:
                gen_data = []
                for generation in generation_list:
                    gen_data.append({
                        "text": generation.text,
                        "generation_info": generation.generation_info or {}
                    })
                response_data["generations"].append(gen_data)

            # Extract token usage and cost if available
            llm_output = response.llm_output or {}
            token_usage = llm_output.get("token_usage", {})

            if token_usage:
                response_data["prompt_tokens"] = token_usage.get("prompt_tokens", 0)
                response_data["completion_tokens"] = token_usage.get("completion_tokens", 0)
                response_data["total_tokens"] = token_usage.get("total_tokens", 0)

            # Try to calculate cost if we have model info
            if hasattr(response, 'model_name') or "model" in llm_output:
                model_name = getattr(response, 'model_name', llm_output.get("model", ""))
                cost = self._estimate_cost(model_name, token_usage)
                if cost:
                    response_data["total_cost"] = cost

            return response_data

        except Exception as e:
            logger.warning(f"Failed to extract LLM response: {e}")
            return {"error": f"Failed to extract response: {e}"}

    def _estimate_cost(self, model_name: str, token_usage: Dict[str, int]) -> Optional[float]:
        """Estimate cost based on model and token usage."""
        try:
            if not token_usage or not model_name:
                return None

            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)

            # Simple cost estimation (would need to be more comprehensive)
            model_lower = model_name.lower()

            if "gpt-4" in model_lower:
                return (prompt_tokens / 1000 * 0.03) + (completion_tokens / 1000 * 0.06)
            elif "gpt-3.5" in model_lower:
                return (prompt_tokens / 1000 * 0.001) + (completion_tokens / 1000 * 0.002)
            elif "claude" in model_lower:
                return (prompt_tokens / 1000 * 0.008) + (completion_tokens / 1000 * 0.024)

            return None

        except:
            return None

    def _clean_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Clean kwargs for serialization."""
        try:
            # Remove non-serializable items
            cleaned = {}
            for key, value in kwargs.items():
                if key in ["run_id", "parent_run_id", "tags"]:
                    cleaned[key] = value
                elif isinstance(value, (str, int, float, bool, list, dict)):
                    cleaned[key] = value

            return cleaned

        except:
            return {}


class LangChainInstrumentation:
    """Auto-instrumentation for LangChain library."""

    def __init__(self):
        self.config = get_config()
        self._client = None
        self._callback_handler = None
        self._instrumented = False

    @property
    def client(self):
        """Get or create Brokle client for observability."""
        if self._client is None:
            self._client = get_client()
        return self._client

    def is_available(self) -> bool:
        """Check if LangChain library is available."""
        return LANGCHAIN_AVAILABLE

    def instrument(self) -> bool:
        """Instrument LangChain library for automatic observability."""
        if not self.is_available():
            logger.warning("LangChain library not available for instrumentation")
            return False

        if self._instrumented:
            logger.info("LangChain already instrumented")
            return True

        try:
            # Create callback handler
            self._callback_handler = BrokleCallbackHandler(self.client)

            # Add to global callback manager
            if hasattr(langchain, 'callbacks') and hasattr(langchain.callbacks, 'manager'):
                manager = langchain.callbacks.manager
                if hasattr(manager, 'add_handler'):
                    manager.add_handler(self._callback_handler)

            self._instrumented = True
            logger.info("LangChain instrumentation enabled successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to instrument LangChain: {e}")
            return False

    def uninstrument(self) -> bool:
        """Remove LangChain instrumentation."""
        if not self._instrumented:
            return True

        try:
            # Remove from global callback manager
            if (hasattr(langchain, 'callbacks') and
                hasattr(langchain.callbacks, 'manager') and
                self._callback_handler):

                manager = langchain.callbacks.manager
                if hasattr(manager, 'remove_handler'):
                    manager.remove_handler(self._callback_handler)

            self._callback_handler = None
            self._instrumented = False
            logger.info("LangChain instrumentation removed successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to uninstrument LangChain: {e}")
            return False

    def get_callback_handler(self) -> Optional[BrokleCallbackHandler]:
        """Get the Brokle callback handler for manual use."""
        if not self._instrumented:
            self._callback_handler = BrokleCallbackHandler(self.client)

        return self._callback_handler


# Global instance
_langchain_instrumentation = LangChainInstrumentation()


def instrument_langchain() -> bool:
    """Instrument LangChain library for automatic observability."""
    return _langchain_instrumentation.instrument()


def uninstrument_langchain() -> bool:
    """Remove LangChain instrumentation."""
    return _langchain_instrumentation.uninstrument()


def is_langchain_instrumented() -> bool:
    """Check if LangChain is currently instrumented."""
    return _langchain_instrumentation._instrumented


def get_brokle_callback() -> Optional[BrokleCallbackHandler]:
    """Get Brokle callback handler for manual use with LangChain components."""
    return _langchain_instrumentation.get_callback_handler()