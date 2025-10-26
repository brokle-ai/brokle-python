"""
Wrapper configuration validation utilities.

Provides shared validation logic for all AI provider wrapper functions
to ensure consistent configuration validation across the SDK.
"""

from ..exceptions import ValidationError


def validate_wrapper_config(**config) -> None:
    """
    Validate wrapper configuration parameters.

    Args:
        **config: Configuration parameters to validate

    Raises:
        ValidationError: If any configuration parameter is invalid

    Validates:
        - capture_content, capture_metadata: Must be boolean
        - tags: Must be list of strings (max 50 chars each)
        - session_id, user_id: Must be strings (max 100 chars)
    """
    # Validate boolean parameters
    bool_params = ["capture_content", "capture_metadata"]
    for param in bool_params:
        if param in config and not isinstance(config[param], bool):
            raise ValidationError(
                f"{param} must be a boolean, got {type(config[param]).__name__}"
            )

    # Validate tags
    if "tags" in config and config["tags"] is not None:
        tags = config["tags"]
        if not isinstance(tags, list):
            raise ValidationError(f"tags must be a list, got {type(tags).__name__}")

        for i, tag in enumerate(tags):
            if not isinstance(tag, str):
                raise ValidationError(
                    f"tags[{i}] must be a string, got {type(tag).__name__}"
                )

            if len(tag) > 50:
                raise ValidationError(
                    f"tags[{i}] must be <= 50 characters, got {len(tag)}"
                )

    # Validate session_id
    if "session_id" in config and config["session_id"] is not None:
        session_id = config["session_id"]
        if not isinstance(session_id, str):
            raise ValidationError(
                f"session_id must be a string, got {type(session_id).__name__}"
            )

        if len(session_id) > 100:
            raise ValidationError(
                f"session_id must be <= 100 characters, got {len(session_id)}"
            )

    # Validate user_id
    if "user_id" in config and config["user_id"] is not None:
        user_id = config["user_id"]
        if not isinstance(user_id, str):
            raise ValidationError(
                f"user_id must be a string, got {type(user_id).__name__}"
            )

        if len(user_id) > 100:
            raise ValidationError(
                f"user_id must be <= 100 characters, got {len(user_id)}"
            )
