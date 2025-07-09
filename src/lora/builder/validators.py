from typing import List
from src.core.schema import Example

MIN_EXAMPLES = 3
MAX_CONTEXT_LENGTH = 4096  # Example value, should be configurable

class ValidationError(ValueError):
    pass

def validate_roles(example: Example):
    roles = [msg["role"] for msg in example.messages]
    if not roles:
        raise ValidationError("Example has no messages.")
    
    # Check for valid starting role (system or user)
    if roles[0] not in ["system", "user"]:
        raise ValidationError(f"Invalid starting role: {roles[0]}")

    # Check for alternating user/assistant roles
    for i in range(1, len(roles)):
        if roles[i] == roles[i-1]:
            raise ValidationError(f"Consecutive roles found: {roles[i]}")
        if roles[i] not in ["user", "assistant"]:
            raise ValidationError(f"Invalid role in conversation: {roles[i]}")

def validate_content_length(example: Example):
    for message in example.messages:
        if len(message.get("content", "")) > MAX_CONTEXT_LENGTH:
            raise ValidationError("Message content exceeds max context length.")

def validate_examples(examples: List[Example]):
    if len(examples) < MIN_EXAMPLES:
        raise ValidationError(f"Requires at least {MIN_EXAMPLES} examples, found {len(examples)}.")
    
    for i, example in enumerate(examples):
        try:
            validate_roles(example)
            validate_content_length(example)
        except ValidationError as e:
            raise ValidationError(f"Error in example {i}: {e}") from e