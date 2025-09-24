"""
@observe Decorator Usage Example

This example demonstrates the @observe decorator for
comprehensive observability and tracing.
"""

import asyncio
from typing import List, Dict, Any

# ✨ PATTERN 2: Universal Decorator - AI-aware observability
from brokle import observe, Brokle
from openai import OpenAI

# Initialize Brokle client (auto-registers for @observe decorators)
client = Brokle(
    api_key="ak_your_api_key_here",
    host="http://localhost:8080",
    project_id="proj_your_project_id",
)

# Initialize standard OpenAI client (will be auto-detected by @observe decorator)
openai = OpenAI()


@observe()
def simple_function():
    """Basic function observability."""
    return "Hello, World!"


@observe(name="custom-function-name")
def custom_named_function():
    """Function with custom observation name."""
    return "Custom named function result"


@observe(
    name="llm-generation",
    as_type="generation",
    capture_input=True,
    capture_output=True
)
def generate_story(prompt: str) -> str:
    """Generate a story using LLM with generation-type observation."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a creative storyteller."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200,
        routing_strategy="quality_optimized",
        custom_tags={"type": "story", "genre": "any"}
    )
    
    return response.choices[0].message.content


@observe(
    user_id="user123",
    session_id="session456",
    tags=["translation", "api-call"],
    metadata={"version": "1.0", "feature": "translation"}
)
def translate_text(text: str, target_language: str) -> str:
    """Translate text with user and session tracking."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Translate the following text to {target_language}."},
            {"role": "user", "content": text}
        ],
        routing_strategy="cost_optimized",
        custom_tags={"operation": "translation", "target_lang": target_language}
    )
    
    return response.choices[0].message.content


@observe(capture_input=True, capture_output=True)
def process_documents(documents: List[str]) -> Dict[str, Any]:
    """Process multiple documents with input/output capture."""
    summaries = []
    
    for doc in documents:
        summary = summarize_document(doc)
        summaries.append(summary)
    
    return {
        "total_documents": len(documents),
        "summaries": summaries,
        "average_length": sum(len(s) for s in summaries) / len(summaries)
    }


@observe(name="document-summarization", as_type="generation")
def summarize_document(document: str) -> str:
    """Summarize a document - nested observation."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Summarize the following document concisely."},
            {"role": "user", "content": document}
        ],
        max_tokens=150,
        routing_strategy="balanced",
        custom_tags={"operation": "summarization"}
    )
    
    return response.choices[0].message.content


@observe()
async def async_function():
    """Async function with observability."""
    await asyncio.sleep(0.1)  # Simulate async work
    return "Async result"


@observe(as_type="generation")
async def async_generate_code(requirement: str) -> str:
    """Generate code asynchronously with generation tracking."""
    from openai import AsyncOpenAI
    async_openai = AsyncOpenAI()

    response = await async_openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert programmer. Write clean, well-documented code."},
            {"role": "user", "content": f"Write Python code for: {requirement}"}
        ],
        max_tokens=500,
        routing_strategy="quality_optimized",
        custom_tags={"operation": "code_generation", "language": "python"}
    )
    
    return response.choices[0].message.content


@observe(capture_input=False, capture_output=True)
def sensitive_operation(api_key: str, user_data: Dict[str, Any]) -> str:
    """Operation with sensitive input - don't capture input."""
    # Process sensitive data
    processed_data = f"Processed {len(user_data)} fields"
    return processed_data


@observe()
def error_prone_function():
    """Function that demonstrates error handling in observations."""
    raise ValueError("This is a test error for demonstration")


@observe(
    name="complex-workflow",
    user_id="user789",
    session_id="session999",
    tags=["workflow", "multi-step"],
    metadata={"complexity": "high", "steps": 3}
)
def complex_workflow(input_data: str) -> Dict[str, Any]:
    """Complex workflow with multiple steps and comprehensive tracking."""
    
    # Step 1: Process input
    processed = process_input(input_data)
    
    # Step 2: Generate content
    content = generate_content(processed)
    
    # Step 3: Validate and format
    result = validate_and_format(content)
    
    return {
        "input": input_data,
        "processed": processed,
        "content": content,
        "result": result,
        "workflow_completed": True
    }


@observe(name="input-processing")
def process_input(data: str) -> str:
    """Process input data - step 1 of workflow."""
    return data.upper().strip()


@observe(name="content-generation", as_type="generation")
def generate_content(processed_data: str) -> str:
    """Generate content based on processed data - step 2 of workflow."""
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Generate creative content based on the input."},
            {"role": "user", "content": processed_data}
        ],
        max_tokens=100,
        routing_strategy="balanced",
        custom_tags={"step": "content_generation"}
    )
    
    return response.choices[0].message.content


@observe(name="validation-formatting")
def validate_and_format(content: str) -> str:
    """Validate and format content - step 3 of workflow."""
    # Simple validation and formatting
    if len(content) < 10:
        raise ValueError("Content too short")
    
    return content.strip().capitalize()


def main():
    """Main function demonstrating various observation patterns."""
    print("=== @observe Decorator Examples ===\n")
    
    # Simple function
    print("1. Simple Function:")
    result = simple_function()
    print(f"Result: {result}\n")
    
    # Custom named function
    print("2. Custom Named Function:")
    result = custom_named_function()
    print(f"Result: {result}\n")
    
    # LLM generation with observation
    print("3. LLM Generation with Observation:")
    try:
        story = generate_story("A robot discovers emotions")
        print(f"Generated story: {story[:100]}...\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Translation with user/session tracking
    print("4. Translation with Tracking:")
    try:
        translated = translate_text("Hello, how are you?", "Spanish")
        print(f"Translated: {translated}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Document processing
    print("5. Document Processing:")
    try:
        documents = [
            "This is a sample document about machine learning.",
            "Another document discussing artificial intelligence.",
            "A third document about natural language processing."
        ]
        result = process_documents(documents)
        print(f"Processing result: {result}\n")
    except Exception as e:
        print(f"Error: {e}\n")
    
    # Sensitive operation
    print("6. Sensitive Operation (input not captured):")
    result = sensitive_operation("secret_token", {"name": "John", "email": "john@example.com"})
    print(f"Result: {result}\n")
    
    # Error handling
    print("7. Error Handling:")
    try:
        error_prone_function()
    except ValueError as e:
        print(f"Caught error: {e}\n")
    
    # Complex workflow
    print("8. Complex Workflow:")
    try:
        result = complex_workflow("process this data")
        print(f"Workflow result: {result}\n")
    except Exception as e:
        print(f"Error: {e}\n")


async def async_main():
    """Async main function demonstrating async observations."""
    print("=== Async @observe Examples ===\n")
    
    # Simple async function
    print("1. Async Function:")
    result = await async_function()
    print(f"Result: {result}\n")
    
    # Async code generation
    print("2. Async Code Generation:")
    try:
        code = await async_generate_code("A function to calculate fibonacci numbers")
        print(f"Generated code: {code[:200]}...\n")
    except Exception as e:
        print(f"Error: {e}\n")


if __name__ == "__main__":
    # Run synchronous examples
    main()
    
    # Run asynchronous examples
    asyncio.run(async_main())
    
    print("=== All examples completed! ===")
