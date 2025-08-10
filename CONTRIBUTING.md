# Contributing to Brokle Python SDK

Thank you for your interest in contributing to the Brokle Python SDK! We welcome contributions from the community and are excited to work with you.

## Table of Contents
- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Commit Message Guidelines](#commit-message-guidelines)
- [Pull Request Process](#pull-request-process)
- [Testing](#testing)
- [Code Style](#code-style)

## Code of Conduct

By participating in this project, you agree to abide by our Code of Conduct. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites
- Python 3.8 or higher
- pip or poetry for dependency management
- Git

### Development Setup

1. **Fork and clone the repository**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/brokle-python.git
   cd brokle-python
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**:
   ```bash
   make test
   ```

## Making Changes

### Creating a Branch
Create a descriptive branch name:
```bash
git checkout -b feature/semantic-caching-support
git checkout -b fix/authentication-timeout
git checkout -b docs/update-installation-guide
```

### Development Process
1. Make your changes in small, logical commits
2. Write or update tests for your changes
3. Update documentation if needed
4. Ensure all tests pass
5. Run code quality checks

### Testing Your Changes
```bash
# Run all tests
make test

# Run tests with coverage
make test-coverage

# Run specific test file
make test-specific TEST=test_config.py

# Run integration tests
make integration-test
```

### Code Quality Checks
```bash
# Format code
make format

# Run linter
make lint

# Type checking
make type-check

# Full development check
make dev-check
```

## Commit Message Guidelines

We appreciate well-formatted commit messages! While not strictly enforced, following these guidelines helps maintain a clean project history and makes it easier for maintainers to understand your changes.

### Preferred Format
```
<type>: <description>

[optional body]
```

### Types
- **`feat`**: New features or enhancements
- **`fix`**: Bug fixes
- **`docs`**: Documentation changes
- **`test`**: Adding or updating tests
- **`refactor`**: Code improvements without changing functionality
- **`perf`**: Performance improvements
- **`style`**: Code formatting or style changes
- **`chore`**: Maintenance tasks, dependency updates

### Examples
```bash
# Good examples
feat: add semantic caching support
fix: resolve authentication timeout issue
docs: update installation guide with new requirements
test: add unit tests for OpenAI compatibility layer
refactor: simplify error handling logic
perf: optimize request batching for better throughput

# Also acceptable (we're flexible!)
Add semantic caching feature
Fixed timeout bug in auth
Update docs
```

### Tips for Good Commit Messages
- Use imperative mood ("add" not "added")
- Keep the first line under 50 characters when possible
- Be descriptive but concise
- Explain **what** and **why**, not **how**

**Don't worry if your commit doesn't follow this format perfectly** - we're happy to help improve it during the review process! The most important thing is clear communication about your changes.

## Pull Request Process

### Before Submitting
1. **Ensure tests pass**: Run `make dev-check`
2. **Update documentation**: Add or update docstrings, README, etc.
3. **Add tests**: For new features or bug fixes
4. **Check for breaking changes**: Note any breaking changes in PR description

### Submitting Your PR

1. **Push your branch**:
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create a Pull Request** with:
   - Clear title describing the change
   - Detailed description of what was changed and why
   - Reference any related issues
   - Note any breaking changes
   - Screenshots or examples if applicable

3. **PR Template**:
   ```markdown
   ## Description
   Brief description of changes

   ## Type of Change
   - [ ] Bug fix (non-breaking change which fixes an issue)
   - [ ] New feature (non-breaking change which adds functionality)
   - [ ] Breaking change (fix or feature that would cause existing functionality to change)
   - [ ] Documentation update

   ## How Has This Been Tested?
   - [ ] Unit tests
   - [ ] Integration tests
   - [ ] Manual testing

   ## Checklist
   - [ ] My code follows the code style of this project
   - [ ] I have performed a self-review of my code
   - [ ] I have commented my code, particularly in hard-to-understand areas
   - [ ] I have made corresponding changes to the documentation
   - [ ] My changes generate no new warnings
   - [ ] I have added tests that prove my fix is effective or that my feature works
   - [ ] New and existing unit tests pass locally with my changes
   ```

### Review Process
- Maintainers will review your PR and provide feedback
- You may be asked to make changes or provide clarification
- Once approved, a maintainer will merge your PR
- We may clean up commit messages during the merge process

## Testing

### Test Structure
- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows

### Writing Tests
```python
# Example unit test
def test_config_validation():
    config = Config(api_key="test_key", host="http://test.com")
    assert config.api_key == "test_key"
    assert config.host == "http://test.com"

# Example integration test  
@pytest.mark.asyncio
async def test_chat_completion():
    async with Brokle() as client:
        response = await client.chat.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "Hello"}]
        )
        assert response.choices[0].message.content
```

### Test Coverage
We aim for high test coverage. When adding new features:
- Write tests for happy path scenarios
- Write tests for edge cases and error conditions
- Test both sync and async code paths where applicable

## Code Style

### Python Style Guidelines
We follow PEP 8 with some modifications:
- Line length: 88 characters (Black default)
- Use type hints for all public functions
- Docstrings for all public modules, classes, and functions

### Formatting Tools
We use these tools to maintain consistent code style:
- **Black**: Code formatter
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

These run automatically with `make format` and `make lint`.

### Docstring Format
Use Google-style docstrings:
```python
def create_completion(
    self, 
    model: str, 
    messages: List[Dict[str, str]]
) -> CompletionResponse:
    """Create a chat completion using the Brokle Platform.
    
    Args:
        model: The model to use for completion
        messages: List of message objects
        
    Returns:
        CompletionResponse: The completion response
        
    Raises:
        BrokleError: If the request fails
        AuthenticationError: If authentication fails
    """
```

## Getting Help

### Resources
- [Documentation](https://docs.brokle.com/sdk/python)
- [Examples](./examples/)
- [API Reference](https://docs.brokle.com/api)

### Communication
- **Issues**: Use GitHub issues for bug reports and feature requests
- **Discussions**: Use GitHub discussions for questions and general discussion
- **Email**: For sensitive issues, contact support@brokle.com

### Common Questions

**Q: How do I run tests for a specific file?**
A: Use `make test-specific TEST=test_filename.py`

**Q: My PR failed CI checks, what do I do?**
A: Run `make dev-check` locally to see the same checks that run in CI

**Q: Can I add a new dependency?**
A: Yes, but please discuss in an issue first for significant dependencies

**Q: How do I update documentation?**
A: Update docstrings in code and relevant README sections

## Recognition

Contributors will be recognized in:
- The project's contributor list
- Release notes for significant contributions
- Special recognition for first-time contributors

Thank you for contributing to Brokle! 🚀