# Contributing to IFU Comparative Analysis System

Thank you for your interest in contributing! This document provides guidelines and instructions for contributing to the project.

## 🌟 Ways to Contribute

- **Bug Reports** - Report issues you encounter
- **Feature Requests** - Suggest new features or improvements
- **Code Contributions** - Submit bug fixes or new features
- **Documentation** - Improve or expand documentation
- **Testing** - Add test coverage or test on different platforms

## 🚀 Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then clone your fork
git clone https://github.com/YOUR_USERNAME/ifu-comparative-analysis.git
cd ifu-comparative-analysis
```

### 2. Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

### 3. Create a Branch

```bash
# Create a feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/issue-description
```

## 🧪 Development Workflow

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_graph.py

# Run specific test
pytest tests/test_graph.py::test_create_analysis_graph

# Run with verbose output
pytest -v -s
```

### Code Quality

```bash
# Format code with black
black src/ tests/ examples/

# Lint with ruff
ruff check src/ tests/ examples/

# Type check with mypy
mypy src/
```

### Pre-commit Checks

Before committing, ensure:

1. ✅ All tests pass
2. ✅ Code is formatted with black
3. ✅ No linting errors from ruff
4. ✅ Type checking passes with mypy
5. ✅ New code has tests
6. ✅ Documentation is updated

## 📝 Code Style Guidelines

### Python Style

- **Line length**: 100 characters (configured in pyproject.toml)
- **Docstrings**: Google-style docstrings for all public functions
- **Type hints**: Use type hints for all function signatures
- **Naming**:
  - `snake_case` for functions and variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants

### Example

```python
"""
Module docstring explaining the purpose.
"""

from typing import Optional, List
from ..state import AnalysisState


def process_document(
    document_path: str,
    analysis_type: str = "full",
    max_pages: Optional[int] = None
) -> dict:
    """
    Process a document and return analysis results.

    Args:
        document_path: Path to the document file
        analysis_type: Type of analysis ("full" or "quick")
        max_pages: Optional maximum number of pages to process

    Returns:
        Dictionary containing analysis results

    Raises:
        FileNotFoundError: If document_path does not exist
        ValueError: If analysis_type is invalid
    """
    # Implementation
    pass
```

## 🏗️ Architecture Guidelines

### LangGraph Best Practices

When adding or modifying graph nodes:

1. **State Updates** - Return only modified fields, not full state
2. **Type Safety** - Use TypedDict for state schemas
3. **Reducers** - Use appropriate reducers (operator.add for lists)
4. **Error Handling** - Catch exceptions and add to errors list
5. **Logging** - Use print statements for user feedback

Example node:

```python
def my_node(state: AnalysisState) -> dict:
    """
    Process something and update state.

    Args:
        state: Current analysis state

    Returns:
        Partial state updates
    """
    errors = []

    try:
        # Do processing
        result = process_something(state["input"])

        return {
            "output": result,
            "status": "success",
            "current_step": "processing_complete"
        }

    except Exception as e:
        errors.append(f"Processing failed: {str(e)}")
        return {
            "status": "failed",
            "errors": errors
        }
```

### Tools Best Practices

When adding new tools:

1. **Decorator** - Use `@tool` decorator
2. **Docstrings** - Clear docstrings explaining purpose and parameters
3. **Type Hints** - Full type hints for inputs and outputs
4. **Error Handling** - Graceful error handling with try/except
5. **Claude Integration** - Use Claude Sonnet 4.5 for complex analysis

Example tool:

```python
from langchain_core.tools import tool
from anthropic import Anthropic

@tool
def analyze_section(section_text: str, analysis_type: str) -> dict:
    """
    Analyze a document section using Claude.

    Args:
        section_text: Text content of the section
        analysis_type: Type of analysis to perform

    Returns:
        Dictionary with analysis results
    """
    client = Anthropic()

    response = client.messages.create(
        model="claude-sonnet-4.5",
        max_tokens=4000,
        messages=[{
            "role": "user",
            "content": f"Analyze this section: {section_text}"
        }]
    )

    return {"analysis": response.content[0].text}
```

## 🐛 Reporting Bugs

### Before Reporting

1. Check if the bug has already been reported in Issues
2. Ensure you're using the latest version
3. Try to isolate the bug (minimal reproduction case)

### Bug Report Should Include

- **Description** - Clear description of the bug
- **Steps to Reproduce** - Minimal steps to reproduce the issue
- **Expected Behavior** - What you expected to happen
- **Actual Behavior** - What actually happened
- **Environment**:
  - Python version
  - LangGraph version
  - Operating system
- **Error Messages** - Full error messages and stack traces
- **Sample Files** - If possible, sample files that trigger the bug

## 💡 Feature Requests

Feature requests are welcome! Please include:

- **Use Case** - Why is this feature needed?
- **Proposed Solution** - How do you envision it working?
- **Alternatives** - Other approaches you've considered
- **Additional Context** - Any other relevant information

## 📥 Pull Request Process

### 1. Prepare Your PR

- Ensure all tests pass
- Add tests for new functionality
- Update documentation as needed
- Follow code style guidelines
- Keep commits focused and atomic

### 2. PR Description

Include in your PR description:

- **Summary** - Brief description of changes
- **Motivation** - Why is this change needed?
- **Changes** - List of specific changes made
- **Testing** - How was this tested?
- **Screenshots** - If UI changes (if applicable)

### 3. Review Process

- Maintainers will review your PR
- Address any feedback or requested changes
- Once approved, your PR will be merged

### 4. After Merge

- Delete your feature branch
- Pull the latest main branch
- Your contribution will be acknowledged in release notes

## 🎯 Priority Areas

Areas where contributions are especially welcome:

1. **Test Coverage** - Expand test suite
2. **Documentation** - Improve or add documentation
3. **Error Handling** - Better error messages and recovery
4. **Performance** - Optimize for speed and cost
5. **Multi-language Support** - IFU analysis in other languages
6. **Report Templates** - Additional report formats or templates

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🤝 Code of Conduct

### Our Pledge

We pledge to make participation in this project a harassment-free experience for everyone.

### Our Standards

- **Be respectful** - Respect differing viewpoints and experiences
- **Be constructive** - Provide constructive feedback
- **Be collaborative** - Work together towards common goals
- **Be inclusive** - Welcome newcomers and help them learn

## 📞 Questions?

If you have questions about contributing:

- Open a Discussion on GitHub
- Email: ifu-analysis@example.com

## 🙏 Thank You!

Thank you for contributing to make IFU Comparative Analysis better!
