# IFU Comparative Analysis System

> Advanced agentic platform for comprehensive IFU (Instructions For Use) comparison using **Claude Sonnet 4.5** with native vision and **LangGraph 1.0**.

## 🌟 Features

### Core Capabilities

- **🔍 Native PDF Processing** - Leverages Claude Sonnet 4.5's native vision to process PDFs directly, capturing text, layout, images, and formatting
- **📊 Comprehensive Diff Detection** - Detects ALL changes including:
  - Text additions, deletions, and modifications
  - Visual changes in diagrams and images
  - Table structure and content changes
  - Warning and safety information changes
  - Numerical and measurement changes (critical for medical devices)
- **📝 Professional Word Reports** - Generates formatted Word documents with:
  - Side-by-side comparison (bicolunado)
  - Color-coded changes (red=removed, green=added)
  - Executive summary with statistics
  - Detailed differences table
  - Organized by sections and pages
  - Table of contents
- **✅ Progress Tracking** - Real-time checklist showing analysis progress
- **👤 Human-in-the-Loop** - Pauses for review when critical changes detected
- **💾 Persistent Checkpointing** - Resume long-running analyses after interruptions
- **🔄 Retry Policies** - Automatic retry for transient failures

### Built with Best Practices

Following **LangGraph 1.0** and **Checkpoints 3.0** best practices:

- ✅ StateGraph with typed state schemas
- ✅ Annotated fields with appropriate reducers
- ✅ Conditional edges for flow control
- ✅ SQLite/PostgreSQL checkpointing for durability
- ✅ Command API for state updates + routing
- ✅ Interrupt API for human review
- ✅ Comprehensive error handling
- ✅ Streaming for progress updates

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    IFU Analysis Workflow                     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │  Initialize   │ ─── Validate PDFs
                      └───────┬───────┘     Create checklist
                              │
                              ▼
                      ┌───────────────┐
                      │   Extract     │ ─── Extract via Claude vision
                      └───────┬───────┘     Parse structure & content
                              │
                              ▼
                      ┌───────────────┐
                      │   Analyze     │ ─── Organize sections
                      └───────┬───────┘     Build document tree
                              │
                              ▼
                      ┌───────────────┐
                      │   Compare     │ ─── Multi-level comparison:
                      └───────┬───────┘     • Section-level
                              │             • Page-by-page visual
                              │             • Text-level granular
                              │             • Missing content
                              ▼
                    ┌─────────────────┐
                    │ Critical changes?│
                    └────┬──────┬─────┘
                      Yes│      │No
                         ▼      │
                    ┌────────┐  │
                    │ Review │  │ ◄── Human-in-the-loop
                    └───┬────┘  │
                        │       │
                        └───┬───┘
                            ▼
                      ┌───────────────┐
                      │   Generate    │ ─── Create Word report
                      └───────┬───────┘     Format side-by-side
                              │             Color-code changes
                              ▼
                      ┌───────────────┐
                      │  Finalize     │ ─── Complete!
                      └───────────────┘
```

## 📁 Project Structure

```
ifu-comparative-analysis/
├── README.md                          # This file
├── pyproject.toml                     # Python project configuration
├── langgraph.json                     # LangGraph CLI configuration
├── .env.example                       # Environment variables template
│
├── src/                               # Source code
│   ├── __init__.py
│   ├── state.py                       # State schemas (TypedDict)
│   ├── graph.py                       # Main StateGraph definition
│   │
│   ├── nodes/                         # Graph nodes
│   │   ├── __init__.py
│   │   ├── initialize.py              # Initialization & validation
│   │   ├── extract.py                 # PDF extraction
│   │   ├── analyze.py                 # Structure analysis
│   │   ├── compare.py                 # Difference detection
│   │   ├── generate.py                # Report generation
│   │   └── checklist.py               # Progress tracking
│   │
│   ├── tools/                         # LangChain tools
│   │   ├── __init__.py
│   │   ├── pdf_tools.py               # PDF processing (Claude vision)
│   │   ├── diff_tools.py              # Diff detection (Claude analysis)
│   │   └── word_tools.py              # Word report generation
│   │
│   └── utils/                         # Utilities
│       ├── __init__.py
│       └── config.py                  # Configuration management
│
├── examples/                          # Usage examples
│   ├── basic_comparison.py            # Basic usage
│   └── sample_ifus/                   # Sample PDF files (place yours here)
│
├── tests/                             # Test suite
│   └── __init__.py
│
└── reports/                           # Generated reports (auto-created)
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or navigate to the project
cd ifu-comparative-analysis

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For PostgreSQL support
pip install -e ".[postgres]"
```

### 2. Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env and add your Anthropic API key
# ANTHROPIC_API_KEY=your_key_here
```

### 3. Basic Usage

```python
from src import run_ifu_analysis

# Run analysis
result = run_ifu_analysis(
    old_pdf_path="path/to/old_ifu.pdf",
    new_pdf_path="path/to/new_ifu.pdf",
    enable_human_review=True,
    checkpoint_type="sqlite"
)

# Check result
if result["status"] == "completed":
    print(f"Report: {result['report_path']}")
    print(f"Differences: {len(result['differences'])}")
```

### 4. Run Example

```bash
# Place your IFU PDFs in examples/sample_ifus/
# Name them: ifu_v1.0.pdf (old) and ifu_v2.0.pdf (new)

# Run example
python examples/basic_comparison.py
```

## 📖 Detailed Usage

### Complete Analysis with All Features

```python
from src import run_ifu_analysis, resume_analysis

# Run analysis with all features enabled
result = run_ifu_analysis(
    old_pdf_path="examples/sample_ifus/ifu_v1.0.pdf",
    new_pdf_path="examples/sample_ifus/ifu_v2.0.pdf",
    analysis_id="my_analysis_001",     # Optional: specify ID
    enable_human_review=True,           # Pause for critical changes
    checkpoint_type="sqlite"            # Persist state
)

# If human review is needed
if result["status"] == "awaiting_review":
    print(f"Review required for analysis: {result['analysis_id']}")

    # User reviews changes...

    # Resume with approval
    final_result = resume_analysis(
        analysis_id=result["analysis_id"],
        approved=True,
        notes="Reviewed by John Doe - all changes acceptable"
    )

    print(f"Report: {final_result['report_path']}")
```

### Custom Configuration

```python
from src.utils.config import load_config
from src import run_ifu_analysis

# Load and customize config
config = load_config()
config["max_pages_to_compare"] = 50
config["enable_human_review"] = False

# Run with custom config
result = run_ifu_analysis(
    old_pdf_path="old.pdf",
    new_pdf_path="new.pdf",
    enable_human_review=config["enable_human_review"]
)
```

### Streaming Progress Updates

```python
from src.graph import create_analysis_graph
from src.state import AnalysisState
from datetime import datetime

# Create graph
graph = create_analysis_graph(checkpointer_type="sqlite")

# Initial state
initial_state = AnalysisState(
    old_pdf_path="old.pdf",
    new_pdf_path="new.pdf",
    analysis_id="streaming_example",
    started_at=datetime.now(),
    current_step="initialized",
    status="initialized",
    report_status="pending",
    requires_review=False,
    differences=[],
    checklist=[],
    errors=[]
)

# Stream progress
config = {"configurable": {"thread_id": "streaming_example"}}

for event in graph.stream(initial_state, config, stream_mode="updates"):
    for node_name, output in event.items():
        print(f"[{node_name}] {output['current_step']}")

        # Show checklist
        checklist = output.get("checklist", [])
        completed = sum(1 for item in checklist if item["status"] == "completed")
        print(f"  Progress: {completed}/{len(checklist)}")
```

## 🎯 Use Cases

### Medical Device IFU Comparison

Perfect for regulatory compliance when updating IFU documents:

- **Pre-submission Review** - Identify all changes before regulatory submission
- **Change Impact Analysis** - Assess whether changes require regulatory notification
- **Quality Assurance** - Ensure no critical information was accidentally removed
- **Version Control** - Track document evolution over time

### Key Features for Medical Devices

- **Safety-Critical Detection** - Automatically flags changes to warnings, contraindications, and adverse events
- **Measurement Validation** - Detects any changes to numerical values (doses, measurements, specifications)
- **Regulatory Readiness** - Report format suitable for regulatory documentation
- **Audit Trail** - Complete checkpoint history for compliance

## 🛠️ Advanced Features

### Checkpoint Persistence

```python
# SQLite (default) - single file, good for single machine
result = run_ifu_analysis(
    ...,
    checkpoint_type="sqlite"
)

# PostgreSQL - for production, multi-machine deployments
result = run_ifu_analysis(
    ...,
    checkpoint_type="postgres"
)

# In-memory - testing only
result = run_ifu_analysis(
    ...,
    checkpoint_type="memory"
)
```

### Accessing State History

```python
from src.graph import create_analysis_graph

graph = create_analysis_graph()
config = {"configurable": {"thread_id": "my_analysis"}}

# Get current state
state = graph.get_state(config)
print(state.values)
print(state.next)  # Next nodes to execute

# Get state history
history = list(graph.get_state_history(config))
for checkpoint in history:
    print(f"Step {checkpoint.metadata['step']}: {checkpoint.values['current_step']}")
```

### Error Handling

The system includes comprehensive error handling:

- **Validation Errors** - PDF file validation before processing
- **Extraction Errors** - Graceful handling of PDF parsing issues
- **API Errors** - Automatic retry for Claude API calls
- **Report Generation Errors** - Detailed error messages for debugging

All errors are collected in the `errors` field of the state:

```python
result = run_ifu_analysis(...)

if result["status"] == "failed":
    print("Errors encountered:")
    for error in result.get("errors", []):
        print(f"  - {error}")
```

## 📊 Report Output

The generated Word report includes:

### 1. Title Page
- Report title
- Generation date
- Company information

### 2. Table of Contents
- Auto-generated TOC
- Hyperlinked sections

### 3. Executive Summary
- Change statistics (total, by severity)
- Critical highlights
- Overall assessment
- Regulatory implications

### 4. Section-by-Section Comparison
- **Two-column layout** (old | new)
- **Color coding**:
  - 🔴 Red strikethrough = Removed
  - 🟢 Green bold = Added
  - ⚫ Black = Unchanged
- Change list below each section
- Page references

### 5. Detailed Differences Table
- Complete list of all changes
- Filterable by severity
- Section references
- Page numbers

### 6. Footer
- Page numbers
- Report identification

## 🔧 Configuration Options

### Environment Variables

See `.env.example` for all available options:

```bash
# Model Configuration
MODEL_NAME=claude-sonnet-4.5          # Claude model to use
MODEL_TEMPERATURE=0.0                  # Temperature (0 = deterministic)
MODEL_MAX_TOKENS=16000                 # Max tokens per request

# Analysis Configuration
ENABLE_HUMAN_REVIEW=true               # Enable human-in-the-loop
SEVERITY_THRESHOLD=all                 # all, major, critical

# Report Configuration
OUTPUT_DIR=./reports                   # Report output directory
INCLUDE_TOC=true                       # Include table of contents
INCLUDE_SUMMARY=true                   # Include executive summary

# Processing
MAX_PAGES_TO_COMPARE=100              # Maximum pages to compare
```

## 🧪 Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/test_graph.py
```

## 📈 Performance

- **Speed**: ~2-5 minutes for typical IFU (20-50 pages)
- **Accuracy**: Leverages Claude Sonnet 4.5's vision for maximum accuracy
- **Scalability**: Handles documents up to 100+ pages
- **Cost**: ~$0.50-2.00 per comparison (depends on document size)

## 🔒 Security

- **API Keys**: Stored in environment variables, never in code
- **Encryption**: Optional encryption for checkpoint data
- **Data Privacy**: All processing via Anthropic's API (see their privacy policy)
- **Audit Trail**: Complete state history for compliance

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Run `pytest` and ensure all tests pass
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Built with:
- **LangGraph 1.0** - Orchestration framework
- **Claude Sonnet 4.5** - AI with native PDF vision
- **python-docx** - Word document generation
- **Anthropic SDK** - Claude API access

## 📞 Support

For issues, questions, or feature requests:
- Open an issue on GitHub
- Contact: ifu-analysis@example.com

## 🗺️ Roadmap

Future enhancements planned:
- [ ] Multi-language support (beyond English)
- [ ] Custom report templates
- [ ] PDF output format
- [ ] Batch processing (multiple document pairs)
- [ ] Web UI for non-technical users
- [ ] Integration with document management systems
- [ ] Advanced filtering and search in reports

---

**Made with ❤️ using Claude Sonnet 4.5 and LangGraph 1.0**
