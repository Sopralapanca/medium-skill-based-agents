Use python 3.12

## Environment Setup

1. **Install UV Package Manager**
```bash
# Install uv (recommended Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Setup Environment**
```bash
# Create and activate environment
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .
```