# LMTree

LMTree is an automated feature engineering framework that leverages Large Language Models (LLMs) to generate and optimize features for machine learning tasks. It combines Monte Carlo Tree Search (MCTS) with LLM-powered feature generation to discover high-quality features automatically.

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ACMISLab/LMTree.git
cd LMTree
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure your OpenAI API key in `LMTree/conf/conf.py`:
```python
os.environ["OPENAI_API_KEY"] = "your-api-key-here"
```

## Quick Start

### Basic Usage

```python
import pandas as pd
from LMTree.method.LMTree import LMTree

# Load your dataset
df = pd.read_csv('your_data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Initialize LMTree
lm_tree = LMTree(
    task_type='classification',  # or 'regression'
    max_depth=3,
    max_iterations=50,
    exploration_weight=1.4
)

# Run feature engineering
best_features = lm_tree.run(X, y)
print("Best generated features:", best_features)
```

### Running the Example

The project includes a test script that demonstrates LMTree on sample datasets:

```bash
python test_method.py
```

This will run LMTree on the Australian and Abalone datasets included in the `data/` directory.

## Features

- **Automated Feature Generation**: Uses LLMs to generate meaningful feature combinations
- **MCTS Optimization**: Employs Monte Carlo Tree Search to explore the feature space efficiently  
- **Multi-task Support**: Works with both classification and regression tasks

## Project Structure

```
LMTree/
├── LMTree/
│   ├── method/          # Core algorithms
│   ├── conf/           # Configuration files
│   └── llm/            # LLM integration utilities
├── data/               # Sample datasets
├── test_method.py      # Example usage script
└── requirements.txt    # Dependencies
```
