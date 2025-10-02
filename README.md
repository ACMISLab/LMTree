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

3. Configure your OpenAI API settings in `LMTree/conf/conf.py`:
```python
import os

# OpenAI API configuration
os.environ["OPENAI_BASE_URL"] = "https://api.openai-proxy.org/v1"  # Configure API base URL
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # Configure your API key
model = "gpt-4o-mini"  # Configure the model to use
```

## Quick Start

### Running the Example

The project includes a test script that demonstrates how to use LMTree on sample datasets:

1. First, ensure you have properly configured the API key and URL in `LMTree/conf/conf.py`
2. Run the test script directly:

```bash
python test_method.py
```

This script will run LMTree on the Australian dataset in the `data/` directory. You can switch to other datasets (such as "abalone", "boston", etc.) by modifying the `dataName` variable at the beginning of the `test_method.py` file.

### Custom Usage

To use LMTree on your own dataset, you need to prepare the following files:

1. **CSV data file**: Contains features and target columns
2. **JSON configuration file**: Contains dataset metadata in the following format:

```json
{
    "dataset_name": "your_dataset_name",
    "type": true,  // true for classification tasks, false for regression tasks
    "num_samples": 690,
    "description": "Dataset description",
    "attribute_introduction": {
        "feature1": "Description of feature 1",
        "feature2": "Description of feature 2"
    },
    "target": "target_column_name",
    "is_categorical": [false, true, false]  // Indicates whether each feature is categorical
}
```

Then use the following code:

```python
import json
import pandas as pd
from LMTree.method.LMTree import LMTree

# Load data and configuration
df = pd.read_csv('your_data.csv')
with open('your_config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)

# Initialize LMTree
lm_tree = LMTree(
    df=df,
    target_column_name=config['target'],
    dataName=config['dataset_name'],
    attribute_introduction=config['attribute_introduction'],
    is_categorical=config['is_categorical'],
    taskType="classification" if config['type'] else "regression",
    content_desc=config['description'],
    max_iterations=50,  # Adjustable parameter
    max_depth=3
)

# Run feature engineering
result_data = lm_tree.run()
print("Feature engineering completed, generated dataset:", result_data.columns.tolist())
```

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
