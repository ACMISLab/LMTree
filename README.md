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
# OpenAI API configuration
os.environ["OPENAI_BASE_URL"] = "https://api.openai-proxy.org/v1"  # 配置API基础URL
os.environ["OPENAI_API_KEY"] = "your-api-key-here"  # 配置你的API密钥
model = "gpt-4o-mini"  # 配置使用的模型
```

## Quick Start

### 运行示例

项目包含一个测试脚本，演示了如何在示例数据集上使用LMTree：

1. 首先确保已正确配置 `LMTree/conf/conf.py` 中的API密钥和URL
2. 直接运行测试脚本：

```bash
python test_method.py
```

该脚本将在 `data/` 目录中的Australian数据集上运行LMTree。你可以通过修改 `test_method.py` 文件开头的 `dataName` 变量来切换到其他数据集（如 "abalone", "boston" 等）。

### 自定义使用

如果要在自己的数据集上使用LMTree，需要准备以下文件：

1. **CSV数据文件**: 包含特征和目标列的数据
2. **JSON配置文件**: 包含数据集元信息，格式如下：

```json
{
    "dataset_name": "your_dataset_name",
    "type": true,  // true表示分类任务，false表示回归任务
    "num_samples": 690,
    "description": "数据集描述信息",
    "attribute_introduction": {
        "feature1": "特征1的描述",
        "feature2": "特征2的描述"
    },
    "target": "target_column_name",
    "is_categorical": [false, true, false]  // 对应每个特征是否为分类特征
}
```

然后使用以下代码：

```python
import json
import pandas as pd
from LMTree.method.LMTree import LMTree

# 加载数据和配置
df = pd.read_csv('your_data.csv')
with open('your_config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)

# 初始化LMTree
lm_tree = LMTree(
    df=df,
    target_column_name=config['target'],
    dataName=config['dataset_name'],
    attribute_introduction=config['attribute_introduction'],
    is_categorical=config['is_categorical'],
    taskType="classification" if config['type'] else "regression",
    content_desc=config['description'],
    max_iterations=50,  # 可调整的参数
    max_depth=3
)

# 运行特征工程
result_data = lm_tree.run()
print("特征工程完成，生成的数据集:", result_data.columns.tolist())
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
