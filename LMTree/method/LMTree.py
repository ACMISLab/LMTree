import math
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, r2_score
from sklearn.feature_selection import f_classif, mutual_info_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, XGBRegressor
import json
import re
import openai
from datetime import datetime
import warnings

from ..conf.conf import model, temperature, max_tokens
from ..llm.run_llm_code import run_llm_code
from .FeatureGraph import FeatureGraph
from .FeatureHistoryLibrary import FeatureHistoryLibrary

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


class LLMExecutor:
    def __init__(self, max_history_length=5, max_retries=3):
        """
        Initialize LLM executor.
        
        Args:
            max_history_length (int): Maximum conversation history length
            max_retries (int): Maximum retry attempts
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_history_length = max_history_length
        self.max_retries = max_retries
        self.conversation_history = []

    def _parse_json_response(self, message):
        """Parse JSON response from LLM output."""
        try:
            json_patterns = [
                r'```json\s*([\s\S]*?)```',  # JSON code block
                r'```\s*([\s\S]*?)```',     # Generic code block
                r'\[.*?\]',                 # JSON array
            ]
            for pattern in json_patterns:
                match = re.search(pattern, message, re.DOTALL | re.IGNORECASE)
                if match:
                    json_content = match.group(1)
                    json_obj = json.loads(json_content.strip())
                    # Validate JSON object format
                    if isinstance(json_obj, list) and all(
                            isinstance(item, dict) and
                            all(key in item for key in ['feature_expression', 'explanation_useful', 'execute_code'])
                            for item in json_obj
                    ):
                        return json_obj
            print("Unable to find valid JSON content")
            return None
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return None
        except Exception as e:
            print(f"Unknown error occurred during parsing: {e}")
            return None

    def execute(self, prompt, system_prompt=None):
        """
        Execute LLM request and maintain conversation history.

        Args:
            prompt (str): User input prompt
            system_prompt (str, optional): System prompt

        Returns:
            tuple: Feature expressions, explanations, execution code and token information
        """
        if not system_prompt:
            system_prompt = "You are a senior feature engineering engineer. Based on the dataset information provided by users, you need to construct new features according to user requirements to improve the performance of “predicting target features based on features.”"

        # Build message list including conversation history
        if "llama" in self.model:
            messages = [{"role": "assistant", "content": system_prompt}]
        else:
            messages = [{"role": "system", "content": system_prompt}]

        # Add conversation history (limited length)
        for entry in self.conversation_history[-self.max_history_length:]:
            messages.append({"role": "user", "content": entry['user']})
            messages.append({"role": "assistant", "content": entry['assistant']})

        messages.append({"role": "user", "content": prompt})

        for attempt in range(self.max_retries):
            try:
                response = openai.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )

                message = response.choices[0].message.content
                import re
                match = re.search(r'```(.*?)```', message, re.DOTALL)
                if match:
                    message = f"""```{match.group(1).strip()}```"""
                else:
                    message = "None"
                total_tokens = response.usage.total_tokens
                prompt_tokens = response.usage.prompt_tokens
                completion_tokens = response.usage.completion_tokens
                all_tokens_lists = [total_tokens, prompt_tokens, completion_tokens]

                json_result = self._parse_json_response(message)

                if json_result is None:
                    print(f"The {attempt + 1}th attempt to resolve failed. Continuing to retry...")
                    continue

                # Record conversation history
                conversation_entry = {
                    "timestamp": datetime.now().isoformat(),
                    "user": prompt,
                    "assistant": message
                }
                self.conversation_history.append(conversation_entry)

                return (json_result, all_tokens_lists)

            except openai.OpenAIError as e:
                print(f"API error ({attempt + 1} attempt): {e}")
                if isinstance(e, openai.RateLimitError):
                    import time
                    time.sleep(2 ** attempt)  # Exponential backoff
            except Exception as e:
                print(f"Unknown error ({attempt + 1} attempt): {e}")
        raise RuntimeError("After multiple attempts, the LLM request still cannot be successfully executed.")


class LMTree:
    def __init__(self,
                 df,
                 target_column_name,
                 dataName,
                 attribute_introduction,
                 is_categorical,
                 taskType="classification",
                 max_iterations=50,
                 base_evaluator=None,
                 optimization_metric: str = "auc",
                 random_state=12,
                 splitSize=0.7,
                 content_desc=None,
                 num_expand_features=3,
                 threshold=0.9,
                 exploration_constant=0.1,
                 max_depth=3,
                 ):
        """
        Initialize LMTree model.
        
        Args:
            df: Input dataframe
            target_column_name: Target column name
            dataName: Dataset name
            attribute_introduction: Feature descriptions
            is_categorical: Boolean list indicating categorical features
            taskType: Task type ("classification" or "regression")
            max_iterations: Maximum iteration count for feature construction
            base_evaluator: Base ML model for evaluation
            optimization_metric: Optimization metric (auc/accuracy for classification, rmae/r2 for regression)
            random_state: Random seed
            splitSize: Train-test split ratio
            content_desc: Dataset description
            num_expand_features: Number of features to expand per iteration
            threshold: Selection threshold
            exploration_constant: Exploration constant for UCB
            max_depth: Maximum tree depth
        """
        self.initTime = time.time()
        self.data = df
        self.target_column_name = target_column_name
        self.max_iterations = max_iterations
        self.GraphData = FeatureGraph()
        self.FeatureLibrary = FeatureHistoryLibrary()
        self.base_evaluator = base_evaluator
        if self.base_evaluator is None:
            if taskType == "classification":
                self.base_evaluator = XGBClassifier(random_state=18)
            else:
                self.base_evaluator = XGBRegressor(random_state=18)

        self.num_expand_features = num_expand_features
        self.optimization_metric = optimization_metric
        self.random_state = random_state
        self.splitSize = splitSize
        self.taskType = taskType
        self.is_dynamic_uci_C = True
        self.uci_constant = 1
        self.over_uci_cons = 0.05
        self.uci_realTimeGain = (self.uci_constant - self.over_uci_cons) / (self.max_iterations - 1)
        self.DataFullFeatures = self.data
        self.dataName = dataName
        self.attribute_introduction = attribute_introduction
        self.is_categorical = is_categorical
        self.content_desc = content_desc

        self.max_depth = max_depth
        self.exploration_constant = exploration_constant
        self.threshold = threshold

        self.X = self.data.drop(columns=[self.target_column_name], axis=1)
        self.Y = self.data[self.target_column_name]
        self.Train_X, self.Test_X, self.Train_Y, self.Test_Y = train_test_split(self.X, self.Y,
                                                                                test_size=self.splitSize,
                                                                                random_state=self.random_state)
        self.allTokensLists = [0, 0, 0]  # Store token information: total, input, output

        self.initialize_feature_scores()

    def initialize_feature_scores(self):
        """Initialize feature scores and basic ML scores."""
        self.init_column_feature = {}
        root_node = self.GraphData.get_node("root")
        for columName in self.data.columns.tolist():
            if columName != self.target_column_name:
                score = self.evaluate_feature_set(self.data, columName)
                explain_useful = self.attribute_introduction[columName]
                self.GraphData.add_feature(f'{columName}', f'{columName}', 0, explain_useful, score=score,
                                           Q_value=score)
                self.init_column_feature[columName] = score
        self.ML_InitScore = self.ScoreModel(self.X, self.Y)

    def calculate_initial_score(self, feature):
        """Calculate initial feature score using multiple metrics."""
        def calculate_variance_score(feature):
            return np.var(self.data[feature])

        def calculate_correlation_score(feature):
            return np.abs(self.data[feature].corr(self.target_column_name))

        def calculate_anova_score(feature):
            return f_classif(self.data[[feature]], self.target_column_name)[0][0]

        def calculate_mutual_info_score(feature):
            return mutual_info_classif(self.data[[feature]], self.target_column_name)[0]

        def calculate_chi2_score(feature):
            return chi2(self.data[[feature]], self.target_column_name)[0][0]

        var_score = calculate_variance_score(feature)
        corr_score = calculate_correlation_score(feature)
        anova_score = calculate_anova_score(feature)
        mi_score = calculate_mutual_info_score(feature)
        chi2_score = calculate_chi2_score(feature)
        return np.mean([var_score, corr_score, anova_score, mi_score, chi2_score])

    def Data_sampling(self, X_train, y_train, X_test, y_test, Random_status=42):
        """Data sampling to prevent excessive computation time."""
        X_train_num_samples = len(X_train)
        X_train_sample = X_train.sample(n=min(5000, X_train_num_samples), random_state=Random_status)
        y_train_sample = y_train.loc[X_train_sample.index]
        return X_train_sample, y_train_sample, X_test, y_test

    def ScoreModel(self, xData, yData):
        """Evaluate model performance using specified metrics."""
        X_train_temp, X_test_temp, y_train_temp, y_test_temp = train_test_split(xData, yData, test_size=self.splitSize,
                                                                                random_state=42)
        X_train, y_train, X_test, y_test = self.Data_sampling(X_train_temp, y_train_temp, X_test_temp, y_test_temp)

        model = self.base_evaluator
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        if self.taskType == "classification":
            acc = accuracy_score(y_test, pred)
            pred_proba = model.predict_proba(X_test)
            unique_classes = np.unique(yData)
            if len(unique_classes) > 2:
                auc = roc_auc_score(y_test, pred_proba, multi_class="ovo")
            else:
                auc = roc_auc_score(y_test, pred_proba[:, 1])
            if self.optimization_metric == "auc":
                return auc
            elif self.optimization_metric == "accuracy":
                return acc
            else:
                return auc
        else:
            r2 = r2_score(y_test, pred)
            y_mean = np.mean(y_test)
            mae = np.sum(np.abs(y_test - pred))
            mae_baseline = np.sum(np.abs(y_test - y_mean))
            relative_mae = 1 - (mae / mae_baseline)
            if self.optimization_metric == "r2":
                return r2
            elif self.optimization_metric == "rmae":
                return relative_mae
            else:
                return relative_mae

    def evaluate_feature_set(self, data, pending_feature):
        # 评估特征集
        yData = data[self.target_column_name]  #目标列
        xFullData = data.drop(columns=[self.target_column_name])
        xLackData = xFullData.drop(columns=[pending_feature])

        ScoreFull = self.ScoreModel(xFullData, yData)
        ScoreLack = self.ScoreModel(xLackData, yData)
        return ScoreFull - ScoreLack

    def uci_formula(self, node, start_node=None):  ###################
        # 计算UCI值
        # 在计算start_node时，可能确认无法找到start_node 的父节点。确实无法找到时，start_node=none
        if node.depth == -1:  #若是根节点
            uci = node.score
        else:
            ##总访问次数计算方式一：获取node所有对应父节点，统计各父节点总访问次数
            # 获取指向节点的所有父节点，不包括所有祖先节点
            ParentNodesList = self.GraphData.getParentsList(node.name)
            N_c_total = 0
            for parentName in ParentNodesList:
                N_c_total += parentName.visits_number

            # ##总访问次数计算方式二：仅获取从最初节点指向node路径上的父节点，即传入参数start_node
            # # 获取指向节点的所有父节点，不包括所有祖先节点
            # N_c_total = start_node.visits_number

            N_j = node.visits_number  #当前节点的访问次数
            ln_N_c = math.log(N_c_total)  # 计算 \ln N_C
            fraction = (ln_N_c + 1) / N_j  # 计算 \frac{ \ln N_C}{N_j}
            exploration_term = math.sqrt(fraction)  # 计算平方根
            uci = node.Q_value + self.uci_constant * exploration_term  #最终UCI计算值 Q+UCI_constant*sqrt(ln(N_c)/N_j)

        return uci


    def select_features(self, StartNodeName):
        # 获取当前节点
        start_node = self.GraphData.get_node(StartNodeName)
        child_neighbors = self.GraphData.get_neighbors(start_node.name)  # 获取所有孩子节点

        # 如果没有孩子节点，直接返回当前节点（端点节点）
        if not child_neighbors:
            return start_node

        # 初始化最优节点和最大 UCI 值
        best_featureNode = None
        best_uci = -np.inf

        # 遍历所有孩子节点，计算 UCI 值并找到最优的孩子节点
        for neighbor in child_neighbors:
            current_uci = self.uci_formula(neighbor, start_node)  # 计算当前孩子的 UCI 值
            if current_uci > best_uci:  # 更新最优节点和最大 UCI 值
                best_uci = current_uci
                best_featureNode = neighbor
        # 如果找到的最佳孩子节点的 UCI 值大于当前节点的 UCI 值，则继续向下选择
        if best_uci > self.uci_formula(start_node):  # 比较最佳子节点和当前节点
            return self.select_features(best_featureNode.name)  # 递归选择下一个节点
        else:# 否则，返回当前节点
            return start_node  # 返回当前节点

    def promptBuilde(self, node, FeatureNameList):
        def contruct_dataBaseInfo():
            baseInof = ("III. Dataset Information\n    The {dataName} dataset (target variable {targetName}) is a {tasktype} task.")
            tasktype = "regression"
            if self.taskType == "classification":
                yNum = len(self.Y.unique())
                if yNum == 2:
                    tasktype = "binary classification"
                elif yNum == 3:
                    tasktype = "three-class classification"
                elif yNum == 4:
                    tasktype = "four-class classification"
                elif yNum == 5:
                    tasktype = "five-class classification"
                elif yNum == 6:
                    tasktype = "six-class classification"
                elif yNum == 7:
                    tasktype = "seven-class classification"
                elif yNum == 8:
                    tasktype = "eight-class classification"
                elif yNum == 9:
                    tasktype = "nine-class classification"
                elif yNum == 10:
                    tasktype = "ten-class classification"
                else:
                    tasktype = "classification"

            CategoricalNum = self.is_categorical[:-1].count(True)  #离散属性个数
            ContinuousNu = self.is_categorical[:-1].count(False)  #连续属性个数
            if ContinuousNu != 0 and CategoricalNum != 0:
                con1 = f"The dataset has {CategoricalNum} discrete features and {ContinuousNu} continuous features."
            elif CategoricalNum != 0:
                con1 = f"All features in the dataset are discrete, with {CategoricalNum} features."
            else:
                con1 = f"All features in the dataset are continuous, with {ContinuousNu} features."
            baseInof += con1

            baseInof = baseInof.format(dataName=self.dataName, targetName=self.target_column_name, tasktype=tasktype)
            if self.content_desc is not None:
                baseInof += self.content_desc

            return baseInof

        def contruct_originalFearturesDesc():
            con = ""
            for order, attr in enumerate(list(self.attribute_introduction.keys())):
                if self.is_categorical[order]:
                    attrType = "discrete type"
                    sampleValue = self.data[attr].unique().tolist()
                else:
                    attrType = "continuous type"
                    sampleValue = self.data[attr].sample(n=10, replace=True).tolist()

                attr_str = f"{attr}|{attrType}|{self.attribute_introduction[attr]}|{sampleValue}"
                con += attr_str
                con += "\n"
            return con

        def contruct_GoodFeatures():
            FResScore = self.FeatureLibrary.extract_feature_scores_dict()  #历史特征得分字典
            sorted_result_score = dict(sorted(FResScore.items(), key=lambda item: item[1], reverse=True))
            sorted_featureList = list(sorted_result_score.keys())# 提取排序后的所有key值

            con = ""
            for featureName in sorted_featureList:
                NodeObject = self.GraphData.get_node(featureName)

                con += f"{NodeObject.name}:{NodeObject.expression},score:{NodeObject.score:.5f}\n"  #只需要特征示例和构建表达式

            return con

        DatasetBasicInfo = contruct_dataBaseInfo()
        OriginalFeatureDesc = contruct_originalFearturesDesc()
        Example_Constructed_Features = contruct_GoodFeatures()

        part1 = """Objective: Based on the following requirements, dataset information, and DataFrame object df, construct new features {BuildFeatureName}. The new features must be constructed by combining '{selectedFeature}' and expressed in reverse Polish notation (postfix expression) to improve the prediction performance of '{target_name}'. Requirements: When constructing {FeatureNameList_Sample}, the expressions must be simple, involving only 1–3 unary or binary operators, and should prioritize the use of original dataset features; when constructing {FeatureNameList_Complex}, complex expressions involving multi-operator expressions must be designed.
    
I.Requirements for feature construction
    1. New features must be constructed by combining features selected from the “original feature column” and “feature history library,” and the '{selectedFeature}' feature must be included. Additionally, it is strictly prohibited to fabricate or use other features.
    2. Operator usage: {operator_frequency}
    3. Avoid constructing feature expressions that are similar to those in the “Feature History Library,” especially when selecting operators and features.
    4. Prioritize features that are related to `{BuildFeatureName}`, such as domain-specific calculation formulas (e.g., [power × runtime] in energy consumption metrics).
    5. Operator selection references: Includes but is not limited to the following operators.
        Unary: normalization, taking the reciprocal, frequency of certain types of discrete features, rounding features, absolute value, logarithm of absolute value, square root of absolute value, exponential, Sigmoid transformation, standardization, squaring. For example, np.tanh(df[‘col’])
        Binary: logarithmic ratio, conditional operation, addition, subtraction, multiplication, division, column modulo, Euclidean distance, etc. e.g., np.where(df[‘col1’] > df[‘col2’], 1, 0)
        Multivariate: maximum value of a column, minimum value of a column, mean of a column, standard deviation, quantile difference, variance, etc. e.g., df[[‘col1’, ‘col2’, ‘col3’]].mean(axis=1)

II.Example Output
    Please refer to the output format below and strictly follow the format to ensure that it can be parsed by Python's `json.loads` function. Do not output any other content. execute_code is the execution code related to feature_expression, and explanation_useful is a description of the usefulness of the new feature.
    ```json
    {outputExamples}
    ```
    
"""
        part3 = """

    The following is information related to the “original feature columns” of the dataset itself, and the last {target_name} column is the target feature column.
Column Name|Feature Type|Feature Description|Sample Value
{OriginalFeatureDesc}

IV. Feature History Library
In addition to the original feature attributes of the dataset, the following are newly constructed features. These features have already been constructed and can be used directly as dataset attribute columns.
{Example_Constructed_Features}

"""
        TempPormpt = part1 + DatasetBasicInfo + part3

        #分别构造两个方向的特征list
        FeatureNameList_Sample = FeatureNameList[0:len(FeatureNameList) // 2]  #简单特征list
        FeatureNameList_Complex = FeatureNameList[len(FeatureNameList) // 2:]  #复杂特征list
        FeatureListStr_SampleStr = ""  #简单特征list 对应字符串 如："feature_0, feature_1"
        FeatureListStr_ComplexStr = ""  #复杂特征list 对应字符串 如："feature_2, feature_3"
        FeatureListStr_SampleStr += ", ".join(FeatureNameList_Sample)  # 如："feature_0, feature_1"
        FeatureListStr_ComplexStr += ", ".join(FeatureNameList_Complex)  # 如："feature_2, feature_3"
        FeatureListStr = ""  #需要构建的特征 List 总的特征list
        FeatureListStr += ", ".join(FeatureNameList)  # 如："feature_0, feature_1, feature_2"
        outputExamples = self.FeatureLibrary.contruct_outputExamples(FeatureNameList_Sample, FeatureNameList_Complex)

        TempPormpt = TempPormpt.format(target_name=self.target_column_name, BuildFeatureName=FeatureListStr,
                                       selectedFeature=node.name, OriginalFeatureDesc=OriginalFeatureDesc,
                                       Example_Constructed_Features=Example_Constructed_Features,
                                       operator_frequency=self.FeatureLibrary.get_operator_frequency(),
                                       outputExamples=outputExamples, FeatureNameList_Sample=FeatureListStr_SampleStr,
                                       FeatureNameList_Complex=FeatureListStr_ComplexStr)
        return TempPormpt

    def getFromPointNodes(self, featureExpression):
        """Extract node names from feature expression."""
        NodesList = []
        for columName in self.data.columns.tolist():
            if columName in featureExpression:
                node = self.GraphData.get_node(columName)
            else:
                node = None
            if node is not None:
                NodesList.append(node.name)

        return NodesList

    def getGraphMaxDepth(self, featureExpression):
        """Get maximum depth from all nodes in feature expression."""
        maxDepth = 0
        NodesList = self.getFromPointNodes(featureExpression)
        for nodeName in NodesList:
            node = self.GraphData.get_node(nodeName)
            if maxDepth < node.depth:
                maxDepth = node.depth
        return maxDepth

    def expand_features(self, node, OrderFeOrigin):
        def detect_and_log_transform(df, column, threshold=1e10, verbose=True):
            """
            Detect and handle extreme values and infinite values in columns.

            Args:
                df: DataFrame
                column: Column name to check
                threshold: Threshold for "too large" values, default 10^10
                verbose: Whether to print detailed logs

            Returns:
                Processed DataFrame
            """
            df = df.copy()

            if column not in df.columns:
                if verbose: 
                    print(f"Warning: Column {column} does not exist.")
                return df

            try:
                df[column] = pd.to_numeric(df[column], errors='coerce')
            except Exception as e:
                if verbose: 
                    print(f"Error converting column {column} to numeric type: {e}")
                return df

            df[column].replace([np.inf, -np.inf], np.nan, inplace=True)
            df[column].fillna(df[column].median(), inplace=True)

            max_value = df[column].max()
            min_value = df[column].min()

            if max_value > threshold or abs(min_value) > threshold:
                if verbose: 
                    print(f"Detected that column {column} contains excessively large numbers; logarithmic transformation applied.")

                positive_mask = df[column] > 0
                df.loc[positive_mask, column] = np.log1p(df.loc[positive_mask, column])

                negative_mask = df[column] < 0
                df.loc[negative_mask, column] = -np.log1p(abs(df.loc[negative_mask, column]))

                if verbose: 
                    print(f"Column {column} logarithmic transformation completed")
            else:
                if verbose: 
                    print(f"The values in column {column} are within the normal range and do not require any action.")
            return df

        max_iter_k = 3

        beginOrder = OrderFeOrigin * self.num_expand_features * 2
        overOrder = (OrderFeOrigin + 1) * self.num_expand_features * 2
        FeatureNameList = []
        for i in range(beginOrder, overOrder):
            FeatureNameList.append(f"feature_{i}")
            self.allColumnsLists.append(f"feature_{i}")

        # Use LLM to expand features
        prompt = self.promptBuilde(node, FeatureNameList)
        llm_executor = LLMExecutor()
        tokens_list_temp = [0, 0, 0]
        isValidationEnabled = True
        
        for _ in range(max_iter_k):
            try:
                ResultList, all_tokens_lists = llm_executor.execute(prompt)

                # System validation module
                is_has_selected_feature = False
                num_status = 0
                all_find_similar_features = []

                for order, result in enumerate(ResultList):
                    feature_expression = result['feature_expression']
                    execute_code = result['execute_code']

                    if node.name in execute_code:
                        is_has_selected_feature = True

                    featureName = FeatureNameList[order]

                    ParseFeature = self.FeatureLibrary.parse_expression(featureName, feature_expression, 0,
                                                                        self.allColumnsLists, is_userful=0)
                    similar_features = self.FeatureLibrary.find_similar_features(ParseFeature)
                    ResultList[order]["is_useful"] = 1
                    if similar_features and isValidationEnabled:
                        all_find_similar_features.extend(similar_features)
                        ResultList[order]["is_useful"] = 0
                        num_status += 1

                if is_has_selected_feature == False:
                    FeatureListStr = ", ".join(FeatureNameList)
                    prompt = f"""I noticed that the 'execute_code' feature in the previously constructed feature {FeatureListStr} was not synthesized based on the feature '{node.name}'. This is contrary to my requirements, so please reconstruct these features based on '{node.name}'."""
                    print(f"The synthetic features do not include the selected features, so the features need to be regenerated. Try {_} times.")

                    if _ != max_iter_k - 2:
                        tokens_list_temp[0] += all_tokens_lists[0]
                        tokens_list_temp[1] += all_tokens_lists[1]
                        tokens_list_temp[2] += all_tokens_lists[2]
                        continue
                    else:
                        tokens_list_temp[0] += all_tokens_lists[0]
                        tokens_list_temp[1] += all_tokens_lists[1]
                        tokens_list_temp[2] += all_tokens_lists[2]

                elif num_status == len(FeatureNameList):
                    FeatureListStr = ", ".join(FeatureNameList)
                    con2 = f"\n\nI need you to reconstruct the {FeatureListStr} feature. The feature you constructed previously is similar to the feature below in terms of feature fields, operators used, overall structure, etc. Now, you need to strictly distinguish it from the feature below and carefully reconstruct the new feature {FeatureListStr}.\n"
                    for similar_feature in all_find_similar_features:
                        con2 += f"{similar_feature['feature']['feature_name']}: {similar_feature['feature']['full_expression']}\n"
                    NewOperatorDict = self.FeatureLibrary.find_unknownOperator(operator_k=10)
                    con2 += f"\n\nAdditionally, when reconstructing {FeatureListStr}, you should carefully consider using the following operators to reconstruct the new feature {FeatureListStr}. Please note: some operators do not have ready-made Numpy Pandas methods and require you to construct expressions yourself. For example, np.frequency_encoding and np.mean_abs_dev cannot be used directly.\n"
                    for Key, Value in NewOperatorDict.items():
                        con2 += f"{Key}: {Value[0]} # {Value[1]}\n"

                    prompt = con2
                    print(f"qgz_test_content")
                    print(f"prompt:{prompt}")
                    print(f"Similar characteristics, need to be re-expanded, try {_} times")

                    if _ != max_iter_k - 1:
                        tokens_list_temp[0] += all_tokens_lists[0]
                        tokens_list_temp[1] += all_tokens_lists[1]
                        tokens_list_temp[2] += all_tokens_lists[2]
                        continue
                    else:
                        tokens_list_temp[0] += all_tokens_lists[0]
                        tokens_list_temp[1] += all_tokens_lists[1]
                        tokens_list_temp[2] += all_tokens_lists[2]
                else:
                    tokens_list_temp[0] += all_tokens_lists[0]
                    tokens_list_temp[1] += all_tokens_lists[1]
                    tokens_list_temp[2] += all_tokens_lists[2]

                CopyData = self.data.copy(deep=True)
                UseFeatureNodeList = []
                e_info = ""
                
                for order, result in enumerate(ResultList):
                    if result["is_useful"]:
                        feature_expression = result['feature_expression']
                        explanation_useful = result['explanation_useful']
                        execute_code = result['execute_code']
                        featureName = FeatureNameList[order]
                        print(f" *New Feature{order}: {execute_code}")

                        try:
                            self.DataFullFeatures = run_llm_code(execute_code, self.DataFullFeatures)
                            self.DataFullFeatures = detect_and_log_transform(self.DataFullFeatures, featureName,
                                                                             verbose=False)

                        except Exception as e:
                            ResultList[order]["is_useful"] = 0
                            num_status += 1
                            e_info += f"execute_code:{execute_code}\nThe error message is:{e}. Checking for the presence of a feature can only be done by selecting from the already features."
                            print(f"    Failed, skip current feature, error message:{e}")
                            continue

                        CopyData[featureName] = self.DataFullFeatures[featureName]
                        # 更新图结构中的扩展节点。添加节点、边、以及设定参数
                        maxDepth = self.getGraphMaxDepth(feature_expression)
                        fromNodes = self.getFromPointNodes(feature_expression)
                        self.GraphData.add_feature(featureName, feature_expression, maxDepth + 1, explanation_useful)
                        for fromNode in fromNodes:
                            self.GraphData.add_edge(fromNode, featureName)
                        # 需要在扩展节点中 修改UCI值
                        FeatureNode = self.GraphData.get_node(featureName)
                        UseFeatureNodeList.append(FeatureNode)

                if num_status == len(FeatureNameList):  # 表示全部校验失败，需要重新构造
                    prompt = f"""{e_info}\n Reconstruct new features according to the above requirements, and strictly prohibit fictitious features."""
                    continue

                # 统计好总的tokensList对象
                self.allTokensLists[0] += tokens_list_temp[0]
                self.allTokensLists[1] += tokens_list_temp[1]
                self.allTokensLists[2] += tokens_list_temp[2]

                return CopyData, UseFeatureNodeList

            except Exception as e:
                prompt = f"""You can only select existing features to combine and construct; fictitious features are strictly prohibited! Feature expansion failed, error message:{e}"""
                print(f"Feature expansion failed, attempted {_} times, error message:{e}")
        
        print(f"Feature Start-{OrderFeOrigin} Extension failed completely")
        return None, None

    def generate_new_features(self, node):
        """Generate new features."""
        return []

    def simulate_and_evaluate(self, data, UseFeatureNodeList):
        """Simulate evaluation and calculate feature scores."""
        feature_score_list = []
        for Order, FeatureNode in enumerate(UseFeatureNodeList):
            TestData = self.data.copy(deep=True)
            TestData[FeatureNode.name] = data[FeatureNode.name]
            score = self.evaluate_feature_set(TestData, FeatureNode.name)
            feature_record = self.FeatureLibrary.parse_expression(FeatureNode.name, FeatureNode.expression, score,
                                                 self.allColumnsLists)
            Num_operators = len(feature_record['operators'])
            Num_features = len(feature_record['feature_fields'])
            L_a = Num_operators * 0.0000125 + Num_features * 0.0000125  # Complexity penalty
            init_Q_value = score - L_a  # Initialize Q value

            self.GraphData.update_score(FeatureNode.name, score)
            self.GraphData.update_Q_value(FeatureNode.name, init_Q_value)
            feature_score_list.append(score)
        print(f"Feature score gain:{feature_score_list}\n")

    def backpropagate(self, UseFeatureNodeList):
        """Backpropagate to update visit counts and Q values."""
        for Order, FeatureNode in enumerate(UseFeatureNodeList):
            PathNodes = self.GraphData.getPathNodes(FeatureNode.name)
            for nodeName in PathNodes:
                self.GraphData.update_visits_number(nodeName)
                childMaxQ_value = self.GraphData.get_max_q_value(nodeName)
                if childMaxQ_value is None:
                    print(f"Error: The current node {nodeName} has no child nodes.")
                    continue
                node = self.GraphData.get_node(nodeName)
                current_Q_value = (node.Q_value + childMaxQ_value) * 0.5
                self.GraphData.update_Q_value(nodeName, current_Q_value)

    def BestFeatureCombination(self):
        """Select the best feature combination through feature selection."""
        finallFeatureDict = self.init_column_feature | self.FeatureLibrary.extract_feature_scores_dict()
        sorted_result_score = dict(sorted(finallFeatureDict.items(), key=lambda item: item[1], reverse=True))
        featureList = list(sorted_result_score.keys())

        # Method 2: Select based on original columns
        XData = self.X.copy(deep=True)
        firstFeatureName = XData.columns.tolist()

        for feature in firstFeatureName:
            if feature in featureList:
                featureList.remove(feature)

        # Evaluate each feature to select feature combination
        for index, featureName in enumerate(featureList[1:]):
            XData[featureName] = self.DataFullFeatures[featureName]

            score = self.ScoreModel(XData, self.Y)
            if score >= self.best_score:
                self.best_score = score
                self.best_featureNameList = XData.columns.tolist()
                self.FeatureData = XData
            else:
                del XData[featureName]

            if index // 5 == 0:
                self.histroy_evaluate_result.append(self.best_score)

        # Compare with original column scores and take the better result
        if self.ML_InitScore >= self.best_score:
            print(f"Instead of the original column score, replace it with the original column {self.ML_InitScore}>{self.best_score}")
            self.best_score = self.ML_InitScore
            self.best_featureNameList = self.X.columns.tolist()
            self.FeatureData = self.X

        self.histroy_evaluate_result.append(self.best_score)

    def run(self):
        """Main execution method for iterative feature construction."""
        self.temp_time = 1
        self.histroy_evaluate_result = []
        self.best_score = 0
        self.best_featureNameList = []
        self.FeatureData = self.X

        self.allColumnsLists = self.data.columns.tolist().copy()
        for OrderFeOrigin in range(self.max_iterations):
            # Execute four framework steps: selection, expansion, simulation, backpropagation
            selected_featureNode = self.select_features(StartNodeName="root")
            print(f"##### The {OrderFeOrigin+1}|{self.max_iterations}th selected base feature: {selected_featureNode.name} #####")
            CopyData, UseFeatureNodeList = self.expand_features(selected_featureNode, OrderFeOrigin)
            if UseFeatureNodeList:
                self.simulate_and_evaluate(CopyData, UseFeatureNodeList)
                self.backpropagate(UseFeatureNodeList)
            if self.is_dynamic_uci_C:
                self.uci_constant -= self.uci_realTimeGain

        self.BestFeatureCombination()
        self.FeatureData[self.target_column_name] = self.Y
        end_time = time.time()
        self.elapsed_time = end_time - self.initTime

        print(f"Total time spent:{self.elapsed_time:.2f}seconds; Total consumption tokens：{self.allTokensLists[0]}")
        print(f"Optimal combination of features:{self.best_featureNameList},Best score：{self.best_score}")
        print(f"Feature column:{self.FeatureData.columns.tolist()}")

        for fe in self.best_featureNameList:
            if "feature" in fe:
                print(f"    {fe:<10}:{self.GraphData.get_node(fe).expression}")

        return self.FeatureData
