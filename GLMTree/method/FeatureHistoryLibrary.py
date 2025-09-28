import re
from collections import Counter
import random


class FeatureHistoryLibrary:
    def __init__(self):
        self.feature_history = []  # Feature history library
        self.operator_frequency = Counter()  # Operator frequency statistics
        self.operator_examples = {
    # Unary operations
    'np.sqrt': ["np.sqrt(df['col'])", "Square root"],
    'np.abs': ["np.abs(df['col'])", "Absolute value"],
    'np.tanh': ["np.tanh(df['col'])", "Hyperbolic tangent"],
    'np.log': ["np.log1p(np.abs(df['col']))", "Logarithm (handles negative values)"],
    'np.reciprocal': ["np.reciprocal(df['col'])", "Reciprocal"],
    'np.round': ["np.round(df['col'])", "Round to the nearest integer"],
    'np.exp': ["np.exp(df['col'])", "Exponential"],
    'np.square': ["np.square(df['col'])", "Square"],
    'normalize': ["(df['col'] - df['col'].min()) / (df['col'].max() - df['col'].min())", "Normalization"],
    'np.floor': ["np.floor(df['col'])", "Floor to the nearest integer"],
    'np.ceil': ["np.ceil(df['col'])", "Ceil to the nearest integer"],
    'np.cbrt': ["np.cbrt(df['col'])", "Cube root"],
    'relu': ["np.maximum(df['col'], 0)", "ReLU activation function, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'softmax': ["np.exp(df['col']) / np.sum(np.exp(df['col']))", "Softmax, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'sigmoid': ["1 / (1 + np.exp(-df['col']))", "Sigmoid transformation, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'standardize': ["(df['col'] - df['col'].mean()) / df['col'].std()", "Standardization, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'frequency_encoding': ["df['col'].map(df['col'].value_counts(normalize=True))", "Frequency encoding, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'unit_norm': ["df['col'] / np.linalg.norm(df['col'])", "Unit normalization, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],

    # Binary operations
    '+': ["df['col1'] + df['col2']", "Addition"],
    '-': ["df['col1'] - df['col2']", "Subtraction"],
    '*': ["df['col1'] * df['col2']", "Multiplication"],
    '/': ["df['col1'] / df['col2']", "Division"],
    '%': ["df['col1'] % df['col2']", "Modulo"],
    'np.where': ["np.where(df['col1'] > df['col2'], 1, 0)", "Conditional operation"],
    'np.linalg.norm': ["np.linalg.norm(df[\"col1\", \"col2\"].values)", "Euclidean distance"],
    'np.power': ["np.power(df['col1'], df['col2'])", "Power operation"],
    'np.arctan2': ["np.arctan2(df['col1'], df['col2'])", "Arctangent"],
    'log_ratio': ["np.log(df['col1'] / df['col2'])", "Logarithmic ratio, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'max_diff_ratio': ["(df['col1'] - df['col2']) / (df['col1'] + df['col2'])", "Maximum difference ratio, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'weighted_mean': ["(w1 * df['col1'] + w2 * df['col2']) / (w1 + w2)", "Weighted average, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'abs_diff': ["np.abs(df['col1'] - df['col2'])", "Absolute difference, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],

    # Multi-element operations
    'np.max': ["df[['col1', 'col2', 'col3']].max(axis=1)", "Maximum value"],
    'np.min': ["df[['col1', 'col2', 'col3']].min(axis=1)", "Minimum value"],
    'np.mean': ["df[['col1', 'col2', 'col3']].mean(axis=1)", "Mean"],
    'np.std': ["df[['col1', 'col2', 'col3']].std(axis=1)", "Standard deviation"],
    'np.var': ["df[['col1', 'col2', 'col3']].var(axis=1)", "Variance"],
    'quantile_diff': ["df[['col1', 'col2', 'col3']].quantile(0.75, axis=1) - df[['col1', 'col2', 'col3']].quantile(0.25, axis=1)", "Interquartile range, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'median': ["df[['col1', 'col2', 'col3']].median(axis=1)", "Median"],
    'range_diff': ["df[['col1', 'col2', 'col3']].max(axis=1) - df[['col1', 'col2', 'col3']].min(axis=1)", "Range, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'cumulative_mean': ["df[['col1', 'col2', 'col3']].cumsum(axis=1) / (np.arange(len(df[['col1', 'col2', 'col3']].columns)) + 1)", "Cumulative average, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'product': ["df[['col1', 'col2', 'col3']].prod(axis=1)", "Product"],
    'mean_abs_dev': ["(df[['col1', 'col2', 'col3']] - df[['col1', 'col2', 'col3']].mean(axis=1)).abs().mean(axis=1)", "Mean absolute deviation, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"],
    'covariance': ["df[['col1', 'col2', 'col3']].cov().values[0, 1]", "Covariance, this operation cannot be directly provided by Pandas\\Numypy, and needs to be manually combined and calculated"]
}  # Operator examples dictionary

    def parse_expression(self, feature_name, expression, score, column_lists, is_userful=1):
        """
        Parse feature fields and operators.

        :param expression: Input expression
        :param column_lists: Available column name list
        :param is_userful: Whether this feature is valid, 1-valid 0-invalid. Invalid features are not stored in history library
        :return: Feature record
        """
        # Exact match fields
        feature_fields = [col for col in column_lists if col in expression.split()]

        # Remove fields, remaining parts as operators
        operators_list = expression.split()

        operators = [op for op in operators_list if op not in feature_fields]  # Filter out elements not in field list as operators
        operators = [op for op in operators if not (re.match(r'^-?\d+(\.\d+)?$', op))]  # Remove pure numeric elements

        # Update operator frequency
        self.operator_frequency.update(operators)

        # Create feature record
        feature_record = {
            'feature_name': feature_name,
            'full_expression': expression,
            'score': score,
            'feature_fields': feature_fields,
            'operators': operators
        }

        # Add to history library if feature is valid
        if is_userful == 1:
            self.feature_history.append(feature_record)

        return feature_record

    def get_operator_frequency(self):
        """
        Get operator frequency.

        :return: Dictionary of operators and their frequencies
        """
        return dict(self.operator_frequency)

    def extract_feature_scores_dict(self):
        """
        Extract feature names and corresponding scores to construct dictionary.
        :return: Dictionary of feature names and scores
        """
        feature_scores_dict = {
            record['feature_name']: record['score']
            for record in self.feature_history
        }
        return feature_scores_dict

    def calculate_jaccard_similarity(self, set1, set2):
        """Calculate Jaccard similarity coefficient."""
        intersection = len(set(set1) & set(set2))
        union = len(set(set1) | set(set2))
        return intersection / union if union != 0 else 0

    def calculate_lcs_similarity(self, seq1, seq2):
        """Calculate longest common subsequence similarity."""

        def lcs_length(X, Y):
            m, n = len(X), len(Y)
            L = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(m + 1):
                for j in range(n + 1):
                    if i == 0 or j == 0:
                        L[i][j] = 0
                    elif X[i - 1] == Y[j - 1]:
                        L[i][j] = L[i - 1][j - 1] + 1
                    else:
                        L[i][j] = max(L[i - 1][j], L[i][j - 1])
            return L[m][n]

        lcs_len = lcs_length(seq1, seq2)
        return lcs_len / min(len(seq1), len(seq2)) if min(len(seq1), len(seq2)) > 0 else 0

    def calculate_feature_similarity(self, target_feature, historical_feature, w1=0.35, w2=0.35, w3=0.3):
        """
        Calculate overall similarity between new feature and historical feature.

        :param target_feature: Target feature record
        :param historical_feature: Historical feature record
        :param w1: Feature field weight
        :param w2: Operator weight
        :param w3: Expression weight
        :return: Total similarity score
        """
        # Part 1: Feature field similarity calculation
        jaccard_fields = self.calculate_jaccard_similarity(
            target_feature['feature_fields'],
            historical_feature['feature_fields']
        )
        lcs_fields = self.calculate_lcs_similarity(
            target_feature['feature_fields'],
            historical_feature['feature_fields']
        )
        fields_similarity = (jaccard_fields + lcs_fields)/2

        # Part 2: Operator similarity calculation
        jaccard_operators = self.calculate_jaccard_similarity(
            target_feature['operators'],
            historical_feature['operators']
        )
        lcs_operators = self.calculate_lcs_similarity(
            target_feature['operators'],
            historical_feature['operators']
        )
        operators_similarity = (jaccard_operators + lcs_operators)/2

        # Part 3: Overall expression sequence similarity score calculation
        lcs_all = self.calculate_lcs_similarity(
            target_feature['full_expression'],
            historical_feature['full_expression']
        )

        # Total similarity calculation
        total_similarity = w1 * fields_similarity + w2 * operators_similarity + w3 * lcs_all
        return total_similarity

    def find_similar_features(self, target_feature, similarity_threshold=0.8):
        """
        Find similar features in historical feature library.

        :param target_feature: Target feature record
        :param similarity_threshold: Similarity threshold
        :return: List of similar features
        """
        similar_features = []
        for historical_feature in self.feature_history:
            similarity = self.calculate_feature_similarity(
                target_feature,
                historical_feature
            )

            if similarity >= similarity_threshold:
                similar_features.append({
                    'feature': historical_feature,
                    'similarity_score': similarity
                })
        # Sort by similarity in descending order
        similar_features.sort(key=lambda x: x['similarity_score'], reverse=True)

        return similar_features

    def demo_similarity_matching(self, new_feature_expr):
        """
        Demonstrate new feature similarity matching process.

        :param new_feature_expr: New feature expression
        :return: Similar feature matching results
        """
        # Parse new feature using existing method
        new_feature = self.parse_expression(
            "new_feature",
            new_feature_expr,
            "0.5",
            ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14',
             'feature_0', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9'],
            is_userful=0
        )
        # Find similar features
        similar_features = self.find_similar_features(new_feature)

        print("Target feature:", new_feature_expr)
        print(f"Direct result:\n{similar_features}")
        print("Similar feature matching results:")
        for idx, match in enumerate(similar_features, 1):
            print(f"{match['feature']['feature_name']}: {match['feature']['full_expression']}  Similarity: {match['similarity_score']:.2f}")
        return similar_features

    def match_features(self, match_criteria):
        """
        Match feature history based on given conditions.

        :param match_criteria: Match criteria dictionary
        :return: List of matched feature records
        """
        matched_features = []

        for record in self.feature_history:
            # Check if match criteria are satisfied
            field_match = all(
                field in record['feature_fields']
                for field in match_criteria.get('feature_fields', [])
            )
            operator_match = all(
                op in record['operators']
                for op in match_criteria.get('operators', [])
            )

            if field_match and operator_match:
                matched_features.append(record)

        return matched_features

    def match_by_fields(self, fields):
        """
        Match by feature fields.

        :param fields: List of fields to match
        :return: List of matched feature records
        """
        return [
            entry for entry in self.feature_history
            if set(fields).issubset(set(entry['feature_fields']))
        ]

    def match_by_operators(self, operators):
        """
        Match by operators.

        :param operators: List of operators to match
        :return: List of matched feature records
        """
        return [
            entry for entry in self.feature_history
            if set(operators).issubset(set(entry['operators']))
        ]

    def match_by_expression_pattern(self, pattern):
        """
        Match by expression pattern.

        :param pattern: Match pattern
        :return: List of matched feature records
        """
        return [
            entry for entry in self.feature_history
            if re.search(pattern, entry['full_expression'])
        ]

    def display_history(self):
        """Display complete feature history."""
        for idx, record in enumerate(self.feature_history, 1):
            print(f"Record {idx}   Feature name: {record['feature_name']}")
            print(f"        Full expression: {record['full_expression']}")
            print(f"        Score: {record['score']}")
            print(f"        Feature fields: {record['feature_fields']}")
            print(f"        Operators: {record['operators']}")
            print()

        print("Operator frequency statistics:")
        for operator, frequency in self.operator_frequency.items():
            print(f"{operator}: {frequency} times")
        print(f"\n####################################\n")
        print("Frequency statistics results:")
        print(self.get_operator_frequency())
        print(f"Feature history library results")
        print(self.feature_history)

    def find_unknownOperator(self, operator_k=5):
        """
        Return unused operators from existing operators.
        :return: Dictionary of unused operators
        """
        def fuzzy_match(key_operator, operator_examples_lists):
            if key_operator in operator_examples_lists:  # Exact match
                return True

            for existing_op in operator_examples_lists:  # Partial match
                if key_operator in existing_op or existing_op in key_operator:
                    return True
            return False
        
        # Convert to lists
        operator_examples_lists = list(self.operator_examples.keys())
        given_dict_lists = list(self.get_operator_frequency().keys())

        # Find unmatched operations in given_dict
        unmatched_in_given_dict = [
            key_operator for key_operator in given_dict_lists
            if not fuzzy_match(key_operator, operator_examples_lists)
        ]

        # Find unmatched operations in operator_examples
        unmatched_in_operator_examples = [
            key_operator for key_operator in operator_examples_lists
            if not fuzzy_match(key_operator, given_dict_lists)
        ]

        # Reconstruct dict object from self.operator_examples
        operator_newDict = {oper: self.operator_examples[oper] for oper in unmatched_in_operator_examples}
        random_keys = random.sample(list(operator_newDict.keys()), operator_k)
        random_operator_dict = {key: operator_newDict[key] for key in random_keys}

        return random_operator_dict

    def construct_complex_features(self, FeatureNameList_Complex, features):
        def get_complex_explanation(op_name, f_list):
            explain_map = {
                'np.where': lambda f: f"By determining whether {f[0]} is greater than {f[1]}, the feature space can be segmented, which helps capture nonlinear or threshold effects.",
                'np.power': lambda f: f"By raising {f[0]} to the power of {f[1]}, the nonlinear amplification or attenuation relationship between two features is modeled.",
                'np.arctan2': lambda f: f"Use {f[0]} and {f[1]} to calculate the arctangent, which is often used for polar coordinates and directional modeling.",
                'log_ratio': lambda f: f"Calculate the logarithmic ratio of {f[0]} and {f[1]}, highlighting relative changes, which is often used in ratio and information gain scenarios.",
                'max_diff_ratio': lambda f: f"Use ({f[0]}-{f[1]})/({f[0]}+{f[1]}) to measure the ratio of the maximum difference to the total, which is suitable for normalized comparison.",
                'weighted_mean': lambda f: f"Calculate the weighted average of {f[0]} and {f[1]}, integrating multiple sources of information to improve feature stability.",
                'abs_diff': lambda f: f"Calculate the absolute difference between {f[0]} and {f[1]}, reflecting the deviation of both, which is suitable for anomaly detection.",
                'np.max': lambda f: f"Take the maximum value of {', '.join(f)}, capturing extreme performance, which is suitable for risk and bottleneck analysis.",
                'np.mean': lambda f: f"Take the mean of {', '.join(f)}, reflecting the overall level, which is suitable for comprehensive scoring.",
                'quantile_diff': lambda f: f"Calculate the interquartile range of {', '.join(f)}, measuring the dispersion of the distribution, which is suitable for robustness analysis.",
                'range_diff': lambda f: f"Calculate the range of {', '.join(f)}, reflecting the maximum and minimum differences, which is suitable for modeling volatility.",
                'product': lambda f: f"Take the product of {', '.join(f)}, modeling multi-feature interaction, which is suitable for joint effect analysis.",
                'mean_abs_log_ratio': lambda f: f"First take the mean of {f[0]}, {f[1]}, and {f[2]}, then take the logarithmic ratio with {f[3]}, integrating the overall level and relative changes to enhance the distinction.",
                'where_max': lambda f: f"Determine whether {f[0]} is greater than the maximum value of {f[1]} and {f[2]}, implement segmented decision-making, which is suitable for complex rule modeling."
            }
            if op_name in explain_map:
                return explain_map[op_name](f_list)
            else:
                return f"Construct complex features by combining {', '.join(f_list)}, enhancing model expression ability."

        # Advanced operators and their reverse Polish notation expression templates
        advanced_ops = [
            # Binary/conditional/ratio operations
            ('np.where', "{f1} {f2} > 1 0 np.where", "df['{target}'] = np.where(df['{f1}'] > df['{f2}'], 1, 0)",
             "Conditional operation"),
            ('np.power', "{f1} {f2} np.power", "df['{target}'] = np.power(df['{f1}'], df['{f2}'])", "Power operation"),
            ('np.arctan2', "{f1} {f2} np.arctan2", "df['{target}'] = np.arctan2(df['{f1}'], df['{f2}'])", "Arctangent"),
            ('log_ratio', "{f1} {f2} / np.log", "df['{target}'] = np.log(df['{f1}'] / df['{f2}'])", "Logarithmic ratio"),
            ('max_diff_ratio', "{f1} {f2} - {f1} {f2} + /",
             "df['{target}'] = (df['{f1}'] - df['{f2}']) / (df['{f1}'] + df['{f2}'])", "Maximum difference ratio"),
            ('weighted_mean', "{f1} {f2} 0.6 * 0.4 * + 1.0 /",
             "df['{target}'] = (0.6 * df['{f1}'] + 0.4 * df['{f2}']) / 1.0", "Weighted average"),
            ('abs_diff', "{f1} {f2} - np.abs", "df['{target}'] = np.abs(df['{f1}'] - df['{f2}'])", "Absolute difference"),
            # Multi-element statistical operations
            ('np.max', "{f1} {f2} {f3} np.max", "df['{target}'] = df[['{f1}', '{f2}', '{f3}']].max(axis=1)", "Maximum value"),
            ('np.mean', "{f1} {f2} {f3} np.mean", "df['{target}'] = df[['{f1}', '{f2}', '{f3}']].mean(axis=1)", "Mean"),
            ('quantile_diff', "{f1} {f2} {f3} 0.75 0.25 quantile_diff",
             "df['{target}'] = df[['{f1}', '{f2}', '{f3}']].quantile(0.75, axis=1) - df[['{f1}', '{f2}', '{f3}']].quantile(0.25, axis=1)",
             "Interquartile range"),
            ('range_diff', "{f1} {f2} {f3} np.max np.min -",
             "df['{target}'] = df[['{f1}', '{f2}', '{f3}']].max(axis=1) - df[['{f1}', '{f2}', '{f3}']].min(axis=1)",
             "Range"),
            ('product', "{f1} {f2} {f3} product", "df['{target}'] = df[['{f1}', '{f2}', '{f3}']].prod(axis=1)", "Product"),
            # Nested combination examples
            ('mean_abs_log_ratio', "{f1} {f2} {f3} np.mean {f4} / np.log",
             "df['{target}'] = np.log(df[['{f1}', '{f2}', '{f3}']].mean(axis=1) / df['{f4}'])", "Mean and logarithmic ratio nesting"),
            ('where_max', "{f1} {f2} {f3} np.max > 1 0 np.where",
             "df['{target}'] = np.where(df['{f1}'] > df[['{f2}', '{f3}']].max(axis=1), 1, 0)", "Conditional and maximum value nesting"),
        ]
        result = []
        for target_feature in FeatureNameList_Complex:
            while True:
                op = random.choice(advanced_ops)
                # Determine number of features needed for this operator
                if op[0] == 'mean_abs_log_ratio':
                    num_feats = 4
                elif op[0] == 'where_max':
                    num_feats = 3
                else:
                    num_feats = random.choice([2, 3])
                if len(features) >= num_feats:
                    break
            f_list = random.sample(features, num_feats)
            # Fill expression
            feature_expression = op[1].format(
                f1=f_list[0], f2=f_list[1] if num_feats > 1 else f_list[0],
                f3=f_list[2] if num_feats > 2 else f_list[0],
                f4=f_list[3] if num_feats > 3 else f_list[0]
            )
            execute_code = op[2].format(
                target=target_feature,
                f1=f_list[0], f2=f_list[1] if num_feats > 1 else f_list[0],
                f3=f_list[2] if num_feats > 2 else f_list[0],
                f4=f_list[3] if num_feats > 3 else f_list[0]
            )
            explanation_useful = get_complex_explanation(op[0], f_list)
            feature_dict = {
                "feature_expression": feature_expression,
                "explanation_useful": explanation_useful,
                "execute_code": execute_code
            }
            result.append(feature_dict)
        return result

    def get_simple_explanation(self, f1, f2, op, unary_func=None):
        op_map = {
            '+': f"Add {f1} and {f2}, reflecting the total or comprehensive effect of both, often used to measure the sum or resource superposition.",
            '-': f"Subtract {f2} from {f1}, highlighting the difference or change between the two, suitable for measuring net increases or comparative relationships.",
            '*': f"Multiply {f1} and {f2}, reflecting the interaction or amplification effect between features, often used to model physical quantities such as energy and area.",
            '/': f"Divide {f1} by {f2}, obtaining a ratio or normalized feature, which helps eliminate the impact of dimensional units and highlight relative relationships."
        }
        unary_map = {
            'np.abs': "Take the absolute value, eliminate the positive and negative effects, suitable for scenes that focus on the magnitude rather than direction.",
            'np.log': "Take the logarithm, compress extreme values, suitable for processing skewed distributions or data with large magnitude differences.",
            'np.exp': "Take the exponential, amplify differences, suitable for modeling exponential growth or decay.",
            'np.sqrt': "Take the square root, smooth large values, suitable for processing variance and distance features.",
            'np.tanh': "Take the tanh normalization to [-1,1], suppressing the impact of extreme values.",
            'np.reciprocal': "Take the reciprocal, highlighting the importance of small values, suitable for modeling inverse relationships.",
            'np.square': "Square the value, amplify the difference between large numbers, suitable for modeling energy and other quadratic relationships.",
            'np.round': "Round to the nearest integer, suitable for discretizing continuous features.",
            'np.floor': "Floor to the nearest integer, suitable for segment processing.",
            'np.ceil': "Ceil to the nearest integer, suitable for segment processing."
        }
        if unary_func:
            return op_map[op] + unary_map[unary_func]
        else:
            return op_map[op]

    def contruct_outputExamples(self, FeatureNameList_Sample, FeatureNameList_Complex) -> str:
        """Construct output example set and return as string."""

        # Define operators and functions
        simple_operators = ['+', '-', '*', '/']
        features = ['col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8', 'col9', 'col10']

        result = []

        # Simple features: enhance randomness, allow unary operator nesting
        for target_feature in FeatureNameList_Sample:
            f1, f2 = random.sample(features, 2)
            op = random.choice(simple_operators)
            use_unary = random.random() < 0.5  # 50% probability to add unary operator
            if use_unary:
                unary_func = random.choice(
                    ['np.abs', 'np.log', 'np.exp', 'np.sqrt', 'np.tanh', 'np.reciprocal', 'np.square', 'np.round',
                     'np.floor', 'np.ceil'])
                # Reverse Polish notation: f1 f2 op unary_func
                feature_expression = f"{f1} {f2} {op} {unary_func}"
                execute_code = f"df['{target_feature}'] = {unary_func}(df['{f1}'] {op} df['{f2}'])"
                explanation_useful = self.get_simple_explanation(f1, f2, op, unary_func)
            else:
                feature_expression = f"{f1} {f2} {op}"
                execute_code = f"df['{target_feature}'] = df['{f1}'] {op} df['{f2}']"
                explanation_useful = self.get_simple_explanation(f1, f2, op)
            feature_dict = {
                "feature_expression": feature_expression,
                "explanation_useful": explanation_useful,
                "execute_code": execute_code
            }
            result.append(feature_dict)

        # Complex features: directly call construct_complex_features
        result += self.construct_complex_features(FeatureNameList_Complex, features)

        # Format as JSON string
        import json
        return json.dumps(result, ensure_ascii=False, indent=4)
