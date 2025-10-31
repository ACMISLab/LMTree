import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

class FeatureNode:
    """Feature node for the feature construction graph."""
    def __init__(self, name, expression, depth, explanation_useful, score=0.0, visits_number=0, Q_value=0.0, execute_code=None):
        self.name = name
        self.expression = expression
        self.execute_code = execute_code  # Store the executable Python code
        self.score = score
        self.visits_number = visits_number
        self.Q_value = Q_value  # Q-value for MCTS, different from feature score
        self.depth = depth  # Feature construction depth (original features have depth 0)
        self.explanation_useful = explanation_useful

class FeatureGraph:
    def __init__(self):
        self.graph = nx.DiGraph()  # Directed graph
        self.add_feature('root', 'root', -1, explanation_useful="root_useful", score=-1)

    def add_feature(self, name, expression, depth, explanation_useful, score=0.0, visits_number=1, Q_value=0.0, execute_code=None):
        node = FeatureNode(name, expression, depth, explanation_useful, score, visits_number, Q_value, execute_code)
        self.graph.add_node(name, node=node)
        if depth == 0:  # Original feature node
            self.add_edge('root', name)

    def add_edge(self, from_name, to_name):
        self.graph.add_edge(from_name, to_name)

    def get_node(self, name):
        try:
            return self.graph.nodes[name]['node']
        except:
            return None

    def get_neighbors(self, name):
        return [self.get_node(n) for n in self.graph.neighbors(name)]

    def update_score(self, name, new_score):
        """Update node score."""
        node = self.get_node(name)
        node.score = new_score

    def update_visits_number(self, name):
        """Update node visit count."""
        node = self.get_node(name)
        node.visits_number += 1

    def update_Q_value(self, name, new_Q_value):
        """Update node Q-value."""
        node = self.get_node(name)
        node.Q_value = new_Q_value

    def get_score(self, name):
        return self.get_node(name).score

    def get_visits_number(self, name):
        return self.get_node(name).visits_number

    def get_Q_value(self, name):
        return self.get_node(name).Q_value  

    def get_depth(self, name):
        return self.get_node(name).depth

    def get_explanation_useful(self, name):
        return self.get_node(name).explanation_useful

    def get_max_q_value(self, name):
        """
        Return the maximum Q-value among all direct child nodes of the specified node.
        :param name: Node name
        :return: Maximum Q-value of child nodes, or None if no children exist
        """
        neighbors = self.get_neighbors(name)
        if not neighbors:
            return None
        q_values = [node.Q_value for node in neighbors]
        return max(q_values) if q_values else None

    def getParentsList(self, nodeName):
        """
        Get parent nodes of a node. Only includes direct parents, not all ancestors.
        :param nodeName: Target node name
        :return: List of parent nodes
        """
        NodesList = []
        parents = list(self.graph.predecessors(nodeName))
        for parent in parents:
            NodesList.append(self.get_node(parent))
        return NodesList

    def draw_graph(self, order=None):
        import matplotlib.image as mpimg
        from graphviz import Digraph

        dot = Digraph(
            comment='Process Flowchart',
            format='png',
            graph_attr={
                'ranksep': '1.5',  # Vertical spacing between layers (inches)
                'nodesep': '0.5'   # Horizontal spacing within same layer (inches)
            }
        )

        self.showPngName = "feature_graph"
        if order is not None:
            self.showPngName += str(order)
        
        # Create node labels
        node_labels = {n: (f"{n}\n"
                           f"Expression: {self.get_node(n).expression}\n"
                           f"Score: {self.get_node(n).score}\n"
                           f"Visits: {self.get_node(n).visits_number}\n"
                           f"Q_value: {self.get_node(n).Q_value}\n"
                           f"Depth: {self.get_depth(n)}") for n in self.graph.nodes}

        # Group nodes by depth
        depth_groups = {}
        for n in self.graph.nodes:
            depth = self.get_depth(n)
            if depth not in depth_groups:
                depth_groups[depth] = []
            depth_groups[depth].append(n)

        # Add nodes to graph by depth
        for depth, nodes in depth_groups.items():
            with dot.subgraph() as s:
                s.attr(rank='same')  # Place nodes of same depth on same level
                for featureName in nodes:
                    s.node(featureName, node_labels[featureName], shape='rect')

        # Add edges
        for n in self.graph.nodes:
            neighbors = self.get_neighbors(n)
            for neighborNode in neighbors:
                dot.edge(f'{n}', f'{neighborNode.name}')

        # Render graph as PNG with high DPI
        dot.attr(dpi='600')
        dot.render(f"{self.showPngName}", cleanup=True)

    def getPathNodes(self, nodeName):
        """
        Return all parent and ancestor nodes that lead to nodeName.
        :param nodeName: Target node name
        :return: List containing all parent and ancestor nodes
        """
        path_nodes = []
        visited = set()
        queue = deque()

        # Initialize queue with target node
        queue.append(nodeName)
        visited.add(nodeName)

        while queue:
            current_node = queue.popleft()

            # Get all parent nodes of current node
            predecessors = list(self.graph.predecessors(current_node))

            for parent in predecessors:
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
                    path_nodes.append(parent)
        
        if "root" in path_nodes:
            path_nodes.remove("root")

        return path_nodes
