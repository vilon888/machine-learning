# For File I/O functions.
from tensorflow.python.platform import gfile
import tensorflow as tf
# For Graph Analysis functions.
from collections import defaultdict
# For Graph Search funcitons
from graphsurgeon._utils import _regex_list_contains_string, _generate_iterable_for_search, _clean_input_name

class StaticGraph(object):
    '''
    Acts as a thin wrapper for a read-only TensorFlow GraphDef. Supports indexing based on node name/index as well as iteration over nodes using Python's ``for node in static_graph`` syntax.

    Args:
        graphdef (tensorflow.GraphDef/tensorflow.Graph OR graphsurgeon.StaticGraph/graphsurgeon.DynamicGraph OR str): A TensorFlow GraphDef/Graph or a StaticGraph from which to construct this graph, or a string containing a path to a frozen model.

    Attributes:
        node_outputs (dict(str, list(tensorflow.NodeDef))): A mapping of node names to their respective output nodes.
        node_map (dict(str, tensorflow.NodeDef)): A mapping of node names to their corresponding nodes.
        graph_outputs (list(tensorflow.NodeDef)): A list of likely outputs of the graph.
        graph_inputs (list(tensorflow.NodeDef)): A list of likely inputs of the graph.
    '''
    # Constructor using a TF graphdef
    def __init__(self, graphdef=None):
        self._internal_graphdef = tf.compat.v1.GraphDef()
        self.node_outputs = {}
        self.node_map = {}
        self.graph_outputs = {}
        self.graph_inputs = {}
        if graphdef:
            if isinstance(graphdef, str):
                # Handle pb files
                self.read(graphdef)
            elif isinstance(graphdef, tf.compat.v1.GraphDef):
                # And tf.GraphDefs
                self._internal_graphdef = graphdef
                self._initialize_analysis_data()
            elif isinstance(graphdef, tf.Graph):
                # And tf.Graphs
                self._internal_graphdef = graphdef.as_graph_def()
                self._initialize_analysis_data()
            else:
                # And other graphsurgeon graphs.
                self._internal_graphdef = graphdef._internal_graphdef
                self.node_outputs = graphdef.node_outputs
                self.node_map = graphdef.node_map
                self.graph_outputs = graphdef.graph_outputs

    def __getitem__(self, node_name):
        if isinstance(node_name, str):
            # Support indexing nodes by their names
            return self._node_map[node_name]
        else:
            index = node_name
            # Also support iteration over nodes
            return self._internal_graphdef.node[index]

    def __len__(self):
        return len(self._internal_graphdef.node)

    def as_graph_def(self):
        '''
        Returns this StaticGraph's internal TensorFlow GraphDef.

        Args:
            None

        Returns:
            tensorflow.GraphDef
        '''
        return self._internal_graphdef

    '''Graph Analysis Functions'''
    # This function is responsible for initializing internal data
    # members that keep track of information about the graph.
    def _initialize_analysis_data(self):
        # Given a nodes, returns a dictionary of {node name: list(output node names)}
        def _map_node_outputs():
            node_outputs = defaultdict(list)
            for node in self._internal_graphdef.node:
                # If this node isn't already in the dictionary add it.
                # This way, all nodes show up in the output map even if they have no outputs.
                if node.name not in node_outputs:
                    node_outputs[node.name] = []
                for input_name in node.input:
                    input_name = _clean_input_name(input_name)
                    node_outputs[input_name].append(node)
            return node_outputs

        # A mapping of node names to nodes in the GraphDef
        def _map_nodes():
            return {node.name: node for node in self._internal_graphdef.node}

        # Find the likely outputs of the graph i.e. any nodes which do not have outputs.
        def _infer_graph_outputs(node_outputs):
            graph_outputs = []
            for node in self._internal_graphdef.node:
                # Make sure that we're not using hanging nodes as outputs - must have at least one input.
                if len(node_outputs[node.name]) == 0 and len(node.input) > 0:
                    graph_outputs.append(node)
            return graph_outputs

        # Find the likely inputs of the graph i.e. any placeholder/queue nodes.
        # We cannot just search for nodes without inputs because that would include
        # const nodes, read operations etc.
        def _infer_graph_inputs():
            return self.find_nodes_by_op(["Placeholder", "FIFOQueueV2", "FIFOQueue"])

        self.node_outputs = _map_node_outputs()
        self.node_map = _map_nodes()
        self.graph_outputs = _infer_graph_outputs(self.node_outputs)
        self.graph_inputs = _infer_graph_inputs()

    '''File I/O'''
    # Reads a tensorflow GraphDef from a frozen graph's protobuf.
    def read(self, filename):
        '''
        Reads a frozen protobuf file into this StaticGraph.

        Args:
            filename (str): Name of the protobuf file.

        Returns:
            None
        '''
        with tf.io.gfile.GFile(filename, "rb") as frozen_pb:
            self._internal_graphdef.ParseFromString(frozen_pb.read())
        self._initialize_analysis_data()

    # Write to output protobuf
    def write(self, filename):
        '''
        Writes the StaticGraph's internal TensorFlow GraphDef into a frozen protobuf file.

        Args:
            filename (str): Name of the protobuf file to write.

        Returns:
            None
        '''
        with open(filename, "wb") as ofile:
            ofile.write(self._internal_graphdef.SerializeToString())

    # Write to TensorBoard format.
    def write_tensorboard(self, logdir):
        '''
        Writes the StaticGraph's internal TensorFlow GraphDef into the specified directory, which can then be visualized in TensorBoard.

        Args:
            logdir (str): Name of the directory to write.

        Returns:
            None

        Raises:
            ``Warning: Passing a `GraphDef` to the SummaryWriter is deprecated. Pass a `Graph` object instead, such as `sess.graph`.`` This is a known warning, but currently there is no alternative, since TensorFlow will not be able to convert invalid GraphDefs back to Graphs.
        '''
        try:
            writer = tf.compat.v1.summary.FileWriter(logdir=logdir, graph_def=self._internal_graphdef)
            writer.close()
        except ValueError:
            # TensorFlow will complain about any custom nodes created through this utility.
            # The resulting visualization is still correct, so we suppress the error here.
            pass

    '''Graph Search Functions'''
    # Given a nodes and a set of ops, generates a list of nodes.
    def find_nodes_by_op(self, op):
        '''
        Finds nodes in this graph based on their ops.

        Args:
            op (str OR set(str)): The op to look for. Also accepts iterable containers (preferably hashsets) to search for multiple ops in a single pass of the graph.

        Returns:
            list(tensorflow.NodeDef)
        '''
        ops = _generate_iterable_for_search(op)
        return [node for node in self._internal_graphdef.node if node.op in ops]

    # Given a nodes and a set of names, generates a list of nodes. Names can be RegExs.
    def find_nodes_by_name(self, name):
        '''
        Finds nodes in this graph based on their names.

        Args:
            name (str OR list(str)): The name to look for. Also accepts iterable containers (preferably a list) to search for multiple names in a single pass of the graph. Supports regular expressions.

        Returns:
            list(tensorflow.NodeDef)
        '''
        def has_name(node, names):
            # Strip out path information so we only have the node name.
            node_name = node.name.split('/')[-1]
            return _regex_list_contains_string(names, node_name)

        names = _generate_iterable_for_search(name)
        return [node for node in self._internal_graphdef.node if has_name(node, names)]

    # Matches exact paths. Paths can use RegExs.
    def find_nodes_by_path(self, path):
        '''
        Finds nodes in this graph based on their full paths. This will only match exact paths.

        Args:
            path (str OR list(str)): The path to look for. Also accepts iterable containers (preferably a list) to search for multiple paths in a single pass of the graph. Supports regular expressions.

        Returns:
            list(tensorflow.NodeDef)
        '''
        def has_path(node, paths):
            return _regex_list_contains_string(paths, node.name)

        paths = _generate_iterable_for_search(path)
        return [node for node in self._internal_graphdef.node if has_path(node, paths)]

    # Searches for chains of nodes rather than single nodes, based on their ops.
    def find_node_chains_by_op(self, chain):
        '''
        Finds groups of nodes in this graph that match the specified sequence of ops. Returns a list of matching chains of nodes, with ordering preserved.

        Args:
            chain (list(str)): The sequence of ops to look for. Should be ordered with the input of the chain as the first element, and the output as the last.

        Returns:
            list(list(tensorflow.NodeDef))
        '''
        # TODO: Maybe handle cases with multiple overlapping chains
        # (i.e. two or more chains with a common terminating node) better.
        def find_matching_chain(node, chain):
            if node.op == chain[-1]:
                if len(chain) == 1:
                    # Base case - no inputs to check.
                    return [node]
                # Upon finding a matching node, we need to traverse the chain backwards.
                matching_chain = []
                for input_name in node.input:
                    # Recursively check against all of this node's inputs
                    input_name = _clean_input_name(input_name)
                    input_node = self.node_map[input_name]
                    matching_chain.extend(find_matching_chain(input_node, chain[:-1]))
                if matching_chain:
                    # If matching_chain is not [], then it means at least one of the
                    # inputs matched all the way to the beginning of the chain.
                    matching_chain.append(node)
                    return matching_chain
            # If there's no match, return empty.
            return []

        # Generate a list of lists. Each sub-list is a group of nodes matching the op chain.
        matching_chains = []
        # Save time by first filtering out irrelevant nodes.
        terminating_nodes = self.find_nodes_by_op(chain[-1])
        for node in terminating_nodes:
            matching_chain = find_matching_chain(node, chain)
            if matching_chain:
                # Don't append empty (i.e. match was not found) chains
                matching_chains.append(matching_chain)
        return matching_chains

    def find_node_inputs(self, node):
        '''
        Finds input nodes of a given node.

        Args:
            node (tensorflow.NodeDef): The node in which to perform the search.

        Returns:
            list(tensorflow.NodeDef)
        '''
        return [self.node_map[input_name] for input_name in node.input]

    def find_node_inputs_by_name(self, node, name):
        '''
        Finds input nodes of a given node based on their names.

        Args:
            node (tensorflow.NodeDef): The node in which to perform the search.
            name (str OR list(str)): The name to look for. Also accepts iterable containers (preferably a list) to search for multiple names in a single pass. Supports regular expressions.

        Returns:
            list(tensorflow.NodeDef)
        '''

        def has_name(input_name, names):
            # Strip out path information so we only have the node name.
            input_name = input_name.split('/')[-1]
            return _regex_list_contains_string(names, input_name)

        names = _generate_iterable_for_search(name)
        return [self.node_map[input_name] for input_name in node.input if has_name(input_name, names)]

    def find_node_inputs_by_op(self, node, op):
        '''
        Finds input nodes of a given node based on their ops.

        Args:
            node (tensorflow.NodeDef): The node in which to perform the search.
            op (str OR list(str)): The op to look for. Also accepts iterable containers (preferably a list) to search for multiple op in a single pass.

        Returns:
            list(tensorflow.NodeDef)
        '''
        ops = _generate_iterable_for_search(op)
        return [self.node_map[input_name] for input_name in node.input if self.node_map[input_name].op in ops]
