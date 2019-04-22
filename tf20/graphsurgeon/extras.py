import graphsurgeon as gs

"""
These are more advanced, and more specific functions that don't necessarily fit
in with the more generic search and maniplulation functions. These might include
small functions that compose other graphsurgeon functions for more complex behavior,
or completely free-standing functions that are still typically needed for graph processing.
"""

def process_dilated_conv(dynamic_graph):
    '''
    Replaces **SpaceToBatchND -> Conv2D -> BatchToSpaceND** (this is how TensorFlow represents dilated convolutions internally) with a single node that the UFF converter is able to recognize as a dilated convolution.

    Args:
        dynamic_graph (graphsurgeon.DynamicGraph): DynamicGraph in which to replace dilated convolutions.

    Returns:
        None
    '''
    # Find chains of ops that we recognize as dilated convolutions.
    op_chain = ["SpaceToBatchND", "Conv2D", "BatchToSpaceND"]
    node_chains = dynamic_graph.find_node_chains_by_op(op_chain)
    # Some nodes need to be forwarded, others removed.
    forward_inputs_nodes = []
    remove_nodes = []
    for chain in node_chains:
        # The first node is SpaceToBatchND
        forward_inputs_nodes.append(chain[0])
        # The last node is BatchToSpaceND
        forward_inputs_nodes.append(chain[-1])
        # Reattach the crops node to the Conv instead.
        crops = dynamic_graph.find_node_inputs_by_name(chain[-1], "crops")[0]
        chain[-1].input.remove(crops.name)
        chain[0].input.append(crops.name)
        # Remove all inputs from the BatchToSpaceND except the convolution.
        remove_nodes.extend([dynamic_graph.node_map[inp] for inp in chain[-1].input if dynamic_graph.node_map[inp].op != "Conv2D"])
    # Now remove the const nodes.
    dynamic_graph.remove(remove_nodes, remove_exclusive_dependencies=True)
    # Forward inputs the SpaceToBatchND and BatchToSpaceND nodes.
    dynamic_graph.forward_inputs(forward_inputs_nodes)

def process_softmax(dynamic_graph):
    '''
    Replaces **Sub -> Pack -> Slice -> ConcatV2 -> Reshape -> Softmax -> Reshape** (this is how TensorFlow represents softmax internally) with a single node that the UFF converter is able to recognize as a softmax.

    Args:
        dynamic_graph (graphsurgeon.DynamicGraph): DynamicGraph in which to replace softmax nodes.

    Returns:
        None
    '''
    op_chain = ["Sub", "Pack", "Slice", "ConcatV2", "Reshape", "Softmax", "Reshape"]
    node_chains = dynamic_graph.find_node_chains_by_op(op_chain)
    # Some nodes should be removed, others forwarded.
    forward_nodes = []
    remove_nodes = []
    for chain in node_chains:
        # Sub, Pack and Slice can be removed.
        remove_nodes.extend(chain[0:3])
        # Remove the shape input of the slice node
        remove_nodes.extend(dynamic_graph.find_node_inputs_by_name(chain[2], "Shape.*"))
        # For the concat node, we can remove the values input.
        remove_nodes.extend(dynamic_graph.find_node_inputs_by_name(chain[3], "values.*"))
        # The concat and reshape nodes can be forwarded.
        forward_nodes.extend(chain[3:5])
        # Remove the Shape input of the final reshape node
        remove_nodes.extend(dynamic_graph.find_node_inputs_by_name(chain[6], "Shape.*"))
        # Finally forward the last node.
        forward_nodes.append(chain[6])

    dynamic_graph.remove(remove_nodes)
    dynamic_graph.forward_inputs(forward_nodes)
