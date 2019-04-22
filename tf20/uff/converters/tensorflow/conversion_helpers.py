"""
conversion_helpers.py

Main script for doing uff conversions from
different frameworks.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from .converter_functions import *  # noqa
from .converter import TensorFlowToUFFConverter as tf2uff, _debug_print

try:
    from tensorflow.python.platform import gfile
    import tensorflow as tf
    #from tensorflo import GraphDef
    from tensorflow.core.framework.graph_pb2 import GraphDef
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have TensorFlow installed.
For installation instructions, see:
https://www.tensorflow.org/install/""".format(err))

import numpy as np
import uff
import uff.model
import os

def _replace_ext(path, ext):
    return os.path.splitext(path)[0] + ext

def from_tensorflow(graphdef, output_nodes=[], preprocessor=None, **kwargs):
    """
    Converts a TensorFlow GraphDef to a UFF model.

    Args:
        graphdef (tensorflow.GraphDef): The TensorFlow graph to convert.
        output_nodes (list(str)): The names of the outputs of the graph. If not provided, graphsurgeon is used to automatically deduce output nodes.
        output_filename (str): The UFF file to write.
        preprocessor (str): The path to a preprocessing script that will be executed before the converter. This script should define a ``preprocess`` function which accepts a graphsurgeon DynamicGraph and modifies it in place.
        write_preprocessed (bool): If set to True, the converter will write out the preprocessed graph as well as a TensorBoard visualization. Must be used in conjunction with output_filename.
        text (bool): If set to True, the converter will also write out a human readable UFF file. Must be used in conjunction with output_filename.
        quiet (bool): If set to True, suppresses informational messages. Errors may still be printed.
        list_nodes (bool): If set to True, the converter displays a list of all nodes present in the graph.
        debug_mode (bool): If set to True, the converter prints verbose debug messages.
        return_graph_info (bool): If set to True, this function returns the graph input and output nodes in addition to the serialized UFF graph.

    Returns:
        serialized UFF MetaGraph (str)

        OR, if return_graph_info is set to True,

        serialized UFF MetaGraph (str), graph inputs (list(tensorflow.NodeDef)), graph outputs (list(tensorflow.NodeDef))
    """

    quiet = False
    input_node = []
    text = False
    list_nodes = False
    output_filename = None
    write_preprocessed = False
    debug_mode = False
    return_graph_info = False
    for k, v in kwargs.items():
        if k == "quiet":
            quiet = v
        elif k == "input_node":
            input_node = v
        elif k == "text":
            text = v
        elif k == "list_nodes":
            list_nodes = v
        elif k == "output_filename":
            output_filename = v
        elif k == "write_preprocessed":
            write_preprocessed = v
        elif k == "debug_mode":
            debug_mode = v
        elif k == "return_graph_info":
            return_graph_info = v

    tf_supported_ver = "1.12.0"
    if not quiet:
        print("NOTE: UFF has been tested with TensorFlow " + str(tf_supported_ver) + ". Other versions are not guaranteed to work")
    if tf.__version__ != tf_supported_ver:
        print("WARNING: The version of TensorFlow installed on this system is not guaranteed to work with UFF.")

    try:
        import graphsurgeon as gs
    except ImportError as err:
        raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have graphsurgeon installed.
For installation instructions, see:
https://docs.nvidia.com/deeplearning/sdk/tensorrt-api/#python and click on the 'TensoRT Python API' link""".format(err))
    # Create a dynamic graph so we can adjust it as needed.
    dynamic_graph = gs.DynamicGraph(graphdef)
    # Always remove assert ops.
    assert_nodes = dynamic_graph.find_nodes_by_op("Assert")
    dynamic_graph.remove(assert_nodes, remove_exclusive_dependencies=True)
    # Now, run the preprocessor, if provided.
    if preprocessor:
        import importlib, sys
        # Temporarily insert this working dir into the sys.path
        sys.path.insert(0, os.path.dirname(preprocessor))
        # Import and execute!
        pre = importlib.import_module(os.path.splitext(os.path.basename(preprocessor))[0])
        pre.preprocess(dynamic_graph)
        # Now clean up, by removing the directory from the system path.
        del sys.path[0]
    # Run process_dilated_conv() and process_softmax() so the user doesn't have to.
    gs.extras.process_dilated_conv(dynamic_graph)
    gs.extras.process_softmax(dynamic_graph)

    # Get the modified graphdef back.
    graphdef = dynamic_graph.as_graph_def()

    if write_preprocessed and output_filename:
        preprocessed_output_name = os.path.splitext(output_filename)[0] + "_preprocessed"
        dynamic_graph.write(preprocessed_output_name + ".pb")
        dynamic_graph.write_tensorboard(preprocessed_output_name)
        if not quiet:
            print("Preprocessed graph written to " + preprocessed_output_name + ".pb")
            print("TensorBoard visualization written to " + preprocessed_output_name)

    if not quiet:
        print("UFF Version " + uff.__version__)

    if debug_mode:
        _debug_print("Debug Mode is ENABLED")

    if not input_node:
        if not quiet:
            print("=== Automatically deduced input nodes ===")
            print(str(dynamic_graph.graph_inputs))
            print("=========================================\n")
    # Deduce the likely graph outputs if none are provided
    if not output_nodes:
        output_nodes = [node.name for node in dynamic_graph.graph_outputs]
        if not quiet:
            print("=== Automatically deduced output nodes ===")
            print(str(dynamic_graph.graph_outputs))
            print("==========================================\n")

    if list_nodes:
        for i, node in enumerate(graphdef.node):
            print('%i %s: "%s"' % (i + 1, node.op, node.name))
        return

    for i, name in enumerate(output_nodes):
        if debug_mode:
            _debug_print("Enumerating outputs")
        output_nodes[i] = tf2uff.convert_node_name_or_index_to_name(
            name, graphdef.node, debug_mode=debug_mode)
        if not quiet:
            print("Using output node", output_nodes[i])

    input_replacements = {}
    for i, name_data in enumerate(input_node):
        name, new_name, dtype, shape = name_data.split(',', 3)
        name = tf2uff.convert_node_name_or_index_to_name(name, graphdef.node, debug_mode=debug_mode)
        if new_name == '':
            new_name = name
        dtype = np.dtype(dtype)
        shape = [int(x) for x in shape.split(',')]
        input_replacements[name] = (new_name, dtype, shape)
        if not quiet:
            print("Using input node", name)

    if not quiet:
        print("Converting to UFF graph")

    uff_metagraph = uff.model.MetaGraph()
    tf2uff.add_custom_descriptors(uff_metagraph)
    uff_graph = tf2uff.convert_tf2uff_graph(
        graphdef,
        uff_metagraph,
        output_nodes=output_nodes,
        input_replacements=input_replacements,
        name="main",
        debug_mode=debug_mode)

    uff_metagraph_proto = uff_metagraph.to_uff()
    if not quiet:
        print('No. nodes:', len(uff_graph.nodes))

    if output_filename:
        with open(output_filename, 'wb') as f:
            f.write(uff_metagraph_proto.SerializeToString())
        if not quiet:
            print("UFF Output written to", output_filename)
        if text:  # ASK: Would you want to return the prototxt?
            if not output_filename:
                raise ValueError(
                    "Requested prototxt but did not provide file path")
            output_filename_txt = _replace_ext(output_filename, '.pbtxt')
            with open(output_filename_txt, 'w') as f:
                f.write(str(uff_metagraph.to_uff(debug=True)))
            if not quiet:
                print("UFF Text Output written to", output_filename_txt)
    # Always return the UFF graph!
    if return_graph_info:
        return uff_metagraph_proto.SerializeToString(), dynamic_graph.graph_inputs, dynamic_graph.graph_outputs
    else:
        return uff_metagraph_proto.SerializeToString()

def from_tensorflow_frozen_model(frozen_file, output_nodes=[], preprocessor=None, **kwargs):
    """
    Converts a TensorFlow frozen graph to a UFF model.

    Args:
        frozen_file (str): The path to the frozen TensorFlow graph to convert.
        output_nodes (list(str)): The names of the outputs of the graph. If not provided, graphsurgeon is used to automatically deduce output nodes.
        output_filename (str): The UFF file to write.
        preprocessor (str): The path to a preprocessing script that will be executed before the converter. This script should define a ``preprocess`` function which accepts a graphsurgeon DynamicGraph and modifies it in place.
        write_preprocessed (bool): If set to True, the converter will write out the preprocessed graph as well as a TensorBoard visualization. Must be used in conjunction with output_filename.
        text (bool): If set to True, the converter will also write out a human readable UFF file. Must be used in conjunction with output_filename.
        quiet (bool): If set to True, suppresses informational messages. Errors may still be printed.
        list_nodes (bool): If set to True, the converter displays a list of all nodes present in the graph.
        debug_mode (bool): If set to True, the converter prints verbose debug messages.
        return_graph_info (bool): If set to True, this function returns the graph input and output nodes in addition to the serialized UFF graph.

    Returns:
        serialized UFF MetaGraph (str)

        OR, if return_graph_info is set to True,

        serialized UFF MetaGraph (str), graph inputs (list(tensorflow.NodeDef)), graph outputs (list(tensorflow.NodeDef))
    """
    graphdef = GraphDef()
    with tf.io.gfile.GFile(frozen_file, "rb") as frozen_pb:
        graphdef.ParseFromString(frozen_pb.read())
    return from_tensorflow(graphdef, output_nodes, preprocessor, **kwargs)
