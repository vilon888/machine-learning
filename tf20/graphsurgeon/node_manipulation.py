from tensorflow import TensorShape, float32, int32
from tensorflow.core.framework.node_def_pb2 import NodeDef

import numpy as np
# Compatibility with Python 3.X where basestring no longer exists.
try:
    basestring
except NameError:
    basestring = str

""" Node Creation Functions """
# Creates free standing nodes. _do_suffix controls whether UFF parser suffixes are appended to field names. 
def create_node(name, op=None, _do_suffix=False, **kwargs):
    '''
    Creates a free-standing TensorFlow NodeDef with the specified properties.

    Args:
        name (str): The name of the node.
        op (str): The node's operation.

    Keyword Args:
        dtype (tensorflow.DType): TensorFlow dtype.
        shape (tuple(int)): Iterable container (usually a tuple) describing the shape of a tensor.
        inputs (list(tensorflow.NodeDef) or str): Iterable container (usually a tuple) of input nodes or input node names. Supports mixed-type lists.
        **kwargs (AttrName=Value): Any additional fields that should be present in the node. Currently supports int, float, bool, list(int), list(float) and str.

    Returns:
        tensorflow.NodeDef
    '''
    if not _do_suffix:
        print("WARNING: To create TensorRT plugin nodes, please use the `create_plugin_node` function instead.")
    node = NodeDef()
    node.name = name
    node.op = op if op else name
    for key, val in kwargs.items():
        if key == "dtype":
            node.attr["dtype"].type = val.as_datatype_enum
        elif key == "shape":
            for val in val:
                node.attr[key].shape.dim.add(size=val)
        elif key == "inputs":
            # Accept either nodes or strings. This method accepts mixed lists too.
            for input_node in val:
                if isinstance(input_node, NodeDef):
                    node.input.append(input_node.name)
                elif isinstance(input_node, basestring):
                    node.input.append(input_node)
                else:
                    raise TypeError("Input type unrecognized. Must be a tensorflow.NodeDef or a string.")
        elif isinstance(val, int):
            node.attr[(key + "_u_int") if _do_suffix else key].i = val
        elif isinstance(val, float):
            node.attr[(key + "_u_float") if _do_suffix else key].f = val
        elif isinstance(val, bool):
            node.attr[(key + "_u_bool") if _do_suffix else key].b = val
        elif isinstance(val, list):
            if any(isinstance(n, float) for n in val):
                # If any of the values in the list are floats, the whole list gets promoted to floats.
                node.attr[(key + "_u_flist") if _do_suffix else key].list.f.extend(val)
            elif all(isinstance(n, int) for n in val):
                # For int lists, all values have to be ints - no downcasting should happen.
                node.attr[(key + "_u_ilist") if _do_suffix else key].list.i.extend(val)
        elif isinstance(val, basestring):
            # Workaround for unicode strings.
            try:
                node.attr[(key + "_u_str") if _do_suffix else key].s = str.encode(val)
            except TypeError:
                node.attr[(key + "_u_str") if _do_suffix else key].s = bytes(val)
        else:
            raise TypeError("Type: " + str(type(val)) + " unrecognized.")
    # Return a node will all the correct attributes
    return node

def create_plugin_node(name, op=None, **kwargs):
    '''
    Creates a free-standing TensorFlow NodeDef with the specified properties. This is similar to `create_node`,

    Args:
        name (str): The name of the node.
        op (str): The node's operation.
        dtype (tensorflow.DType): TensorFlow dtype.
        shape (tuple(int)): Iterable container (usually a tuple) describing the shape of a tensor.
        inputs (list(tensorflow.NodeDef) or str): Iterable container (usually a tuple) of input nodes or input node names. Supports mixed-type lists.
        **kwargs (AttrName=Value): Any additional fields that should be present in the node. Currently supports int, float, bool, list(int), list(float) and str.

    Returns:
        tensorflow.NodeDef
    '''
    return create_node(name, op, _do_suffix=True, **kwargs)

def create_constant_tensor(name, c, **kwargs):
    '''
    Creates a free-standing TensorFlow NodeDef being a constant operation. This is equivalent
    to tf.constant(c), where c is an array of any dimension. The type of the tensor
    is determined by the type of an array.
    
    Args:
        name (str): The name of the node
        c (np.array!!): An array representing constant values
        
    Returns:
        tensorflow.NodeDef
    '''
    node = create_node(name, op="Const", **kwargs)
    node.attr["value"].tensor.tensor_shape.CopyFrom(TensorShape(c.shape).as_proto())
    node.attr["dtype"].type = 1
    assert (len(c) > 0), "Invalid constant argument"
    if (all(isinstance(x, np.int_) for x in c)):
            node.attr["value"].tensor.int_val.extend(c)
    elif (all(isinstance(x, np.float_) for x in c)):
            node.attr["value"].tensor.float_val.extend(c)
    else:
        assert False, "Constants tensors of other type than int and float \
                       are not yet supported"
    return node
