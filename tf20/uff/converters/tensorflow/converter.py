"""
converter.py

TensorFlow to UFF Converter class. Builds a UFF graph by calling
the convert function and using few helper functions
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ..converter_base import ConverterBase
from .proto import AttrValue
from uff.model.utils import convert_to_str
from uff.model.exceptions import UffException

import numpy as np
try:
    import tensorflow as tf
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have TensorFlow installed.
For installation instructions, see:
https://www.tensorflow.org/install/""".format(err))

import uff.model

import sys
def _debug_print(message):
    print("DEBUG [" + __file__ + ":" + str(sys._getframe(1).f_lineno) + "] " + str(message))

class TensorFlowToUFFConverter(ConverterBase):
    @classmethod
    def convert_layer(cls, op, name, tf_node, inputs, uff_graph, debug_mode=False, **kwargs):
        inputs = [cls.split_node_name_and_output(inp)[0] for inp in inputs]
        if op not in cls.registry_:
            print("Warning: No conversion function registered for layer: %s yet." % op)
            print("Converting " + name + " as custom op: " + str(op))
            # if debug_mode:
            #     _debug_print(tf_node)
            fields = cls.parse_tf_attrs(tf_node.attr)
            uff_graph.custom_node(op, inputs, name, fields if fields else None)
            return inputs
        else:
            if debug_mode:
                _debug_print("For node " + name + " with op " + tf_node.op + ", using conversion function: " + str(cls.registry_[op]))
            return cls.registry_[op](name, tf_node, inputs, uff_graph, **kwargs)

    @classmethod
    def convert_tf2uff_node(cls, name, tf_nodes, uff_graph, input_replacements, debug_mode=False):
        if name in uff_graph.nodes:
            if debug_mode:
                _debug_print(name + " already in UFF graph, skipping.")
            return []
        if name in input_replacements:
            new_name, dtype, shape = input_replacements[name]
            uff_graph.input(shape, dtype, new_name)
            if debug_mode:
                _debug_print("Replacing " + name + " with: " + new_name + " of type " + str(dtype) + " with shape " + str(shape))
            return []
        if name not in tf_nodes:
            raise UffException(str(name) + " was not found in the graph. Please use the -l option to list nodes in the graph.")
        tf_node = tf_nodes[name]
        inputs = list(tf_node.input)
        if debug_mode:
            _debug_print("Converting " + str(tf_node.op) + " node " + str(tf_node.name))
        # Find any identity inputs and don't add them to the UFF graph.
        for i, inp in enumerate(inputs):
            inp_name, num = cls.split_node_name_and_output(inp)
            if debug_mode:
                _debug_print("Found input " + str(inp_name))
            inp_node = tf_nodes[inp_name]
            if inp_node.op == 'Identity':
                if debug_mode:
                    _debug_print("Removing Identity input from graph")
                inputs[i] = inp_node.input[0]
        op = tf_node.op
        uff_node = cls.convert_layer(
            op, name, tf_node, inputs, uff_graph, tf_nodes=tf_nodes, debug_mode=debug_mode)
        return uff_node

    @classmethod
    def convert_tf2uff_graph(cls, tf_graphdef, uff_metagraph, output_nodes,
                             input_replacements=[], name=None, debug_mode=False):
        tf_nodelist = list(tf_graphdef.node)
        tf_nodes = {node.name: node for node in tf_nodelist}
        if debug_mode:
            _debug_print("Creating new UFF metagraph: " + name)
        uff_graph = uff_metagraph.add_graph(name)

        nodes_to_convert = list(output_nodes)
        while len(nodes_to_convert):
            nodes_to_convert += cls.convert_tf2uff_node(nodes_to_convert.pop(), tf_nodes,
                                                        uff_graph, input_replacements, debug_mode=debug_mode)
        if debug_mode:
            _debug_print("Marking {:} as outputs".format(output_nodes))
        for output in output_nodes:
            uff_graph.mark_output(output)
        return uff_graph

    @classmethod
    def convert_tf2numpy_dtype(cls, dtype):
        return tf.as_dtype(dtype).as_numpy_dtype

    @classmethod
    def get_tf_int_list(cls, a):
        return [int(i) for i in a.list.i]

    @classmethod
    def get_tf_shape_as_int_list(cls, a):
        # if __debug__:
        #     print("Found shape, generating int list: " + str([int(dim.size) for dim in a.shape.dim]))
        return [int(dim.size) for dim in a.shape.dim]

    @classmethod
    def convert_tf2uff_data_format(cls, fmt):
        return {'NCHW': 'NC+', 'NHWC': 'N+C'}[fmt]

    @classmethod
    def convert_tf2numpy_const_node(cls, tf_node):
        if tf_node.op != "Const":
            raise UffException("Const node conversion requested, but node is not Const\n" + str(tf_node))
        tensor = tf_node.attr['value'].tensor
        shape_ = tensor.tensor_shape.dim
        shape = [int(d.size) for d in shape_]
        # Special case for non-array constants
        if (shape == [] or shape is None):
            if (tensor.dtype == tf.string):
                return tensor.string_val

        np_dtype = cls.convert_tf2numpy_dtype(tf_node.attr['dtype'].type)

        if len(tensor.float_val) != 0:
            array = np.array(tensor.float_val, dtype=np_dtype)
        elif len(tensor.int_val) != 0:
            array = np.array(tensor.int_val, dtype=np_dtype)
        elif len(tensor.bool_val) != 0:
            array = np.array(tensor.bool_val, dtype=np_dtype)
        else:
            data = tensor.tensor_content
            array = np.frombuffer(data, dtype=np_dtype)
        # if there is only one element, it has to be broadcasted into the shape
        if array.size == 1:
            array = np.tile(array, shape)
        if array.size == 0:
            shape = 0
        return array.reshape(shape)

    @classmethod
    def apply_fused_padding(cls, tf_node, inputs, tf_nodes):
        tf_padding = convert_to_str(tf_node.attr['padding'].s)
        padding = None
        fields = {}
        if tf_padding == 'SAME':
            fields['implicit_padding'] = 'same'
        elif tf_padding == 'VALID':
            fields['implicit_padding'] = None
            tf_lhs_node = tf_nodes[inputs[0]]
            if tf_lhs_node.op == 'Pad':
                tf_padding_node = tf_nodes[tf_lhs_node.input[1]]
                p = cls.convert_tf2numpy_const_node(tf_padding_node)
                before, after = p[:, 0].tolist(), p[:, 1].tolist()
                if before == after:
                    padding = before
                    inputs[0] = tf_lhs_node.input[0]
        else:
            raise ValueError("Padding mode %s not supported" % tf_padding)
        return inputs, padding, fields

    @classmethod
    def split_node_name_and_output(cls, s):
        name_num = s.replace('^', '').rsplit(':', 1)
        if len(name_num) == 1:
            name_num += [0]
        return name_num

    @classmethod
    def convert_tf2uff_field(cls, code, val):
        if isinstance(val, tf.compat.v1.AttrValue):
            val = getattr(val, code)
        if code == 'i':
            return int(val)
        elif code == 'f':
            return float(val)
        elif code == 's':
            return str(val)
        elif code == 'b':
            return bool(val)
        elif code == 'type':
            return TensorFlowToUFFConverter.convert_tf2numpy_dtype(val)
        elif code == 'list':
            fields = val.ListFields()
            if len(fields) == 0:
                return None
            elif len(fields) > 1:
                raise ValueError("Invalid list field")
            else:
                field_desc, field_value = fields[0]
                code = field_desc.name
                uff_code = {'i': 'i', 'f': 'd', 's': 's', 'b': 'b',
                            'type': 'dtype', 'list': 'list', 'shape': 'shape'}[code]
                return uff.model.List(uff_code, [cls.convert_tf2uff_field(code, v) for v in field_value])
        elif code == 'shape':
            shp = val.dim
            if hasattr(shp, "unknown_rank") and shp.unknown_rank:
                raise ValueError(
                    "Unsupported: shape attribute with unknown rank")
            return uff.model.List('i', [dim.size for dim in shp])
        elif code == 'func':
            return dict(val.ListFields())
        else:
            print(val)
            raise TypeError("Unsupported field type:" + code)

    @classmethod
    def parse_tf_attr_value(cls, val):
        code = val.WhichOneof('value')
        return cls.convert_tf2uff_field(code, val)

    @classmethod
    def parse_tf_attrs(cls, attrs):
        return {key: cls.parse_tf_attr_value(val) for key, val in attrs.items() if val is not None and val.WhichOneof('value') is not None}

    @classmethod
    def add_custom_descriptors(cls, uff_metagraph):
        TF_EXTENSION_DESC = uff.model.Descriptor(
            "tensorflow_extension", 0x1, False, {
                "Conv":
                    uff.model.DescriptorOp()
                    .field_enum("implicit_padding", ["same"], optional=True),
                "ConvTranspose":
                    uff.model.DescriptorOp()
                    .field_enum("implicit_padding", ["same"], optional=True),
                "Pool":
                    uff.model.DescriptorOp()
                    .field_enum("implicit_padding", ["same"], optional=True),
            })
        uff_metagraph.extend_descriptor(TF_EXTENSION_DESC)

    @classmethod
    def convertible_to_int(cls, s):
        try:
            int(s)
            return True
        except ValueError:
            return False

    @classmethod
    def convert_node_name_or_index_to_name(cls, name_or_index, nodes, debug_mode=False):
        if debug_mode:
            _debug_print("Extracting name information from " + str(name_or_index))
        if cls.convertible_to_int(name_or_index):
            node_idx = int(name_or_index)
            if node_idx > 0:
                node_idx -= 1
            elif node_idx < 0:
                node_idx += len(nodes)
            else:
                raise ValueError("Invalid node index: %i" % node_idx)
            return nodes[node_idx].name
        else:
            return name_or_index
