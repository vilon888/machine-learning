import numpy as np
from collections import defaultdict, OrderedDict

from . import uff_pb2 as uff_pb
from .data import FieldType, create_data
from .exceptions import UffException
from .node import Node
from .utils import extend_with_original_traceback, int_types


def _create_fields(default_fields, fields=None):
    default_fields.update(fields if fields else {})
    return default_fields


class Graph(object):

    def __init__(self, meta_graph, name):
        self.name = name
        self.meta_graph = meta_graph
        # here to preserve the orders of node for the pbtxt to be more readable
        self.nodes = OrderedDict()
        self.op_counts = defaultdict(int)

    def to_uff(self, debug=False):
        graph = uff_pb.Graph(id=self.name, nodes=self._check_graph_and_get_nodes())
        if debug:
            graph = uff_pb.Graph(id=self.name,
                                 nodes=[node.to_uff(debug) for node in self.nodes.values()])
        return graph

    def _check_and_get_node(self, node):
        node = node.to_uff()
        for i in node.inputs:
            if i not in self.nodes:
                raise UffException("In node %s, %s input doesn't exist" % (node, i))
        self.meta_graph.descriptor.check_node(node, self.meta_graph.referenced_data)
        return node

    def _check_graph_and_get_nodes(self):
        nodes = []
        for node in self.nodes.values():
            try:
                nodes.append(self._check_and_get_node(node))
            except Exception as e:
                raise extend_with_original_traceback(e, node._trace)

        return nodes

    def _use_or_generate_name(self, op, name):
        if name is not None:
            if name not in self.nodes:
                return name
        else:
            name = op

        idx = 0
        while True:
            key = "%s_%d" % (name, idx)
            if key not in self.nodes:
                return key
            idx += 1

    def _add_node(self, op, name, inputs=None, fields=None, extra_fields={}):
        node = Node(self, op, name, inputs, fields, extra_fields)
        if name in self.nodes:
            raise UffException("node already exist")
        self.nodes[name] = node
        return node

    def input(self, shape, dtype=np.float32, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Input", name)
        fields = _create_fields({"shape": shape, "dtype": dtype}, fields)
        return self._add_node("Input", name, fields=fields, extra_fields=extra_fields)

    def const(self, arr, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Const", name)
        if not isinstance(arr, str):
            data_blob = create_data(np.ascontiguousarray(arr).tobytes(), FieldType.blob)
        else:
            data_blob = create_data(arr, FieldType.s)
            # data_blob = create_data(str.encode(arr), FieldType.blob)
        data_blob_ref = self.meta_graph.create_ref("weights_" + name, data_blob)

        fields = _create_fields({
            "shape": arr.shape if hasattr(arr, "shape") else [],
            "dtype": arr.dtype if hasattr(arr, "dtype") else type(arr),
            "values": data_blob_ref
        }, fields)

        return self._add_node("Const", name, fields=fields, extra_fields=extra_fields)

    def conv(self, left_node, right_node, strides,
             padding=None, dilation=None, number_groups=None,
             left_format="NC+", right_format="KC+",
             name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Conv", name)
        fields = _create_fields({
            "padding": padding,
            "strides": strides,
            "dilation": dilation,
            "number_groups": number_groups,
            "inputs_orders": self.meta_graph.create_orders_ref([left_format, right_format])
        }, fields)
        return self._add_node("Conv", name, inputs=[left_node, right_node],
                              fields=fields, extra_fields=extra_fields)

    def conv_transpose(self, input_node, weights_node, shape_node, strides,
                       padding=None, dilation=None, number_groups=None,
                       left_format="NC+", right_format="KC+",
                       name=None, fields=None, extra_fields=None):

        name = self._use_or_generate_name("ConvTranspose", name)
        fields = _create_fields({
            "padding": padding,
            "strides": strides,
            "dilation": dilation,
            "number_groups": number_groups,
            "inputs_orders": self.meta_graph.create_orders_ref([left_format, right_format])
        }, fields)

        return self._add_node("ConvTranspose", name, inputs=[input_node, weights_node, shape_node],
                              fields=fields, extra_fields=extra_fields)

    def pool(self, prev_node, func, kernel, strides, padding=None, data_format="NC+",
             name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Pool", name)
        fields = _create_fields({
            "func": func.lower(),
            "kernel": kernel,
            "padding": padding,
            "strides": strides,
            "inputs_orders": self.meta_graph.create_orders_ref([data_format])
        }, fields)

        return self._add_node("Pool", name, inputs=[prev_node],
                              fields=fields, extra_fields=extra_fields)

    def fully_connected(self, left_node, right_node, left_format="NC", right_format="KC",
                        name=None, fields=None, extra_fields=None):

        name = self._use_or_generate_name("FullyConnected", name)
        fields = _create_fields({
            "inputs_orders": self.meta_graph.create_orders_ref([left_format, right_format])
        }, fields)

        return self._add_node("FullyConnected", name, inputs=[left_node, right_node],
                              fields=fields, extra_fields=extra_fields)

    def lrn(self, prev_node, window_size, alpha, beta, k, data_format="NC+",
            name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("LRN", name)
        fields = _create_fields({
            "window_size": window_size,
            "alpha": alpha,
            "beta": beta,
            "k": k,
            "inputs_orders": self.meta_graph.create_orders_ref([data_format])
        }, fields)

        return self._add_node("LRN", name, inputs=[prev_node],
                              fields=fields, extra_fields=extra_fields)

    def binary(self, left_node, right_node, func, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Binary", name)
        fields = _create_fields({"func": func}, fields)
        return self._add_node("Binary", name, inputs=[left_node, right_node],
                              fields=fields, extra_fields=extra_fields)

    def unary(self, prev_node, func, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Unary", name)
        fields = _create_fields({"func": func}, fields)
        return self._add_node("Unary", name, inputs=[prev_node],
                              fields=fields, extra_fields=extra_fields)

    def reshape(self, prev_node, shape, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Reshape", name)
        return self._add_node("Reshape", name, inputs=[prev_node, shape],
                              fields=fields, extra_fields=extra_fields)

    def transpose(self, prev_node, permutation, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Transpose", name)
        fields = _create_fields({"permutation": permutation}, fields)
        return self._add_node("Transpose", name, inputs=[prev_node],
                              fields=fields, extra_fields=extra_fields)

    def concat(self, inputs, axis, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Concat", name)
        fields = _create_fields({"axis": axis}, fields)
        return self._add_node("Concat", name, inputs=inputs,
                              fields=fields, extra_fields=extra_fields)

    def reduce(self, prev_node, func, axes, keepdims, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Reduce", name)
        if isinstance(axes, int_types):
            axes = [axes]

        fields = _create_fields({"func": func, "axes": axes, "keepdims": keepdims}, fields)
        return self._add_node("Reduce", name, inputs=[prev_node],
                              fields=fields, extra_fields=extra_fields)

    def stack(self, inputs, axis, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Stack", name)
        fields = _create_fields({"axis": axis}, fields)
        return self._add_node("Stack", name, inputs=inputs,
                              fields=fields, extra_fields=extra_fields)

    def shape(self, input_node, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Shape", name)
        return self._add_node("Shape", name, inputs=[input_node],
                              fields=fields, extra_fields=extra_fields)

    def strided_slice(self, input_node, begin_node, end_node, strides_node,
                      begin_mask=0, end_mask=0, shrink_axis_mask=0,
                      name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("StridedSlice", name)
        fields = _create_fields({
            "begin_mask": begin_mask,
            "end_mask": end_mask,
            "shrink_axis_mask": shrink_axis_mask
        }, fields)
        return self._add_node("StridedSlice", name,
                              inputs=[input_node, begin_node, end_node, strides_node],
                              fields=fields, extra_fields=extra_fields)

    def mark_output(self, output, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("MarkOutput", name)
        return self._add_node("MarkOutput", name, inputs=[output],
                              fields=fields, extra_fields=extra_fields)

    def custom_node(self, op, inputs, name=None, fields=None, extra_fields=None):
        self.meta_graph.enable_custom_descriptor()
        op = "_" + op
        name = self._use_or_generate_name(op, name)
        return self._add_node(op, name, inputs=inputs, fields=fields, extra_fields=extra_fields)

    # TODO transform those into Sub-Graph
    def activation(self, prev_node, func, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Activation", name)
        fields = _create_fields({"func": func}, fields)
        return self._add_node("Activation", name, inputs=[prev_node],
                              fields=fields, extra_fields=extra_fields)

    def softmax(self, prev_node, axis, data_format, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Softmax", name)
        fields = _create_fields({
            "axis": axis,
            "inputs_orders": self.meta_graph.create_orders_ref([data_format])
            }, fields)
        return self._add_node("Softmax", name, inputs=[prev_node],
                              fields=fields, extra_fields=extra_fields)

    def batchnorm(self, prev_node, gamma, beta, moving_mean, moving_variance, epsilon,
                  data_format="NC+", name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("BatchNorm", name)
        fields = _create_fields({
            "epsilon": epsilon,
            "inputs_orders": self.meta_graph.create_orders_ref([data_format])
        }, fields)
        return self._add_node("BatchNorm", name,
                              inputs=[prev_node, gamma, beta, moving_mean, moving_variance],
                              fields=fields, extra_fields=extra_fields)

    def squeeze(self, prev_node, name=None, axis=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Squeeze", name)
        fields = _create_fields({
            "axes": axis,
        }, fields)
        return self._add_node("Squeeze", name, inputs=[prev_node],
                              fields=fields, extra_fields=extra_fields)

    def flatten(self, prev_node, name=None, axis=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Flatten", name)
        return self._add_node("Flatten", name, inputs=[prev_node],
                              fields=fields, extra_fields=extra_fields)

    def pad(self, prev_node, pad, name=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Pad", name)
        return self._add_node("Pad", name, inputs=[prev_node, pad],
                              fields=fields, extra_fields=extra_fields)

    def gather(self, inputs, name=None, indices_dtype=None, params_dtype=None,
        validate_indices=None, fields=None, extra_fields=None):
        name = self._use_or_generate_name("Gather", name)
        fields = _create_fields({
            "indices_dtype": indices_dtype,
            "params_dtype": params_dtype,
            "validate_indices": validate_indices,
        }, fields)
        return self._add_node("Gather", name, inputs=inputs,
                              fields=fields, extra_fields=extra_fields)

    def gather_v2(self, inputs, name=None, axis=0, indices_dtype=None, params_dtype=None,
            fields=None, extra_fields=None):
        name = self._use_or_generate_name("GatherV2", name)
        fields = _create_fields({
            "axis": axis,
            "indices_dtype": indices_dtype,
            "params_dtype": params_dtype,
        }, fields)
        return self._add_node("GatherV2", name, inputs=inputs,
                              fields=fields, extra_fields=extra_fields)
