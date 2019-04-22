from . import uff_pb2 as uff_pb
from .descriptor import Descriptor, DescriptorOp
from .data import FieldType


CORE_DESCRIPTOR = Descriptor(None, 0x1, False, {
    "Input":
        DescriptorOp().inputs_size(0)
        .field(FieldType.dtype, "dtype", uff_pb.DT_FLOAT32)
        .field(FieldType.i_list, "shape"),

    "Identity":
        DescriptorOp().inputs_size(1),

    "Const":
        DescriptorOp().inputs_size(0)
        .field(FieldType.blob, "values").ref_field("values")  # force reference checking
        .field(FieldType.dtype, "dtype")
        .field(FieldType.i_list, "shape"),

    "Conv":
        DescriptorOp().inputs_size(2)  # input, weights
        .fieldOrders()
        .field(FieldType.i, "number_groups", 1)
        .field(FieldType.i_list, "dilation", [])  # FIXME, wrong default value
        .field(FieldType.i_list, "strides", [])   # FIXME, wrong default value
        .field(FieldType.i_list, "padding", []),  # FIXME, wrong default value

    "ConvTranspose":
        DescriptorOp().inputs_size(3)  # input, weights, shape
        .fieldOrders(2)
        .field(FieldType.i, "number_groups", 1)
        .field(FieldType.i_list, "dilation", [])  # FIXME, wrong default value
        .field(FieldType.i_list, "strides", [])   # FIXME, wrong default value
        .field(FieldType.i_list, "padding", []),  # FIXME, wrong default value

    "Pool":
        DescriptorOp().inputs_size(1)
        .fieldOrders()
        .field_enum("func", ["max", "avg"])
        .field(FieldType.i_list, "kernel", [])    # FIXME, wrong default value
        .field(FieldType.i_list, "strides", [])   # FIXME, wrong default value
        .field(FieldType.i_list, "padding", []),  # FIXME, wrong default value

    "FullyConnected":
        DescriptorOp().inputs_size(2)  # input, weights
        .fieldOrders(),

    "LRN":
        DescriptorOp().inputs_size(1)
        .fieldOrders()
        .field(FieldType.i, "window_size")
        .field(FieldType.d, "alpha")
        .field(FieldType.d, "beta")
        .field(FieldType.d, "k"),

    "Binary":
        DescriptorOp().inputs_size(2)
        .field_enum("func", ["min", "max", "mul", "sub", "div", "add", "pow"]),

    "Unary":
        DescriptorOp().inputs_size(1)
        .field_enum("func", ["neg", "exp", "log", "abs", "sqrt", "rsqrt", "square", "sin", "cos", "tan", "sinh", "cosh", "asin", "acos", "atan", "asinh", "acosh", "atanh", "ceil", "floor"]),

    "ExpandDims":
        DescriptorOp().inputs_size(1)
        .field(FieldType.i, "axis"),

    "ArgMax":
        DescriptorOp().inputs_size(1)
        .field(FieldType.i, "axis"),

    "ArgMin":
        DescriptorOp().inputs_size(1)
        .field(FieldType.i, "axis"),

    "Reshape":
        DescriptorOp().inputs_size(2),  # input, shape

    "Transpose":
        DescriptorOp().inputs_size(1)
        .field(FieldType.i_list, "permutation"),

    "Reduce":
        DescriptorOp().inputs_size(1)
        .field_enum("func", ["sum", "prod", "max", "min", "mean"])
        .field(FieldType.i_list, "axes")
        .field(FieldType.b, "keepdims"),

    "Concat":
        DescriptorOp().has_inputs()
        .field(FieldType.i, "axis"),

    "Stack":
        DescriptorOp().has_inputs()
        .field(FieldType.i, "axis"),

    "Shape":
        DescriptorOp().inputs_size(1),

    "StridedSlice":
        DescriptorOp().inputs_size(4)   # input, begin, end, strides
        .field(FieldType.i, "begin_mask")
        .field(FieldType.i, "end_mask")
        .field(FieldType.i, "shrink_axis_mask"),

    "MarkOutput":
        DescriptorOp().has_inputs(),

    # TODO
    # LCN
    # Select
    # Embed

    # TODO: Temporary - to remove when graph pattern match will be implemented in TensoRT importer
    # we will keep the helper function in the Graph for those though
    "Activation":
        DescriptorOp().inputs_size(1)
        .field_enum("func", ["relu", "relu6", "sigmoid", "tanh", "elu", "selu", "softsign", "softplus"]),

    "Softmax":
        DescriptorOp().inputs_size(1)
        .fieldOrders(1)
        .field(FieldType.i, "axis"),

    "BatchNorm":
        DescriptorOp().inputs_size(5)  # input, gamma, beta, moving_mean, moving_variance
        .fieldOrders(1)
        .field(FieldType.d, "epsilon"),

    "Squeeze":
        DescriptorOp().inputs_size(1)
        .field(FieldType.i_list, "axes", []),

    "Flatten":
        DescriptorOp().inputs_size(1),

    "Pad":
        DescriptorOp().inputs_size(2),  # input, padding

    "Gather":
        DescriptorOp().inputs_size(2)
        .field(FieldType.dtype, "indices_dtype")
        .field(FieldType.dtype, "params_dtype")
        .field(FieldType.b, "validate_indices"),  # indices, input

    "GatherV2":
        DescriptorOp().inputs_size(2)
        .field(FieldType.i, "axis")
        .field(FieldType.dtype, "indices_dtype")
        .field(FieldType.dtype, "params_dtype"),
    # END TODO
})


CUSTOM_DESCRIPTOR = Descriptor("custom", 0x1, False, {}).add_regex_operator("_.+")
