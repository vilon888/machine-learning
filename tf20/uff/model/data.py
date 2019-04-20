import numpy as np

from . import uff_pb2 as uff_pb
from .utils import SimpleObject, int_types
from .exceptions import UffException

FieldType = SimpleObject({field.name: field.name for field in uff_pb.Data.DESCRIPTOR.fields})


class List(list):

    def __init__(self, list_type, *args, **kwargs):
        self.list_type = list_type
        return super(List, self).__init__(*args, **kwargs)


def infer_field_type(elt):
    if isinstance(elt, str):
        return FieldType.s
    if isinstance(elt, bool):
        return FieldType.b
    if isinstance(elt, int_types):
        return FieldType.i
    if isinstance(elt, float):
        return FieldType.d
    if isinstance(elt, List):
        if type(elt.list_type) is type:
            return str(infer_field_type(elt.list_type())) + "_list"
        if not elt.list_type.endswith("_list"):
            return str(elt.list_type) + "_list"
    if isinstance(elt, list):
        raise UffException("unsupported list type")

    if isinstance(elt, np.dtype):
        elt = elt.type
    if isinstance(elt, type) and issubclass(elt, np.number):
        return FieldType.dtype

    return ""


_DTYPE_NP_TO_UFF = {
    np.int8: uff_pb.DT_INT8,
    np.int16: uff_pb.DT_INT16,
    np.int32: uff_pb.DT_INT32,
    np.int64: uff_pb.DT_INT64,
    np.float16: uff_pb.DT_FLOAT16,
    np.float32: uff_pb.DT_FLOAT32
}


def _create_dtype(dtype):
    if isinstance(dtype, np.dtype):
        dtype = dtype.type
    if dtype in _DTYPE_NP_TO_UFF:
        return _DTYPE_NP_TO_UFF[dtype]
    if dtype not in uff_pb.DataType.values() or dtype == uff_pb.DT_INVALID:
        raise UffException("dtype {} unknown".format(dtype))
    return dtype


def _create_dim_orders(orders):
    for k, order in orders.items():
        if not isinstance(order, uff_pb.ListINT64):
            orders[k] = uff_pb.ListINT64(val=order)
    return uff_pb.DimensionOrders(orders=orders)


def _create_list_dim_orders(val):
    dims_orders_list = []
    for dim_orders in val:
        if isinstance(dim_orders, uff_pb.DimensionOrders):
            dims_orders_list.append(dim_orders)
        else:
            dims_orders_list.append(_create_dim_orders(dim_orders))
    return uff_pb.ListDimensionOrders(val=dims_orders_list)


_CTOR_LIST = {
    FieldType.s_list: uff_pb.ListString,
    FieldType.b_list: uff_pb.ListBool,
    FieldType.d_list: uff_pb.ListDouble,
    FieldType.i_list: uff_pb.ListINT64,
    FieldType.dtype_list: uff_pb.ListDataType,
    FieldType.dim_orders_list: _create_list_dim_orders,
}


def create_data(elt, field_type=None):
    if elt is None:
        return uff_pb.Data()

    if field_type is None:
        field_type = infer_field_type(elt)

        # FIXME: All of this
    assert(field_type != FieldType.ref)
    # if __debug__:
    #     print("Creating data of type: " + str(type(elt)) + " given FieldType: " + field_type)

    if field_type.endswith("_list"):
        try:
            return uff_pb.Data(**{field_type: _CTOR_LIST[field_type](val=elt)})
        except Exception:
            return uff_pb.Data()

    if field_type == FieldType.dim_orders:
        return uff_pb.Data(dim_orders=_create_dim_orders(elt))

    if field_type == FieldType.dtype:
        try:
            return uff_pb.Data(dtype=_create_dtype(elt))
        except Exception:
            return uff_pb.Data(dtype=7)
    try:
        return uff_pb.Data(**{field_type: elt})
    except Exception:
        return uff_pb.Data()


def convert_to_debug_data(data):
    if data.WhichOneof("data_oneof") == FieldType.blob and len(data.blob) > 32:
        return uff_pb.Data(blob=str.encode("(...%d bytes skipped...)" % len(data.blob)))
    return data
