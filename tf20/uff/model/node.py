import traceback

from . import uff_pb2 as uff_pb
from .data import create_data, convert_to_debug_data


class Node(object):

    def __init__(self, graph, op, name, inputs=None, fields=None, extra_fields=None):
        self.graph = graph
        self.inputs = inputs if inputs else []
        self.fields = fields if fields else {}
        self.extra_fields = extra_fields if extra_fields else {}
        self.name = name
        self.op = op
        self._trace = traceback.format_stack()[:-1]

    def _convert_fields(self, fields, debug):
        descriptor = self.graph.meta_graph.descriptor

        ret_fields = {}
        for k, v in fields.items():
            if v is None:
                continue
            if not isinstance(v, uff_pb.Data):
                if self.op in descriptor:
                    field_type = descriptor[self.op].get_field_type(k)
                    ret_fields[k] = create_data(v, field_type)
                else:
                    ret_fields[k] = create_data(v)
            else:
                ret_fields[k] = v
            if debug:
                ret_fields[k] = convert_to_debug_data(ret_fields[k])
        return ret_fields

    def to_uff(self, debug=False):
        return uff_pb.Node(id=self.name,
                           inputs=[i.name if isinstance(i, Node) else i for i in self.inputs],
                           operation=self.op,
                           fields=self._convert_fields(self.fields, debug),
                           extra_fields=self._convert_fields(self.extra_fields, debug))
