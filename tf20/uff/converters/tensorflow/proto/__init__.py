import sys
import os

# bypass the protobuf limitation with relative import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _tensorflow_core.graph_pb2 import GraphDef
from _tensorflow_core.node_def_pb2 import NodeDef
from _tensorflow_core.attr_value_pb2 import AttrValue

sys.path.pop(0)
