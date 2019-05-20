try:
    import tensorflow as tf
except ImportError as err:
    raise ImportError("""ERROR: Failed to import module ({})
Please make sure you have TensorFlow installed.
https://www.tensorflow.org/install/
""".format(err))

from graphsurgeon.StaticGraph import *
from graphsurgeon.DynamicGraph import *
from graphsurgeon.node_manipulation import *
from graphsurgeon.recovery import *
import graphsurgeon.extras
__version__ = "0.4.1"
