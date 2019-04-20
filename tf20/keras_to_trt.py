import tensorflow as tf

# import tensorflow.python.compiler.tensorrt as trt
# import tensorflow.contrib.tensorrt as trt
from tensorflow.python.compiler.tensorrt import trt_convert as trt
# Inference with TF-TRT frozen graph workflow:
graph = tf.compat.v1.Graph()
with graph.as_default():
    with tf.compat.v1.Session() as sess:
        # First deserialize your frozen graph:
        with tf.io.gfile.GFile("/home/vilon_tao/Projects/machine-learning/tf20/models/freezed_open_shelf_resenet_model.pb", 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        # Now you can create a TensorRT inference graph from your
        # frozen graph:
        trt_graph = trt.create_inference_graph(
            input_graph_def=graph_def,
            outputs=['embeddings/fc_512/BiasAdd','classification/MatMul'],
            max_batch_size=20,
            max_workspace_size_bytes=2 << 10,
            precision_mode="FP32")
        # Import the TensorRT graph into a new graph and run:
        output_node = tf.import_graph_def(
            trt_graph,
            return_elements=['embeddings/fc_512/BiasAdd','classification/MatMul'])
        sess.run(output_node)