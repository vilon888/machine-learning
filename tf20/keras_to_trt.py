import time
# import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt


# Inference with TF-TRT frozen graph workflow:
def trt_classfication_test():
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            # First deserialize your frozen graph:
            with tf.io.gfile.GFile("/home/vilon_tao/Projects/machine-learning/tf20/models/freezed_open_shelf_resenet_model.pb", 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            # Now you can create a TensorRT inference graph from your
            # frozen graph:
            trt_graph = trt.create_inference_graph(
                input_graph_def=graph_def,
                outputs=['embeddings/fc_512/BiasAdd'],
                max_batch_size=20,
                max_workspace_size_bytes=2 << 10,
                precision_mode="FP32")

            tf.import_graph_def(trt_graph, name='')
            tf_input = sess.graph.get_tensor_by_name('input_1:0')
            tf_output = sess.graph.get_tensor_by_name('embeddings/fc_512/BiasAdd:0')
            start_time = time.time()
            for i in range(30):
                embeddings = sess.run(tf_output,
                                      feed_dict={tf_input: np.random.rand(20, 96, 96, 3).astype(dtype=np.float32)})
            print('TRT inference with 20 * 96 * 96 * 3 cost: {} ms.'.format((time.time() - start_time)/30))
            # print(embeddings)


def trt_cfe_test():
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session() as sess:
            # First deserialize your frozen graph:
            with tf.io.gfile.GFile("/home/vilon_tao/Projects/machine-learning/tf20/models/freezed_open_shelf_cfe_model.pb", 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
            # Now you can create a TensorRT inference graph from your
            # frozen graph:
            trt_graph = trt.create_inference_graph(
                input_graph_def=graph_def,
                outputs=['loc_conf/loc_conf_concat/concat'],
                max_batch_size=20,
                max_workspace_size_bytes=2 << 10,
                precision_mode="FP32")

            tf.import_graph_def(trt_graph, name='')
            tf_input = sess.graph.get_tensor_by_name('input_1:0')
            tf_output = sess.graph.get_tensor_by_name('loc_conf/loc_conf_concat/concat:0')
            start_time = time.time()
            for i in range(30):
                predictions = sess.run(tf_output,
                                      feed_dict={tf_input: np.random.rand(20, 512, 512, 3).astype(dtype=np.float32)})
            print('TRT inference with 20 * 512 * 512 *3 cost: {} ms.'.format((time.time() - start_time)/30))
            # print(predictions)


if __name__ == '__main__':
    # trt_classfication_test()
    trt_cfe_test()