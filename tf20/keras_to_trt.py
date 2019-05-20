import time
# import tensorflow as tf
import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.compiler.tensorrt import trt_convert as trt
from tensorflow.python.framework import graph_io


# Inference with TF-TRT frozen graph workflow:
def trt_classfication_test():
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
            # First deserialize your frozen graph:
            with tf.io.gfile.GFile("./models/trt_model/classfication.pb", 'rb') as f:
            # with tf.io.gfile.GFile("/home/vilon_tao/Projects/machine-learning/tf20/models/freezed_open_shelf_resenet_model.pb", 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            # count how many ops in frozen model

            print('before trt:>>>>>>>>>>>>>>>>>')
            for n in graph_def.node:
                print(n.op)
            trt_engine_ops = len([1 for n in graph_def.node if str(n.op) == 'TRTEngineOp'])
            print("numb. of trt_engine_ops in frozen_graph:", trt_engine_ops)
            all_ops = len([1 for n in graph_def.node])
            print("numb. of all_ops in frozen_graph:", all_ops)
            # Now you can create a TensorRT inference graph from your
            # frozen graph:
            trt_graph = trt.create_inference_graph(
                input_graph_def=graph_def,
                outputs=['embeddings/fc_512/BiasAdd'],
                max_batch_size=20,
                # is_dynamic_op=True,
                # maximum_cached_engines=20,
                # cached_engine_batches=[20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20],
                max_workspace_size_bytes=2 << 11,
                precision_mode="FP32")

            # count how many ops in trt_graph
            print('after trt:>>>>>>>>>>>>>>>>>')
            for n in trt_graph.node:
                print (n.op)

            trt_engine_ops = len([1 for n in trt_graph.node if str(n.op) == 'TRTEngineOp'])
            print("numb. of trt_engine_ops in trt_graph", trt_engine_ops)
            all_ops = len([1 for n in trt_graph.node])
            print("numb. of all_ops in in trt_graph:", all_ops)

            # graph_io.write_graph(trt_graph, './models/trt_model/', 'classfication.pb', as_text=False)
            # tf.import_graph_def(graph_def, name='')
            tf.import_graph_def(trt_graph, name='')

            tf_input = sess.graph.get_tensor_by_name('input_1:0')

            tf_output = sess.graph.get_tensor_by_name('embeddings/fc_512/BiasAdd:0')
            start_time = time.time()
            for i in range(100):
                embeddings = sess.run(tf_output,
                                      feed_dict={tf_input: np.random.rand(8, 96, 96, 3).astype(dtype=np.float32)})
            print('TRT inference with 8 * 96 * 96 * 3 cost: {} ms.'.format(1000 * (time.time() - start_time)/100))
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
            # tf.import_graph_def(graph_def, name='')
            tf_input = sess.graph.get_tensor_by_name('input_1:0')
            tf_output = sess.graph.get_tensor_by_name('loc_conf/loc_conf_concat/concat:0')
            start_time = time.time()
            for i in range(100):
                predictions = sess.run(tf_output,
                                      feed_dict={tf_input: np.random.rand(8, 512, 512, 3).astype(dtype=np.float32)})
            print('TRT inference with 8 * 512 * 512 *3 cost: {} ms.'.format(1000 * (time.time() - start_time)/100))
            # print(predictions)


if __name__ == '__main__':
    trt_classfication_test()
    trt_cfe_test()