import tensorflow as tf
import tensorflow.keras as keras

def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))


resnet_50 = tf.keras.applications.ResNet50(include_top=True, weights=None)

# import tensorflow.keras.backend as K
sess = keras.backend.get_session()
graph = sess.graph
stats_graph(graph)