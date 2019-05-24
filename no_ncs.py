import tensorflow as tf

path_to_pb = "./model/retrained_graph.pb"

def load_pb(path_to_pb):
    with tf.gfile.GFile(path_to_pb, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
        return graph

sess = load_pb(path_to_pb)

input = graph.get_tensor_by_name('input:0')
output = graph.get_tensor_by_name('output:0')

sess.run(output, feed_dict={input: some_data})