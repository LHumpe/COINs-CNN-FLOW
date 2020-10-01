import tensorflow as tf

model_path = 'data/AffWIld/vggface/'


with tf.compat.v1.Session() as sess:
    saver = tf.compat.v1.train.import_meta_graph(model_path + 'model.ckpt-975.meta')
    saver.restore(sess, model_path + 'model.ckpt-975')
    tensor_variable = tf.compat.v1.all_variables()
    graph = tf.compat.v1.get_default_graph()
    for tensor_var in tensor_variable:
        print(tensor_var)

