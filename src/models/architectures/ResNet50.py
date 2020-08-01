import tensorflow as tf


def compile_architecture(tf, img_height, img_width):
    base_model = tf.keras.applications.ResNet50(input_shape=(img_height, img_width, 3),
                                                include_top=False,
                                                weights='imagenet')

    base_model.trainable = False
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    dropout_layer = tf.keras.layers.Dropout(0.4)
    prediction_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')

    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        dropout_layer,
        prediction_layer
    ])

    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.0, nesterov=True),
                  loss='binary_crossentropy',
                  metrics=['acc'])

    return model
