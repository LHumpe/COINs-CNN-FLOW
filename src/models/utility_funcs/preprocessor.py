import tensorflow as tf
import numpy as np
import scipy.ndimage as ndimage

tf.get_logger().setLevel('WARN')
AUTOTUNE = tf.data.experimental.AUTOTUNE

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


def random_rotate_image(image):
    image = ndimage.rotate(image, np.random.uniform(-10, 10), reshape=False)
    return image


def get_label(file_path, label_pos=-2):
    parts = tf.strings.split(file_path, "_")
    if label_pos == 'none':
        flow = file_path
    else:
        flow = tf.strings.to_number(parts[label_pos], out_type=tf.float32)
    return [flow]


class Preprocessor:

    def __init__(self, batch_size=50, img_height=150, img_width=150,
                 pad=True, saturation=True, brightness=True, flip=True,
                 quality=True, normalize=True, resnet50=True, rotate=True, label_pos=-2):
        self.BATCH_SIZE = batch_size
        self.IMG_HEIGHT = img_height
        self.IMG_WIDTH = img_width
        self.pad = pad
        self.saturation = saturation
        self.brightness = brightness
        self.flip = flip
        self.quality = quality
        self.normalize = normalize
        self.resnet50 = resnet50
        self.rotate = rotate
        self.label_pos = label_pos

    def decode_img(self, img):
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.cast(img, tf.float32)

        if self.rotate:
            # Random Rotation
            im_shape = img.shape
            [img, ] = tf.py_function(random_rotate_image, [img], [tf.float32])
            img.set_shape(im_shape)

        if self.resnet50:
            # Preprocessing for ResNet-50 Architecture
            img = tf.keras.applications.resnet.preprocess_input(img)

        if self.normalize:
            # Image Standardization
            img = tf.image.per_image_standardization(img)

        if self.saturation:
            # Random Saturation
            img = tf.image.random_saturation(img, lower=1, upper=3, seed=123)

        if self.brightness:
            # Random Brightness
            img = tf.image.random_brightness(img, max_delta=0.25, seed=123)

        if self.flip:
            # Flip Vertically
            img = tf.image.random_flip_left_right(img, seed=123)

        if self.quality:
            # Image Quality
            tf.image.random_jpeg_quality(img, 75, 95, seed=123)

        if self.pad:
            return tf.image.resize_with_pad(img, self.IMG_WIDTH, self.IMG_HEIGHT)
        else:
            return tf.image.resize(img, [self.IMG_WIDTH, self.IMG_HEIGHT])

    @tf.function
    def combine_data(self, file_path):
        label = get_label(file_path, self.label_pos)
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        return img, label

    def prepare_for_training(self, ds, cache=True, shuffle_buffer_size=1000):
        if cache:
            if isinstance(cache, str):
                ds = ds.cache(cache)
            else:
                ds = ds.cache()

        ds = ds.shuffle(buffer_size=shuffle_buffer_size)

        ds = ds.batch(self.BATCH_SIZE)

        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds
