import matplotlib.pyplot as plt
import numpy as np


def show_batch(image_batch, label_batch, title='True'):
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title)
    for n in range(30):
        flow = label_batch[n][0]
        ax = plt.subplot(5, 6, n + 1)
        plt.imshow(image_batch[n])
        if title == 'True':
            plt.title("FLOW: {}".format(flow))
        else:
            plt.title('FLOW: {}'.format((flow < 0.5).astype(np.int)))
        plt.axis('off')
