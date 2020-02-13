import deepdish as dd
import matplotlib.pyplot as plt


def image_sequence(config):
    # Load the image files
    image_stream = dd.io.load(config['image_save_path'] + 'rgb_depth_seg.h5')

    # Figure
    fig, ax = plt.subplots(1, 3, figsize=[12, 5])
    titles = ['RGB Image', 'Depth Image', 'Segmented Image']
    plt.tight_layout()

    for stream in image_stream:
        for i in range(len(titles)):
            ax[i].imshow(stream[i], origin='lower')
            ax[i].title.set_text(titles[i])
        plt.pause(0.01)

        for i in range(len(titles)):
            ax[i].cla()