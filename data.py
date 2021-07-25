import tensorflow as tf
import matplotlib.pyplot as plt

HR_SIZE = 128
SCALE = 4
LR_SIZE = int(HR_SIZE / 4)
BATCH_SIZE = 8


# [====================================================]
# [================ Random Compressions ===============]
# [====================================================]

def random_compression(example):
    hr = example['hr']
    hr_shape = tf.shape(hr)
    compression_idx = tf.random.uniform(shape = (), maxval = 7, dtype = tf.int32)
    
    if compression_idx == 0 or compression_idx == 1:
        # bicubic
        lr = tf.image.resize(hr, [int(hr_shape[0] / SCALE), int(hr_shape[1] / SCALE)], method = 'bicubic')
        lr = tf.cast(tf.round(tf.clip_by_value(lr, 0, 255)), tf.uint8)
    elif compression_idx == 2 or compression_idx == 3:
        # bilinear
        lr = tf.image.resize(hr, [int(hr_shape[0] / SCALE), int(hr_shape[1] / SCALE)], method = 'bilinear')
        lr = tf.cast(tf.round(tf.clip_by_value(lr, 0, 255)), tf.uint8)
    elif compression_idx == 4 or compression_idx == 5:
        # nearest
        lr = tf.image.resize(hr, [int(hr_shape[0] / SCALE), int(hr_shape[1] / SCALE)], method = 'nearest')
        lr = tf.cast(tf.round(tf.clip_by_value(lr, 0, 255)), tf.uint8)
    else:
        # default
        lr = example['lr']
    
    return lr, hr

# [======================================================]
# [============= Spatial Random Augmentations ===========]
# [======================================================]

@tf.function()
def random_crop(lr, hr):
    lr_shape = tf.shape(lr)[:2]

    lr_w = tf.random.uniform(shape = (), maxval = lr_shape[1] - LR_SIZE + 1, dtype = tf.int32)
    lr_h = tf.random.uniform(shape = (), maxval = lr_shape[0] - LR_SIZE + 1, dtype = tf.int32)

    hr_w = lr_w * int(SCALE)
    hr_h = lr_h * int(SCALE)

    lr_cropped = lr[lr_h:lr_h + LR_SIZE, lr_w: lr_w + LR_SIZE]
    hr_cropped = hr[hr_h:hr_h + HR_SIZE, hr_w: hr_w + HR_SIZE]

    return lr_cropped, hr_cropped

@tf.function()
def random_rotate(lr, hr):
    rn = tf.random.uniform(shape = (), maxval = 4, dtype = tf.int32)
    return tf.image.rot90(lr, rn), tf.image.rot90(hr, rn)

@tf.function()
def random_spatial_augmentation(lrs, hrs):
    lrs, hrs = tf.cond(
        tf.random.uniform(shape = (), maxval = 1) < 0.5,
        lambda: (lrs, hrs),
        lambda: random_rotate(lrs, hrs)
    )

    return tf.cast(lrs, tf.float32), tf.cast(hrs, tf.float32)

def visualize_samples(images_lists, titles = None, size = (12, 12), masked = False):
    assert len(images_lists) == len(titles)
    
    cols = len(images_lists)
    
    for images in zip(*images_lists):
        plt.figure(figsize = size)
        for idx, image in enumerate(images):
            plt.subplot(1, cols, idx + 1)
            plt.imshow(tf.cast(tf.round(tf.clip_by_value(image, 0, 255)), tf.uint8))
            plt.axis('off')
            if titles:
                plt.title(titles[idx])
        plt.show()