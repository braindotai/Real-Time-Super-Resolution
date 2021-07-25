import tensorflow as tf
from data import visualize_samples
from models import Generator

generator = Generator()
generator.load_weights('weights/GeneratorVG4(1520).h5')

def enhance_image(lr_image = None, lr_path = None, sr_path = None, visualize = True, size = (20, 16)):
    assert any([lr_image is not None, lr_path])
    if lr_path:
        lr_image = tf.image.decode_jpeg(tf.io.read_file(f"{lr_path}"), channels = 3)

    sr_image = generator(tf.expand_dims(lr_image, 0), training = False)[0]
    sr_image = tf.clip_by_value(sr_image, 0, 255)
    sr_image = tf.round(sr_image)
    sr_image = tf.cast(sr_image, tf.uint8)

    if visualize:
        visualize_samples(images_lists = [[lr_image], [sr_image]], titles = ['LR Image', 'SR_Image'], size = size)

    if sr_path:
        tf.io.write_file(sr_path, tf.image.encode_jpeg(sr_image))

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description = 'Super Resolution for Real Time Image Enhancement')
    parser.add_argument('--lr-path', type = str, help = 'Path to the low resolution image.')
    parser.add_argument('--sr-path', type = str, default = None, help = 'Output path where the enhanced image would be saved.')

    args = parser.parse_args()

    enhance_image(
        lr_path = args.lr_path,
        sr_path = args.sr_path
    )
