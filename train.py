import os
import tensorflow as tf
from tensorflow.keras import Model, optimizers
import tensorflow_datasets as tfds
from data import *
from models import *
from callbacks import *
from losses import *

train_data = tfds.load(f'div2k/bicubic_x{SCALE}', split = 'train', shuffle_files = True)
train_data = train_data.map(random_compression, num_parallel_calls = tf.data.AUTOTUNE)
train_data = train_data.map(random_crop, num_parallel_calls = tf.data.AUTOTUNE)
train_data = train_data.batch(BATCH_SIZE, drop_remainder = True)
train_data = train_data.map(random_spatial_augmentation, num_parallel_calls = tf.data.AUTOTUNE)

train_data = train_data.prefetch(tf.data.AUTOTUNE)

for lrs, hrs in train_data:
    break

print(lrs.shape, hrs.shape)
print(lrs.dtype, hrs.dtype)
print(tf.reduce_min(lrs), tf.reduce_max(lrs))
print(tf.reduce_min(hrs), tf.reduce_max(hrs))

visualize_samples(images_lists = (lrs[:15], hrs[:15]), titles = ('Low Resolution', 'High Resolution'), size = (8, 8))

class SRGAN(
        Model,
        PixelLossTraining,
        GramStyleTraining,
        VGGContentTraining,
        AdversarialTraining
    ):
    def __init__(
        self,
        generator,
        discriminator,
    ):
        super(SRGAN, self).__init__(self, dynamic = True)

        self.generator = generator
        self.discriminator = discriminator
    
    def compile(
        self,

        generator_optimizer, 
        discriminator_optimizer,

        perceptual_finetune,

        pixel_loss,
        style_loss,
        content_loss,
        adv_loss,

        loss_weights,
    ):
        super(SRGAN, self).compile()

        self.generator.optimizer = generator_optimizer
        self.discriminator.optimizer = discriminator_optimizer

        self.perceptual_finetune = perceptual_finetune

        self.setup_pixel_loss(pixel_loss)
        # self.setup_gram_style_loss(style_loss)
        # uncomment this to utilize style loss function
        self.setup_content_loss(content_loss)
        self.setup_adversarial_loss(adv_loss)

        if self.perceptual_finetune:
            self.loss_weights = loss_weights

    def train_step(self, batch):
        self.lrs = batch[0]
        self.hrs = batch[1]

        if self.perceptual_finetune:
            # [=================== Training Discriminator ===================]

            with tf.GradientTape() as disc_tape, tf.GradientTape() as gen_tape:
                self.srs = self.generator(self.lrs, training = True)

                real_logits = self.discriminator(self.hrs, training = True)
                fake_logits = self.discriminator(self.srs, training = True)

                content_loss = self.loss_weights['content_loss'] * self.content_loss(self.srs, self.hrs)
                gen_adv_loss = self.loss_weights['adv_loss'] * self.gen_adv_loss(fake_logits, real_logits)
                perceptual_loss = content_loss + gen_adv_loss
                
                # style_loss = self.loss_weights['style_loss'] * self.gram_style_loss(self.srs, self.hrs)
                # uncomment this and add it to gen_loss to utilize style loss function

                gen_loss = perceptual_loss

                disc_adv_loss = self.disc_adv_loss(fake_logits, real_logits)
            
            discriminator_gradients = disc_tape.gradient(disc_adv_loss, self.discriminator.trainable_variables)
            generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            
            self.discriminator.optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))
            self.generator.optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

            return {
                'Perceptual Loss': perceptual_loss,
                # 'Style Loss': style_loss,
                'Generator Adv Loss': gen_adv_loss,
                'Discriminator Adv Loss': disc_adv_loss,
            }
        
        else:
            with tf.GradientTape() as gen_tape:
                self.srs = self.generator(self.lrs, training = True)

                pixel_loss = self.pixel_loss(self.srs, self.hrs)

            generator_gradients = gen_tape.gradient(pixel_loss, self.generator.trainable_variables)
            self.generator.optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))

            return {
                'Pixel Loss': pixel_loss,
            }

EPOCHS = 1000
LR = 0.002
BETA_1 = 0.9
BETA_2 = 0.999

PERCEPTUAL_FINETUNE = False
# first train the model for simply minimizing the simple pixel loss
# you can set the LR a bit high like .001 - .004
# once you pixel loss is saturated then set PERCEPTUAL_FINETUNE to True,
# and reduce the LR down to something like .0004, .0002
# keep monitoring your outputs, and manually reduce the lr when you see the outputs aren't improving anymore
# I tried reducing lr on when loss function becomes plateau, but it didn't work as expected

PIXEL_LOSS = 'l1'
STYLE_LOSS = 'l1'
CONTENT_LOSS = 'l1'
# l1 loss type worked much better for all above losses in my experiment
ADV_LOSS = 'ragan' 

LOSS_WEIGHTS = {'content_loss': 1.0, 'adv_loss': 0.09, 'style_loss': 1.0}
# Don't forget to tune this adv_loss weights, observe the outputs per epoch
# you'll get to see artifacts by the adversarial training
# partial artificats would be fine, but when you see the outputs are getting weird
# then reduce the adv_loss weights 

CHECKPOINT_DIR = os.path.join('drive', 'MyDrive', 'Model-Checkpoints', 'Super Resolution')

generator_optimizer = optimizers.Adam(
    learning_rate = LR,
    beta_1 = BETA_1,
    beta_2 = BETA_2
)
discriminator_optimizer = optimizers.Adam(
    learning_rate = LR,
    beta_1 = BETA_1,
    beta_2 = BETA_2
)

generator = Generator()
generator.summary(100)

discriminator = Discriminator()
discriminator.summary(100)


srgan = SRGAN(generator, discriminator)
srgan.compile(
    generator_optimizer = generator_optimizer,
    discriminator_optimizer = discriminator_optimizer,
    
    perceptual_finetune = PERCEPTUAL_FINETUNE,
    pixel_loss = PIXEL_LOSS,
    style_loss = STYLE_LOSS,
    content_loss = CONTENT_LOSS,
    adv_loss = ADV_LOSS,

    loss_weights = LOSS_WEIGHTS
)

ckpt_callback = CheckpointCallback(
    checkpoint_dir = CHECKPOINT_DIR,
    resume = True,
    epoch_step = 4
)
ckpt_callback.set_model(srgan)
ckpt_callback.setup_checkpoint(srgan)
ckpt_callback.set_lr(LR, BETA_1)

srgan.fit(
    train_data.repeat(EPOCHS // 10),
    epochs = 10,
    callbacks = [
        ckpt_callback,
        ProgressCallback(
            logs_step = 0.2,
            generator_step = 2
        )
    ]
)