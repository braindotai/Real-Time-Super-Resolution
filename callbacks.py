import tensorflow as tf

class CheckpointCallback(tf.keras.callbacks.Callback):
    def __init__(self, checkpoint_dir, resume = False, epoch_step = 1):
        super(CheckpointCallback, self).__init__()
        
        self.checkpoint_dir = checkpoint_dir
        self.resume = resume
        self.epoch_step = epoch_step
    
    def setup_checkpoint(self, *args, **kwargs):
        self.checkpoint = tf.train.Checkpoint(
            generator = self.model.generator,
            discriminator = self.model.discriminator,
            generator_optimizer = self.model.generator.optimizer,
            discriminator_optimizer = self.model.discriminator.optimizer
        )
        self.manager = tf.train.CheckpointManager(
            self.checkpoint,
            directory = self.checkpoint_dir,
            checkpoint_name = 'SRGAN',
            max_to_keep = 1
        )

        if self.resume:
            self.load_checkpoint()
        else:
            print('Starting training from scratch...\n')
        
    def on_batch_end(self, batch, *args, **kwargs): 
        if (batch + 1) % int(self.epoch_step * len(train_data)) == 0:
            print(f"\n\nCheckpoint saved to {self.manager.save()}\n")
    
    def load_checkpoint(self):
        if self.manager.latest_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            print(f"Checkpoint restored from '{self.manager.latest_checkpoint}'\n")
        else:
            print("No checkpoints found, initializing from scratch...\n")
    
    def set_lr(self, lr, beta_1 = 0.9):
        print(f'Continuing with learning rate: {lr}')
        self.model.generator.optimizer.beta_1 = beta_1
        self.model.generator.optimizer.learning_rate = lr
        self.model.discriminator.optimizer.beta_1 = beta_1
        self.model.discriminator.optimizer.learning_rate = lr

class ProgressCallback(tf.keras.callbacks.Callback):
    def __init__(self, logs_step, generator_step):
        super(ProgressCallback, self).__init__()

        self.logs_step = logs_step
        self.generator_step = generator_step

    def on_batch_end(self, batch, logs, **kwargs):
        if (batch + 1) % int(self.generator_step * len(train_data)) == 0:
            if self.model.perceptual_finetune:
                visualize_samples(
                    images_lists = (self.model.lrs[:3], self.model.srs[:3], self.model.hrs[:3]),
                    titles = ('Low Resolution', 'Predicted Enhanced', 'High Resolution'),
                    size = (11, 11)
                )
            else:
                visualize_samples(
                    images_lists = (self.model.lrs[:3], self.model.srs[:3]),
                    titles = ('Low Resolution', 'Predicted Enhanced'),
                    size = (7, 7)
                )