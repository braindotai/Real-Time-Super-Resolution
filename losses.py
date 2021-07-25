import tensorflow as tf
from tensorflow.keras import losses, applications, Model

class PixelLossTraining:
    def setup_pixel_loss(self, pixel_loss):
        if pixel_loss == 'l1':
            self.pixel_loss_type = losses.MeanAbsoluteError()
        elif pixel_loss == 'l2':
            self.pixel_loss_type = losses.MeanSquaredError()

    @tf.function
    def pixel_loss(self, srs, hrs):
        return self.pixel_loss_type(hrs, srs)

class VGGContentTraining:
    def setup_content_loss(self, content_loss):
        if content_loss == 'l1':
            self.content_loss_type = losses.MeanAbsoluteError()
        elif content_loss == 'l2':
            self.content_loss_type = losses.MeanSquaredError()
        
        vgg = applications.VGG19(
            input_shape = (224, 224, 3),
            include_top = False,
            weights = 'imagenet'
        )
        
        vgg.layers[5].activation = None
        vgg.layers[10].activation = None
        vgg.layers[20].activation = None

        self.feature_extrator = Model(
            inputs = vgg.input,
            outputs = [
                vgg.layers[5].output,
                vgg.layers[10].output,
                vgg.layers[20].output
            ]
        )
        for layer in self.feature_extrator.layers:
            layer.trainable = False
    
    @tf.function
    def content_loss(self, srs, hrs):
        srs = applications.vgg19.preprocess_input(tf.image.resize(srs, (224, 224)))
        hrs = applications.vgg19.preprocess_input(tf.image.resize(hrs, (224, 224)))
        
        srs_features = self.feature_extrator(srs)
        hrs_features = self.feature_extrator(hrs)

        loss = 0.0
        for srs_feature, hrs_feature in zip(srs_features, hrs_features):
            loss += self.content_loss_type(hrs_feature / 12.75, srs_feature / 12.75)

        return loss

class GramStyleTraining:
    # I tried using this loss but didn't see notice any significant help
    # so after first few epochs of training I stopped optimizing the model for this loss to save some time
    def setup_gram_style_loss(self, style_loss):
        if style_loss == 'l1':
            self.style_loss_type = losses.MeanAbsoluteError()
        elif style_loss == 'l2':
            self.style_loss_type = losses.MeanSquaredError()
        
        efficientnet = applications.EfficientNetB4(
            input_shape = (224, 224, 3),
            include_top = False,
            weights = 'imagenet'
        )
        
        self.style_features_extractor = Model(
            inputs = efficientnet.input,
            outputs = [
                # efficientnet.layers[25].output,
                # efficientnet.layers[84].output,
                # efficientnet.layers[143].output,
                efficientnet.layers[320].output,
                # efficientnet.layers[467].output,
            ]
        )
        for layer in self.style_features_extractor.layers:
            layer.trainable = False

    @tf.function
    def gram_matrix(self, features):
        features = tf.transpose(features, (0, 3, 1, 2)) # (-1, C, H, W)
        features_a = tf.reshape(features, (tf.shape(features)[0], tf.shape(features)[1], -1)) # (-1, C, H * W)
        features_b = tf.reshape(features, (tf.shape(features)[0], -1, tf.shape(features)[1])) # (-1, H * W, C)
        
        return tf.linalg.matmul(features_a, features_b) # (-1, C, C)

    @tf.function
    def gram_style_loss(self, srs, hrs):
        srs = applications.efficientnet.preprocess_input(tf.image.resize(srs, (224, 224)))
        hrs = applications.efficientnet.preprocess_input(tf.image.resize(hrs, (224, 224)))

        srs_features = self.style_features_extractor(srs) # (2, -1, H, W, C)
        hrs_features = self.style_features_extractor(hrs) # (2, -1, H, W, C)

        # style_loss = 0.0
        # for srs_feature, hrs_feature in zip(srs_features, hrs_features):
        srs_gram = self.gram_matrix(srs_features)
        hrs_gram = self.gram_matrix(hrs_features)

        style_loss = self.style_loss_type(hrs_gram, srs_gram)

        return style_loss

class AdversarialTraining:
    def setup_adversarial_loss(self, adv_loss):
        self.adv_loss_type = adv_loss
        self.binary_cross_entropy = losses.BinaryCrossentropy(from_logits = True)

    @tf.function
    def gen_adv_loss(self, fake_logits, real_logits = None):
        if self.adv_loss_type == 'gan':
            loss = self.binary_cross_entropy(tf.ones_like(fake_logits), fake_logits)
        
        elif self.adv_loss_type == 'ragan':
            real_loss = self.binary_cross_entropy(tf.ones_like(fake_logits), fake_logits - tf.reduce_mean(real_logits))
            fake_loss = self.binary_cross_entropy(tf.zeros_like(real_logits), real_logits - tf.reduce_mean(fake_logits))
            loss = real_loss + fake_loss
        
        return loss
        
    @tf.function
    def disc_adv_loss(self, fake_logits, real_logits):
        if self.adv_loss_type == 'gan':
            real_loss = self.binary_cross_entropy(tf.ones_like(real_logits), real_logits)
            fake_loss = self.binary_cross_entropy(tf.zeros_like(fake_logits), fake_logits)
        
        elif self.adv_loss_type == 'ragan':
            real_loss = self.binary_cross_entropy(tf.ones_like(real_logits), real_logits - tf.reduce_mean(fake_logits))
            fake_loss = self.binary_cross_entropy(tf.zeros_like(fake_logits), fake_logits - tf.reduce_mean(real_logits))
        
        return real_loss + fake_loss