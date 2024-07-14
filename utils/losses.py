import tensorflow as tf

class DiceLoss(tf.keras.losses.Loss):
    def __init__(self, name='dice_loss'):
        super(DiceLoss, self).__init__(name=name)

    def call(self, y_true, y_pred):
        smooth = 1e-15
        dice = 0
        num_classes = y_pred.shape[-1]
        for c in range(num_classes):
            intersection = tf.reduce_sum(y_true[..., c] * y_pred[..., c])
            union = tf.reduce_sum(y_true[..., c]) + tf.reduce_sum(y_pred[..., c])
            dice += (2. * intersection + smooth) / (union + smooth)
        return 1.0 - dice / num_classes

class TverskyLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.5, beta=0.5, name='tversky_loss'):
        super(TverskyLoss, self).__init__(name=name)
        self.alpha = alpha
        self.beta = beta

    def call(self, y_true, y_pred):
        smooth = 1e-15
        tversky = 0
        num_classes = y_pred.shape[-1]
        for c in range(num_classes):
            true_positives = tf.reduce_sum(y_true[..., c] * y_pred[..., c])
            false_negatives = tf.reduce_sum(y_true[..., c] * (1 - y_pred[..., c]))
            false_positives = tf.reduce_sum((1 - y_true[..., c]) * y_pred[..., c])
            tversky += (true_positives + smooth) / (true_positives + self.alpha * false_negatives + self.beta * false_positives + smooth)
        return 1.0 - tversky / num_classes

BCELoss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#alpha and beta here if you want to use TverskyLoss
alpha = 0.5
beta = 1 - alpha
