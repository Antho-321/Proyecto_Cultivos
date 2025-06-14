import tensorflow as tf

def weighted_categorical_crossentropy(class_weights):
    # class_weights: dict {clase: peso}
    weights = tf.constant([class_weights[i] for i in sorted(class_weights)], dtype=tf.float32)
    def loss(y_true, y_pred):
        # y_true: shape (B, H, W), valores enteros de 0 a C-1
        # y_pred: shape (B, H, W, C), probabilidades
        y_true = tf.cast(y_true[...,0], tf.int32)
        y_true_onehot = tf.one_hot(y_true, depth=len(weights))
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1-7e-7)
        # aplicar peso a la verdadera clase
        weighted_losses = -tf.reduce_sum(y_true_onehot * tf.math.log(y_pred) * weights, axis=-1)
        return tf.reduce_mean(weighted_losses)
    return loss

# Uso
loss_fn = weighted_categorical_crossentropy(class_weights)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

"""
Otra funcion de perdida para probar:

# Focal Loss estándar en tf.keras
def focal_loss(alpha=None, gamma=2.0):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true[...,0], tf.int32)
        y_true_oh = tf.one_hot(y_true, depth=y_pred.shape[-1])
        p_t = tf.reduce_sum(y_true_oh * y_pred, axis=-1) + 1e-7
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        if alpha is not None:
            alpha_t = tf.reduce_sum(y_true_oh * tf.constant(alpha, dtype=tf.float32), axis=-1)
        else:
            alpha_t = 1.0
        loss = -alpha_t * modulating_factor * tf.math.log(p_t)
        return tf.reduce_mean(loss)
    return loss

def weighted_focal_loss(class_weights, gamma=2.0):
    alpha = [class_weights[i] for i in sorted(class_weights)]
    return focal_loss(alpha=alpha, gamma=gamma)

# Al compilar tu modelo:
loss_fn = weighted_focal_loss(class_weights, gamma=2.0)
model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])

loss_fn = weighted_focal_loss(class_weights, gamma=2.0)

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

"""