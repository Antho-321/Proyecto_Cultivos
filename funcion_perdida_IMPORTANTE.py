
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

def weighted_focal_lovasz_loss(class_weights, gamma=2.0, lovasz_weight=1.0):
    # Focal modulado
    fl = weighted_focal_loss(class_weights, gamma)
    # Lovász-Softmax
    ls = lovasz_softmax_loss()

    def loss(y_true, y_pred):
        return fl(y_true, y_pred) + lovasz_weight * ls(y_true, y_pred)

    return loss

"""