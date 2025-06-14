import tensorflow as tf

# --------------------------------------------------
# 1. Funciones de soporte para Lovász-Softmax
# --------------------------------------------------

def lovasz_grad(gt_sorted):
    """
    Gradiente de la extensión de Lovász (Ecuación 9 en [Berman et al., CVPR’18]).
    gt_sorted: Tensor de forma [P] con valores 0/1 ordenados por error descendente.
    """
    gts = tf.reduce_sum(gt_sorted)
    intersection = gts - tf.cumsum(gt_sorted)
    union = gts + tf.cumsum(1. - gt_sorted)
    jaccard = 1. - intersection / union
    # gradiente = [j(0), j(1)-j(0), j(2)-j(1), ...]
    return tf.concat([jaccard[:1], jaccard[1:] - jaccard[:-1]], axis=0)

def flatten_probas(probas, labels, ignore=None):
    """
    Aplana predicciones y etiquetas:
      probas: [B, H, W, C] → [P, C], con P=B*H*W
      labels: [B, H, W]       → [P]
    Opcionalmente omite pixeles con etiqueta == ignore.
    """
    probas = tf.reshape(probas, [-1, tf.shape(probas)[-1]])
    labels = tf.reshape(labels, [-1])
    if ignore is None:
        return probas, labels
    valid = tf.not_equal(labels, ignore)
    return tf.boolean_mask(probas, valid), tf.boolean_mask(labels, valid)

def lovasz_softmax_flat(probas, labels, classes='present', ignore=None):
    """
    Pérdida Lovász-Softmax sobre vectores aplanados.
    probas: [P, C], labels: [P] enteros en [0..C-1].
    """
    C = probas.shape[-1]
    losses = []
    # Selección de clases: 'all' o solo las presentes en labels
    if classes == 'all':
        class_range = range(C)
    else:  # 'present'
        class_range = [c for c in range(C)
                       if tf.reduce_sum(tf.cast(labels == c, probas.dtype)) > 0]
    for c in class_range:
        fg = tf.cast(labels == c, probas.dtype)
        errors = tf.abs(fg - probas[:, c])
        errors_sorted, perm = tf.nn.top_k(errors, k=tf.shape(errors)[0], sorted=True)
        fg_sorted = tf.gather(fg, perm)
        grad = lovasz_grad(fg_sorted)
        losses.append(tf.tensordot(errors_sorted, tf.stop_gradient(grad), 1))
    if not losses:
        return tf.constant(0.0)
    return tf.add_n(losses) / tf.cast(len(losses), tf.float32)

def lovasz_softmax_loss(ignore=None, per_image=False):
    """
    Crea la función de pérdida Lovász-Softmax para Keras/TensorFlow.

    Args:
      ignore: etiqueta a ignorar (ej. void) o None.
      per_image: si True, calcula la pérdida por imagen y luego promedia.

    Uso:
      model.compile(loss=lovasz_softmax_loss(), optimizer=...)
    """
    def loss(y_true, y_pred):
        # y_true: [B, H, W, 1] logits o one-hot con canal final; convertimos a int32
        y_true = tf.cast(tf.squeeze(y_true, axis=-1), tf.int32)
        # asumimos logits, aplicamos softmax
        probas = tf.nn.softmax(y_pred, axis=-1)

        if per_image:
            # mapear por cada imagen
            def image_loss(args):
                p, t = args
                p_flat, t_flat = flatten_probas(tf.expand_dims(p,0), tf.expand_dims(t,0), ignore)
                return lovasz_softmax_flat(p_flat, t_flat, ignore=ignore)
            losses = tf.map_fn(image_loss, (probas, y_true), dtype=tf.float32)
            return tf.reduce_mean(losses)
        else:
            # todo el batch de una vez
            p_flat, t_flat = flatten_probas(probas, y_true, ignore)
            return lovasz_softmax_flat(p_flat, t_flat, ignore=ignore)
    return loss
