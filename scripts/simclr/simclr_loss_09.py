import tensorflow as tf

def nt_xent_loss(z_i, z_j, temperature=0.1):
    """Normalized temperature-scaled cross entropy loss"""
    batch_size = tf.shape(z_i)[0]
    z = tf.concat([z_i, z_j], axis=0)  # shape: (2N, D)

    # Normalize
    z = tf.math.l2_normalize(z, axis=1)

    # Cosine similarity matrix
    similarity_matrix = tf.matmul(z, z, transpose_b=True)

    # Mask out self-similarities
    logits_mask = tf.ones_like(similarity_matrix) - tf.eye(2 * batch_size)
    logits = similarity_matrix * logits_mask
    logits /= temperature

    # Positive pair similarity
    labels = tf.range(batch_size)
    labels = tf.concat([labels + batch_size, labels], axis=0)

    positives = tf.linalg.diag_part(tf.matmul(z, tf.roll(z, shift=batch_size, axis=0), transpose_b=True)) / temperature

    labels = tf.cast(labels, tf.int32)
    loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)
    return tf.reduce_mean(loss)