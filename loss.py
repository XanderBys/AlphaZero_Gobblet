import tensorflow as tf

def softmax_crossentropy(y_true, y_pred):
    zeros = tf.zeros(shape=tf.shape(y_true), dtype=tf.float32)
    negatives = tf.fill(tf.shape(y_true), -100.0)
    
    logits = tf.where(tf.equal(y_true, zeros), negatives, y_pred)
    
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=logits)
    
    return loss