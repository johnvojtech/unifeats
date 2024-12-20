#!python3
import keras
import keras_nlp
import numpy as np


def feature_attention_model(word_limit:int, feat_len:int):
    """
    Arguments:
    ----------
    word_limit(int): size of input layer; limit of clipping the words
    feat_len(int): size of output layer; no of possible values of the morphological feature

    Returns:
    --------
    <model>, <attention_model>(keras models):
        <model> returning the classification and <attention_model> returning the attention scores. 
        Models share the initial layers up to the attention layer. 
    """
    inputs = keras.layers.Input(shape=(word_limit,))
    # embedding_layer = keras_nlp.layers.TokenAndPositionEmbedding(65536, word_limit, 64, mask_zero=True)
    embedding_layer = keras.layers.Embedding(65536, 64, mask_zero=True)
    embeddings = embedding_layer(inputs)
    nmask = embedding_layer.compute_mask(inputs)
    mask = keras.ops.cast(nmask, int)
    mask = mask[:, np.newaxis, :] * mask[:, :, np.newaxis]
    mask = keras.ops.cast(mask, bool)
    att_out, att_score = keras.layers.MultiHeadAttention(1, word_limit, dropout=0.5)(embeddings, embeddings, return_attention_scores=True, attention_mask=mask)
    att_score = keras.layers.Multiply()([att_score, keras.ops.cast(mask, int)])
    #print(nmask)

    output = keras.layers.Flatten()(att_out)
    for _ in range(3):
        output = keras.layers.Dense(512, activation="relu")(output)
        output = keras.layers.Dropout(0.2)(output)
    output = keras.layers.Dense(feat_len, activation="softmax")(output)

    model, attention_model = keras.models.Model(inputs=inputs, outputs=output), keras.models.Model(inputs=inputs, outputs=att_score)
    return model, attention_model
