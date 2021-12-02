import tensorflow as tf

from Generation.transformer_copy.model.layer import EncoderLayer, DecoderLayer


class Encoder(tf.keras.layers.Layer):
    """
    encoder的原始输入x是向量，shape: (batch_size, input_seq_len)，dtype: int64，在call方法里会将x进行编码变成shape: (batch_size, input_seq_len, d_model)传到enc_layers里
    这里把embedding单独抽出来，encoder的输入变成embed_enc_input， shape: (batch_size, input_seq_len, d_model)
    返回x是Tensor，shape: (batch_size, input_seq_len, d_model)
    """

    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, adapter_size=None):
        super(Encoder, self).__init__()

        self.num_layers = num_layers

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate, adapter_size=adapter_size)
                           for _ in range(num_layers)]

        self.dropout = tf.keras.layers.Dropout(rate)

    def call(self, embed_enc_input, training, mask):
        x = self.dropout(embed_enc_input, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(tf.keras.layers.Layer):
    """
    encoder的输入x是向量，shape: (batch_size, target_seq_len)，dtype: int64, 在call方法里会将x进行编码变成shape: (batch_size, target_seq_len, d_model)传到dec_layers里
    输入enc_output就是encoder的输出，shape: (batch_size, input_seq_len, d_model)
    这里把embedding单独抽出来，decoder的输入变成embed_dec_input， shape: (batch_size, target_seq_len, d_model)
    返回x是Tensor，shape: (batch_size, input_seq_len, d_model)
    """

    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1, adapter_size=None):
        super(Decoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.depth = d_model // self.num_heads

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate, adapter_size=adapter_size)
                           for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        self.Wh = tf.keras.layers.Dense(1)
        self.Ws = tf.keras.layers.Dense(1)
        self.Wx = tf.keras.layers.Dense(1)
        self.V = tf.keras.layers.Dense(1)

    def call(self, embed_dec_input, enc_output, training, look_ahead_mask, padding_mask):
        attention_weights = {}

        x = self.dropout(embed_dec_input, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](x, enc_output, training, look_ahead_mask, padding_mask)

            attention_weights[f'decoder_layer{i + 1}_block1'] = block1
            attention_weights[f'decoder_layer{i + 1}_block2'] = block2

        # context vectors
        enc_out_shape = tf.shape(enc_output)  # (batch_size, inp_seq_len, dim)
        context = tf.reshape(enc_output, (enc_out_shape[0], enc_out_shape[1], self.num_heads,
                                          self.depth))  # shape : (batch_size, input_seq_len, num_heads, depth)
        context = tf.transpose(context, [0, 2, 1, 3])  # (batch_size, num_heads, input_seq_len, depth)
        context = tf.expand_dims(context, axis=2)  # (batch_size, num_heads, 1, input_seq_len, depth)

        attn = tf.expand_dims(block2, axis=-1)  # (batch_size, num_heads, target_seq_len, input_seq_len, 1)

        context = context * attn  # (batch_size, num_heads, target_seq_len, input_seq_len, depth)
        context = tf.reduce_sum(context, axis=3)  # (batch_size, num_heads, target_seq_len, depth)
        context = tf.transpose(context, [0, 2, 1, 3])  # (batch_size, target_seq_len, num_heads, depth)
        context = tf.reshape(context, (
            tf.shape(context)[0], tf.shape(context)[1], self.d_model))  # (batch_size, target_seq_len, d_model)

        # P_gens computing
        a = self.Wx(embed_dec_input)
        b = self.Ws(x)
        c = self.Wh(context)
        p_gens = tf.sigmoid(self.V(a + b + c))  # (batch_size, tar_seq_len, 1)

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights, p_gens


class R_Net(tf.keras.layers.Layer):
    def __init__(self, enc_units):
        super(R_Net, self).__init__()

        # The GRU RNN layer processes those vectors sequentially.
        self.bi_gru = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(enc_units,
                                                                        # Return the sequence and state
                                                                        return_sequences=True,
                                                                        recurrent_initializer='glorot_uniform'),
                                                    merge_mode='concat')

        self.dense = tf.keras.layers.Dense(enc_units, activation='tanh')

    def call(self, embed_enc_input):
        vectors = embed_enc_input

        # The GRU processes the embedding sequence.
        #    output shape: (batch, s, enc_units)
        output = self.bi_gru(vectors)
        output = self.dense(output)

        # Returns the new sequence and its state.
        return output
