import tensorflow as tf
from Generation.transformer_copy.model.attention import MultiHeadAttention
from Generation.transformer_copy.model.ffn import point_wise_feed_forward_network
from Generation.transformer_copy.model.adapter import Adapter


class EncoderLayer(tf.keras.layers.Layer):
    """
    encoder_layer的输入x是Tensor，shape: (batch_size, input_seq_len, d_model)
    返回out2是Tensor，shape: (batch_size, input_seq_len, d_model)
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1, adapter_size=None):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.ffn1 = point_wise_feed_forward_network(d_model, dff)
        self.ffn2 = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self.adapter1 = Adapter(d_model, adapter_size)
        self.adapter2 = Adapter(d_model, adapter_size)

    def call(self, x, training, mask):
        # ---MHA---
        attn_output, _ = self.mha(x, x, x, mask)  # MHA(self), (batch_size, input_seq_len, d_model)
        # ---FFN---
        attn_output = self.ffn1(attn_output)
        # ---Adapter---
        attn_output = self.adapter1(attn_output)
        # ---Add&Norm---
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)  # Add & Norm, (batch_size, input_seq_len, d_model)

        # ---FFN---
        ffn_output = self.ffn(out1)  # Feed Forward, (batch_size, input_seq_len, d_model)
        ffn_output = self.ffn2(ffn_output)  # Feed Forward, (batch_size, input_seq_len, d_model)
        # ---Adapter---
        ffn_output = self.adapter2(ffn_output)
        # ---Add&Norm---
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # Add & norm, (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(tf.keras.layers.Layer):
    """
    decoder_layer的输入, x是Tensor，shape: (batch_size, target_seq_len, d_model);
    enc_output是Tensor, shape: (batch_size, input_seq_len, d_model)
    返回out3是Tensor，shape: (batch_size, target_seq_len, d_model)
    """

    def __init__(self, d_model, num_heads, dff, rate=0.1, adapter_size=None):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)

        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.ffn1 = point_wise_feed_forward_network(d_model, dff)
        self.ffn2 = point_wise_feed_forward_network(d_model, dff)
        self.ffn3 = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

        # adapter
        self.adapter1 = Adapter(d_model, adapter_size)
        self.adapter2 = Adapter(d_model, adapter_size)
        self.adapter3 = Adapter(d_model, adapter_size)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        # ---MHA---
        attn1, attn_weights_block1 = self.mha1(x, x, x,
                                               look_ahead_mask)  # Masked MHA(self), (batch_size, target_seq_len, d_model)
        # ---FFN---
        attn1 = self.ffn1(attn1)
        # ---Adapter---
        attn1 = self.adapter1(attn1)
        # ---Add&Norm---
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)  # Add & Norm

        # ---MHA---
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1,
                                               padding_mask)  # MHA(), (batch_size, target_seq_len, d_model)
        # ---FFN---
        attn2 = self.ffn2(attn2)
        # ---Adapter---
        attn2 = self.adapter2(attn2)
        # ---Add&Norm---
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # Add & Norm, (batch_size, target_seq_len, d_model)

        # ---FFN---
        ffn_output = self.ffn(out2)  # Feed Forward, (batch_size, target_seq_len, d_model)
        ffn_output = self.ffn3(ffn_output)  # Feed Forward, (batch_size, target_seq_len, d_model)
        # ---Adapter---
        ffn_output = self.adapter3(ffn_output)
        # ---Add&Norm---
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # Add & Norm, (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2
