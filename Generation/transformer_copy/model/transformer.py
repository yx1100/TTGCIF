import tensorflow as tf
from Generation.transformer_copy.model.coder import Encoder, Decoder, R_Net
from Generation.transformer_copy.module.utils import _calc_final_dist
from Generation.transformer_copy.module.loss_and_metrics import distance


class Transformer(tf.keras.Model):
    """
    transformer的输入有2个，
    inp是传给encoder的，shape: (batch_size, imp_seq_len), dtype: int64
    tar是传给decoder的，shape: (batch_size, tar_seq_len), dtype: int64
    返回值final_output，shape: (batch_size, tar_seq_len, vocab_size)
    """

    # 参数vocab_size作为Encoder和Decoder的参数只用在embedding上，所以是没有意义的；传给final_layer是有意义的
    def __init__(self, num_layers, d_model, num_heads, dff, enc_units, vocab_size, rate=0.1, adapter_size=None):
        super(Transformer, self).__init__()

        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.num_heads = num_heads

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate, adapter_size)  # adapter
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, rate, adapter_size)  # adapter

        self.final_layer = tf.keras.layers.Dense(vocab_size)

        self.r_net = R_Net(enc_units)  # 创建R-Net

    def call(self, x_inp, y_tar, y_inp, a_inp, extended_inp, extended_ann_inp, max_r_oov_len, max_a_oov_len, training,
             enc_padding_mask, look_ahead_mask, dec_padding_mask, look_ahead_mask_a, dec_padding_mask_a, batch_size,
             trainable=True,
             lambda_xy=0, lambda_ay=0):
        # x_inp, y_tar, y_inp, a_inp是经过embedding的表示

        if not trainable:
            self.r_net.trainable = False

        if training:
            output1, r_output1, enc_output1 = self.l_xy(x_inp, y_tar, y_inp, extended_inp, max_r_oov_len, training,
                                                        enc_padding_mask, look_ahead_mask, dec_padding_mask, batch_size)

            if lambda_xy == 0:
                dis1 = 0
            if lambda_xy == 1:
                dis1 = distance(r_output1, enc_output1)
                dis1 = lambda_xy * dis1

            output2, r_output2, enc_output2 = self.l_ay(y_tar, y_inp, a_inp, extended_ann_inp, max_a_oov_len, training,
                                                        look_ahead_mask_a, dec_padding_mask_a, batch_size)

            if lambda_ay == 0:
                dis2 = 0
            if lambda_ay == 1:
                dis2 = distance(r_output2, enc_output2)
                dis2 = lambda_ay * dis2

            return output1, output2, dis1, dis2

        else:
            embed_enc_input = x_inp
            embed_dec_input = y_tar

            enc_output = self.encoder(embed_enc_input, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
            final_output = self.decode(embed_dec_input, enc_output, training, look_ahead_mask, dec_padding_mask,
                                       extended_inp, max_r_oov_len, batch_size)

            return final_output

    def l_ay(self, y_tar, y_inp, a_inp, extended_inp, max_oov_len, training, look_ahead_mask, dec_padding_mask,
             batch_size):
        embed_dec_input = y_tar  # tar就是经过bert的词嵌入

        enc_output = self.r_net(a_inp)
        r_output = self.r_net(y_inp)

        final_output = self.decode(embed_dec_input, enc_output, training, look_ahead_mask, dec_padding_mask,
                                   extended_inp, max_oov_len, batch_size)

        return final_output, r_output, enc_output

    def l_xy(self, x_inp, y_tar, y_inp, extended_inp, max_oov_len, training, enc_padding_mask, look_ahead_mask,
             dec_padding_mask, batch_size):
        embed_enc_input = x_inp  # inp就是经过bert的词嵌入
        embed_dec_input = y_tar  # tar就是经过bert的词嵌入

        enc_output = self.encoder(embed_enc_input, training, enc_padding_mask)
        r_output = self.r_net(y_inp)  # R(y_inp)

        final_output = self.decode(embed_dec_input, enc_output, training, look_ahead_mask, dec_padding_mask,
                                   extended_inp, max_oov_len, batch_size)

        return final_output, r_output, enc_output

    def decode(self, embed_dec_input, enc_output, training, look_ahead_mask, dec_padding_mask, extended_inp,
               max_r_oov_len, batch_size):
        # ----- decoder -----
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights, p_gens = self.decoder(embed_dec_input, enc_output, training, look_ahead_mask,
                                                             dec_padding_mask)

        output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, vocab_size)
        output = tf.nn.softmax(output)  # (batch_size, tar_seq_len, vocab_size)

        attn_dists = attention_weights[
            'decoder_layer{}_block2'.format(self.num_layers)]  # (batch_size,num_heads, tar_seq_len, inp_seq_len)
        attn_dists = tf.reduce_sum(attn_dists, axis=1) / self.num_heads  # (batch_size, tar _seq_len, inp_seq_len)

        final_dists = _calc_final_dist(vocab_dists=tf.unstack(output, axis=1),
                                       attn_dists=tf.unstack(attn_dists, axis=1),
                                       p_gens=tf.unstack(p_gens, axis=1),
                                       enc_batch_extend_vocab=extended_inp,
                                       batch_oov_len=max_r_oov_len,
                                       vocab_size=self.vocab_size,
                                       batch_size=batch_size)
        final_output = tf.stack(final_dists, axis=1)

        return final_output
