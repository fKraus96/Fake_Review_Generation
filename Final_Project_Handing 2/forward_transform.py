#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.layers import Layer,Dropout,Embedding,Dense
from keras.models import Sequential

import tensorflow as tf
from keras.layers import LayerNormalization
from keras.layers import MultiHeadAttention

class TokenAndPositionEmbedding(Layer):
    
    def __init__(self, MAX_LENGTH, vocab_size, embed_dim,**kwargs):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim = vocab_size,output_dim = embed_dim,input_length=MAX_LENGTH)
        self.pos_emb = Embedding(input_dim=MAX_LENGTH, output_dim=embed_dim)
        self.MAX_LENGTH = MAX_LENGTH
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, x):
        MAX_LENGTH = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=MAX_LENGTH, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
        
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
            'MAX_LENGTH': self.MAX_LENGTH,
            'vocab_size': self.vocab_size,
            'embed_dim': self.embed_dim,
    
        })
        return config



class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, hidden_dim, rate=0.1,**kwargs):
        super(TransformerBlock, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(hidden_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'embed_dim':self.embed_dim,
            'num_heads':self.num_heads,
            'hidden_dim':self.hidden_dim,
            'rate':self.rate,
            'Self_attention': self.att,
            'layer': self.ffn,
            'layernorm_1': self.layernorm1,
            'layernorm_2': self.layernorm2,
            'dropout_1': self.dropout1,
            'dropout_2': self.dropout2,
    
        })
        return config

# In[ ]:




