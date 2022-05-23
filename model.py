from einops import rearrange

"""
Tensorflow and einops aren't friends, so einops works but operations can't be 
wrapped in a graph for accelerated inference/training. 
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential, Model

from tensorflow.keras.layers import (
    Layer, Conv1D, Dense, Dropout, LayerNormalization, Flatten
)


#############################################################################  

class Transformer(Model):
    """
    This is an implementation of the original Transformer architecture
    introduced in Vaswani et al. 
    """
    def __init__(self,
                 model_dim: int, 
                 num_layers: int, 
                 num_heads: int) -> None:
        
        super().__init__()
        
        
        self.encoder = Encoder(model_dim, num_layers, num_heads)
        self.mlp = Sequential([Dense(2)])
       
    
    def call(self, src: tf.Tensor) -> tf.Tensor:
            
        y = self.encoder(src)
        y = self.mlp(y)
            
        return y 
    
#############################################################################
    
class Encoder(Model):
    """
    This is the Transformer Encoder block, used in the original Transformer
    and the subsequent Vision Transformer (ViT) papers. 
    """
    def __init__(self,
                 model_dim: int, 
                 num_layers: int, 
                 num_heads: int) -> None:
        
        super().__init__()
        
        self._encoder_layers = [
            EncoderLayer(model_dim, num_heads)
            for _ in range(num_layers)
        ]
        

    def call(self, src: tf.Tensor) -> tf.Tensor:
        
        for encoder_layer in self._encoder_layers:
            src = encoder_layer(src)
            
        return src
            
#############################################################################

class EncoderLayer(Layer):
    """
    The Encoder in the Transformer block (Encoder) uses multiple passes of 
    this logic. It involves multi-head self-attention and a position wise 
    feed forward neural network (implemented as an MLP).
    """
    def __init__(self, model_dim, num_heads) -> None:
        
        super().__init__()
        
        self.first_layer_norm = LayerNormalization()
        self.second_layer_norm = LayerNormalization()
        
        self.first_drop = Dropout(rate=0.1)
        self.second_drop = Dropout(rate=0.1)
        
        self.pw_net = PositionWiseNet(model_dim, width_mult=4, drop_rate=0.1)
        
        self.mha = MultiHeadAttention(
            model_dim, num_heads, method="luong"
        )
        
        
    def call(self, src: tf.Tensor) -> tf.Tensor:
        
        # first step is layer normalisation, followed by mult-head 
        # attention and dropout.
        src = self.first_layer_norm(src)
        attention = self.first_drop(self.mha(src, src, src))

        # perform skip connection and update 
        res = attention + src
        src = res
        
        # the last step is to perform another layer normalisation 
        # followed by a position-wise neural net and dropout
        res = self.second_layer_norm(res)
        ff_net = self.second_drop(self.pw_net(res))
        ff_net += src
        
        return ff_net
         
 #############################################################################  
    
class PositionWiseNet(Model):
    """
    This "Feed-forward Neural Network" operates position wise on each token. 
    It operates identically at each point.
    """
    def __init__(self, 
                 model_dim: int, 
                 width_mult: int, 
                 drop_rate: float) -> None:
        
        super().__init__()
        
        self.first_layer = Dense(width_mult * model_dim, activation="relu")
        self.hidden_layer = Dense(model_dim)
        self.drop = Dropout(rate=drop_rate)
        
        
    def call(self, x: tf.Tensor) -> tf.Tensor:
        
        y = self.drop(self.first_layer(x))
        y = self.hidden_layer(y)
        
        return y
        
#############################################################################

class MultiHeadAttention(Layer):
    """
    This Layer implements Multi-headed Attention. This was first introduced
    in Vaswani et al. and uses multiple attention heads, which are 
    concatenated together. It can also calculate self-attention if specified.
    """
    def __init__(self, 
                 model_dim: int, 
                 num_heads: int, 
                 method: str) -> None:
        
        super().__init__()
            
        assert model_dim % num_heads == 0, \
        f"model_dim: {model_dim} must be divisible by num_heads: {num_heads}"
        
        self.num_heads = num_heads
        self.head_dim = int(model_dim / num_heads)
        self.attention = Attention(method=method)
        
        self._query = Dense(model_dim)
        self._key = Dense(model_dim)
        self._value = Dense(model_dim)
        self._project = Dense(model_dim)
                
    
    def call(self, 
             query: tf.Tensor, 
             key: tf.Tensor, 
             value: tf.Tensor) -> tf.Tensor:
        
        # in self-attention the query, key and value tensors are identical
        # we allow here for different types of attention. 
        # we then multiple by a learnt matrix (Dense layer)
        query = self._query(query)
        key = self._key(key)
        value = self._value(value)
        
        # separate the heads by reshaping
        query = rearrange(query, "b n (h d) -> b h n d", h=self.num_heads)
        key = rearrange(key, "b n (h d) -> b h n d", h=self.num_heads)
        value = rearrange(value, "b n (h d) -> b h n d", h=self.num_heads)
        
        attention_weights = self.attention(query, key, value)
        
        # reshape back to (b, n h*d)
        # which is equivalent to concatenating
        attention_weights = rearrange(attention_weights, 
                                      "b h n d -> b n (h d)", 
                                      h=self.num_heads)
        
        y = self._project(attention_weights)
        
        return y 
        
#############################################################################
        
class Attention(Layer):
    """
    This Layer calculates the attention weights, used in a Transformer-like
    architecture. Both the Luong (Luong et al. 2015) and Bahdanau 
    (Bahdanau et al. 2014) methods can be used. 
    """
    def __init__(self, method: str) -> None:
        
        super().__init__()
        
        _methods = ["luong","bahdanau"]
        assert method in _methods, f"method must be one of {_methods}"
        
        self._method = method
        
        
    @tf.function
    def _luong_attention(self, 
                        query: tf.Tensor,
                        key: tf.Tensor,
                        value: tf.Tensor) -> tf.Tensor:

        head_dim = tf.cast(tf.shape(key)[-1], tf.float32)
        scale = tf.math.sqrt(head_dim)
        scores = tf.matmul(query, key, transpose_b=True)
        
        attention_weights= tf.nn.softmax(scores, axis=-1)
        attention = tf.matmul(attention_weights, value)   
        
        return attention
        
    
    @tf.function
    def bahdanau_attention(self, 
                           query: tf.Tensor,
                           key: tf.Tensor,
                           value: tf.Tensor) -> tf.Tensor:
        
        raise NotImplementedError
    
    
    def call(self, 
             query: tf.Tensor, 
             key: tf.Tensor, 
             value: tf.Tensor) -> tf.Tensor:
        
        # For now just think in terms of tokens. 
        # For each token we have a query, key and value (in self-attention).
        # We also have a learnable matrix for each W_q, W_k, and W_v such that
        # token_1 * W_q gives the query vector associated with that word. 
        # this is repeated for all the tokens.
        
        # the key is calculated by multiplying the token by W_k. Given a query
        # we calculate the dot product between the query and all of the keys. 
        
        # This is then soft-maxed and multiplied by the value vector found by 
        # multiplying each token by W_v. The softmaxing downweights tokens 
        # which are not related to the current token. 

        if self._method == "badanau":
             y = self._bahdanau_attention(query, key, value)
             
        if self._method == "luong":
             y = self._luong_attention(query, key, value)
        
        return y 
    
############################################################################