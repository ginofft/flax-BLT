import jax.numpy as jnp
import jax
from flax import linen
from typing import Any, Callable, Dict, Iterable, Optional, Text, Tuple, Union

InitializerType = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]
LAYERNORM_EPSILON = 1e-12

class Bias(linen.Module):
    """Adds a (learned) bias to the input.

    Attributes:
        dtype: the dtype of the computation (default: float32).
        bias_init: initializer function for the bias.
    """
    dtype: Any = jnp.float32
    bias_init: Callable[[Any, Tuple[int], Any], Any] = linen.initializers.zeros

    @linen.compact
    def __call__(self, inputs):
        """Applies a linear transformation to the inputs along the last dimension.

        Args:
        inputs: The nd-array to be transformed.

        Returns:
        The transformed input.
        """
        inputs = jnp.asarray(inputs, self.dtype)

        bias_shape = inputs.shape[-1]
        bias = self.param('bias', self.bias_init, bias_shape)
        bias = jnp.asarray(bias, self.dtype)
        bias = jnp.broadcast_to(bias, inputs.shape)

        return inputs + bias

class BertEmbed(linen.Module):
    """Embeds Bert-style."""
    embedding_size: int
    hidden_dropout_prob: float
    vocab_size: int
    max_position_embeddings: int
    initializer_fn: InitializerType
    hidden_size: Optional[int] = None
    
    @linen.compact
    def __call__(self, input_ids, deterministic = True):
        seq_length = input_ids.shape[-1]
        position_ids = jnp.arange(seq_length)[None, :]

        word_embedder = linen.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_size,
            embedding_init=self.initializer_fn,
            name='word_embeddings'
        )
        word_embeddings = word_embedder(input_ids)

        position_embeddings = linen.Embed(
            num_embeddings=self.max_position_embeddings,
            features=self.embedding_size,
            embedding_init=self.initializer_fn,
            name='position_embeddings'
        )(position_ids)

        input_embeddings = linen.LayerNorm(
            epsilon=LAYERNORM_EPSILON,
            name='embeddings_ln'
        )(word_embeddings + position_embeddings)

        if self.hidden_size:
            input_embeddings = linen.Dense(
                features=self.hidden_size,
                kernel_init=self.initializer_fn,
                name='embedding_hidden_mapping'
            )(input_embeddings)
        input_embeddings = linen.Dropout(rate=self.hidden_dropout_prob)(
            input_embeddings, deterministic=deterministic)
        
        return input_embeddings

class BertMlmLayer(linen.Module):
   hidden_size: int
   initializer_fn: InitializerType
   @linen.compact
   def __call__(self, last_layer, embeddings):
      mlm_hidden = linen.Dense(
         features=self.hidden_size,
         kernel_init=self.initializer_fn,
         name='mlm_dense'
      )(last_layer)
      mlm_hidden = jax.nn.gelu(mlm_hidden)
      mlm_hidden = linen.LayerNorm(
         epsilon=LAYERNORM_EPSILON,
         name='mlm_ln'
      )(mlm_hidden)
      output_weights = jnp.transpose(embeddings)
      logits = jnp.matmul(mlm_hidden, output_weights)
      logits  = Bias(name='mlm_bias')(logits)
      return logits
def truncated_normal(stddev, dtype=jnp.float32):
  def init(key, shape, dtype = dtype):
    return jax.random.truncated_normal(
        key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev
  return init

class BertAttention(linen.Module):
    """BERT attention layer - part of each BERT Layer"""
    hidden_size: int
    hidden_dropout_prob: float
    num_attention_heads: int
    attention_dropout_prob: float
    initializer_fn: InitializerType

    @linen.compact
    def __call__(self, layer_input, input_mask, deterministic):
        attention_mask = linen.make_attention_mask(input_mask, input_mask)
        attention_output = linen.attention.SelfAttention(
            num_heads=self.num_attention_heads,
            qkv_features=self.hidden_size,
            dropout_rate=self.attention_dropout_prob,
            deterministic=deterministic,
            kernel_init=self.initializer_fn,
            bias_init=jax.nn.initializers.zeros,
            name='self_attention'
        )(layer_input, attention_mask)

        attention_output = linen.Dropout(rate=self.hidden_dropout_prob)(
            attention_output, deterministic=deterministic)
        attention_output = linen.LayerNorm(
            epsilon=LAYERNORM_EPSILON,
            name='attention_output_ln'
        )(attention_output + layer_input)
        return attention_output

class BertMlp(linen.Module):
    hidden_size: int
    hidden_dropout_prob: float
    intermediate_size: int
    initializer_fn: InitializerType

    @linen.compact
    def __call__(self, attention_output, deterministic):
        intermediate_output = linen.Dense(
            features=self.intermediate_size,
            kernel_init=self.initializer_fn,
            name='intermediate_output'
        )(attention_output)
        intermediate_output = jax.nn.gelu(intermediate_output)

        layer_output = linen.Dense(
            features=self.hidden_size,
            kernel_init=self.initializer_fn,
            name='layer_output'
        )(intermediate_output)
        layer_output = linen.Dropout(rate=self.hidden_dropout_prob)(
            layer_output, deterministic=deterministic)
        layer_output = linen.LayerNorm(
            epsilon=LAYERNORM_EPSILON,
            name='layer_output_ln')(
                layer_output+attention_output)
        
        return layer_output

class BertLayer(linen.Module):
    intermediate_size: int
    hidden_size: int
    hidden_dropout_prob: float
    num_attention_heads: int
    attention_dropout_prob:float
    initializer_fn: InitializerType

    @linen.compact
    def __call__(self, layer_input, input_mask, deterministic):
        attention_output = BertAttention(
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            num_attention_heads=self.num_attention_heads,
            attention_dropout_prob=self.attention_dropout_prob,
            initializer_fn=self.initializer_fn)(
                layer_input = layer_input,
                input_mask = input_mask,
                deterministic = deterministic)
        
        layer_output = BertMlp(
            hidden_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            intermediate_size=self.intermediate_size,
            initializer_fn=self.initializer_fn
        )(attention_output=attention_output, deterministic=deterministic)
        
        return layer_output
    
class Bert(linen.Module):
    """BERT as a Flax module."""
    vocab_size: int
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    pad_token_id: int = -1

    @linen.compact
    def __call__(self,
                input_ids,
                deterministic = True):
        # We assume that all pad tokens should be masked out.
        input_ids = input_ids.astype('int32')
        input_mask = jnp.asarray(input_ids != self.pad_token_id, dtype=jnp.int32)

        input_embeddings = BertEmbed(
            embedding_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            initializer_fn=truncated_normal(self.initializer_range))(
                input_ids=input_ids,
                deterministic=deterministic)

        # Stack BERT layers.
        layer_input = input_embeddings
        for _ in range(self.num_hidden_layers):
            layer_output = BertLayer(
                intermediate_size=self.intermediate_size,
                hidden_size=self.hidden_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                num_attention_heads=self.num_attention_heads,
                attention_dropout_prob=self.attention_dropout_prob,
                initializer_fn=truncated_normal(self.initializer_range))(
                    layer_input=layer_input,
                    input_mask=input_mask,
                    deterministic=deterministic)
            layer_input = layer_output

        word_embedding_matrix = self.variables['params']['BertEmbed_0'][
            'word_embeddings']['embedding']
        logits = BertMlmLayer(
            hidden_size=self.hidden_size,
            initializer_fn=truncated_normal(self.initializer_range))(
                last_layer=layer_output, embeddings=word_embedding_matrix)

        return logits
