import jax
import jax.numpy as jnp
import flax
from flax import linen

from typing import Callable, Iterable, Optional
from . import simplified_bert

InitializerType = Callable[[jnp.ndarray, Iterable[int], jnp.dtype], jnp.ndarray]
LAYERNORM_EPSILON = 1e-12

def truncated_normal(stddev, dtype=jnp.float32):
  def init(key, shape, dtype = dtype):
    return jax.random.truncated_normal(
        key=key, lower=-2, upper=2, shape=shape, dtype=dtype) * stddev
  return init

class LayoutEmbed(linen.Module):
    use_vertical: bool
    embedding_size: int
    hidden_dropout_prob: float
    vocab_size: int
    max_position_embeddings: int
    initializer_fn: InitializerType
    layout_dim: int
    hidden_size: Optional[int] = None

    def setup(self):
        # Token embeddings.
        self.word_embedder = linen.Embed(
            num_embeddings=self.vocab_size,
            features=self.embedding_size,
            embedding_init=self.initializer_fn,
            name='word_embeddings')
        # Position embeddings.
        self.position_embedder = linen.Embed(
            num_embeddings=self.max_position_embeddings,
            features=self.embedding_size,
            embedding_init=self.initializer_fn,
            name='position_embeddings')
        # How many assets in the layout sample.
        self.asset_num_embdder = linen.Embed(
            num_embeddings=50,
            features=self.embedding_size,
            embedding_init=self.initializer_fn,
            name='asset_num_embeddings')
        # Asset segment embeddings.
        self.asset_embedder = linen.Embed(
            num_embeddings=50,
            features=self.embedding_size,
            embedding_init=self.initializer_fn,
            name='asset_embeddings')
        if self.use_vertical:
        # Vertical info embeddings.
            self.label_embedder = linen.Embed(
                num_embeddings=32,
                features=self.embedding_size,
                embedding_init=self.initializer_fn,
                name='label_embedding')
    
    @linen.compact
    def __call__(self, input_ids, labels, deterministic):
        seq_length = input_ids.shape[-1]
        position_ids = jnp.arange(seq_length)[None, :]
        asset_ids = position_ids // (self.layout_dim * 2 + 1)
        asset_num = jnp.expand_dims(
            jnp.sum(input_ids != 0, axis=1) // (self.layout_dim * 2 + 1), 1)

        word_embeddings = self.word_embedder(input_ids)
        # position_embeddings = self.position_embedder(position_ids)
        asset_embeddings = self.asset_embedder(asset_ids)
        asset_num_embeddings = self.asset_num_embdder(asset_num)
        input_embeddings = word_embeddings + asset_embeddings + asset_num_embeddings
        if labels is not None and self.use_vertical:
            labels = labels.astype('int32')
            label_emb = self.label_embedder(labels)
            input_embeddings += label_emb

        input_embeddings = linen.LayerNorm(
            epsilon=LAYERNORM_EPSILON,
            name='embeddings_ln')(input_embeddings)
        if self.hidden_size:
            input_embeddings = linen.Dense(
                features=self.hidden_size,
                kernel_init=self.initializer_fn,
                name='embedding_hidden_mapping')(
                    input_embeddings)
        input_embeddings = linen.Dropout(rate=self.hidden_dropout_prob)(
            input_embeddings, deterministic=deterministic)

        return input_embeddings


class BLT(linen.Module):
    use_vertical: bool
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
    layout_dim: int = 2

    def setup(self):
        self.embedder = LayoutEmbed(
            use_vertical=self.use_vertical,
            embedding_size=self.hidden_size,
            hidden_dropout_prob=self.hidden_dropout_prob,
            vocab_size=self.vocab_size,
            max_position_embeddings=self.max_position_embeddings,
            initializer_fn=simplified_bert.truncated_normal(
                self.initializer_range),
            layout_dim=self.layout_dim)
    
    @linen.compact
    def __call__(self, input_ids, labels, deterministic = True):
        input_ids = input_ids.astype('int32')
        input_mask = jnp.asarray(input_ids != 0, dtype=jnp.int32)
        input_embeddings = self.embedder(
            input_ids=input_ids, labels=labels, deterministic=deterministic)
        
        layer_input = input_embeddings
        for _ in range(self.num_hidden_layers):
            layer_output = simplified_bert.BertLayer(
                intermediate_size=self.intermediate_size,
                hidden_size=self.hidden_size,
                hidden_dropout_prob=self.hidden_dropout_prob,
                num_attention_heads=self.num_attention_heads,
                attention_dropout_prob=self.attention_dropout_prob,
                initializer_fn=simplified_bert.truncated_normal(
                    self.initializer_range))(
                        layer_input=layer_input,
                        input_mask=input_mask,
                        deterministic=deterministic)
            layer_input = layer_output
        word_embedding_matrix = self.variables['params']['embedder']['word_embeddings']['embedding']
        logits = simplified_bert.BertMlmLayer(
            hidden_size=self.hidden_size,
            initializer_fn=simplified_bert.truncated_normal(self.initializer_range))(
                last_layer=layer_output, embeddings=word_embedding_matrix)
        return logits
