import jax
from jax import jit
import jax.numpy as jnp
import flax
from flax.training import train_state
import optax
from clu import metrics

from functools import partial

from .models.biodirectional_layout import BLT
from .utils import attribute_random_masking

@flax.struct.dataclass
class Metrics(metrics.Collection):
    loss: metrics.Average.from_output("loss")

@flax.struct.dataclass
class TrainState(train_state.TrainState):
    metrics: Metrics

class BERTLayoutTrainer:
    """BERT-style Layout Trainer"""

    def __init__(self, config, workdir):
        self.config = config
        self.workdir = workdir
        self.rng = jax.random.PRNGKey(config.seed)
        self.dtype, self.data_dtype = self._get_dtype()
        self.layout_dim = self.config.layout_dim
        self.total_dim = self.layout_dim*2 + 1

    def _get_dtype(self):
        if self.config.dtype == "bfloat16":
            return jnp.bfloat16, jnp.bfloat16
        else:
            return jnp.float32, jnp.float32
    
    def preprocess_batch(self, batch, batch_size, dataset, use_vertical=False):
        label = None
        if batch.shape[0] != batch_size:
            return None, None
        batch = attribute_random_masking(
            inputs=batch, mask_token=3,
            pad_token=0, layout_dim=self.layout_dim)
        return batch, label
    
    def _create_optimizer(self):
        opt_def = optax.adamaxw(
            learning_rate=self.config.optimizer.lr,
            b1=self.config.optimizer.beta1,
            b2=self.config.optimizer.beta2,
            weight_decay=self.config.optimizer.weight_decay
        )
        return opt_def
    def create_train_state(
            self,
            rng,
            inputs
    ):
        model = BLT(use_vertical=self.config.use_vertical_info,
                    vocab_size=self.config.vocab_size,
                    hidden_size=self.config.qkv_dim,
                    num_hidden_layers=self.config.num_layers,
                    num_attention_heads=self.config.num_heads,
                    intermediate_size=self.config.mlp_dim,
                    pad_token_id=0,
                    layout_dim=self.config.layout_dim)
        param_rng, dropout_rng, rng = jax.random.split(rng, 3)
        params = model.init({
                "params": param_rng,
                "dropout": dropout_rng
            },
            input_ids=inputs["inputs"],
            labels=inputs["labels"],
            deterministic=False
        )["params"]
        tx = self._create_optimizer()
        
        return TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx,
            metrics=Metrics.empty()
        )
    
    @partial(jit, static_argnums=(0,))
    def train_step(self,
                    state,
                    input_ids,
                    labels,
                    mask):
        def loss_fn(params):
            logits = state.apply_fn({'params':params}, input_ids=input_ids, labels=labels)
            loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
            loss = loss * mask
            loss = loss.mean()
            normalizing_factor = mask.size / mask.sum()
            return loss * normalizing_factor
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state
        