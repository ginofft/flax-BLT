# Biodirection Layout Transformer - FLAX Implementation
This repository contains an updated and fixed version of the original **[implementation](https://github.com/google-research/google-research/tree/master/layout-blt)**.

If you find this code useful please cite the original research paper.
```
@article{kong2021blt,
  title={BLT: Bidirectional Layout Transformer for Controllable Layout Generation},
  author={Kong, Xiang and Jiang, Lu and Chang, Huiwen and Zhang, Han and Hao, Yuan and Gong, Haifeng and Essa, Irfan},
  journal={arXiv preprint arXiv:2112.05112},
  year={2021}
}
```
I'm planning to keep this updated and implement a Pytorch version, PRs are alway welcomed.

# Changes in this implementation
- Make use of newer implementation of Jax, Jaxlib and Flax
- Make use of orbax_checkpoint and optax
- Fix attribute_random_masking() function.
- Remove some uncessary bits.

# Quick start
Start by cloning and installing the necessary libraries from `requirements.txt`.
As the training is quite slow on CPU. It is highly recommended to setup a different environment in a GPU-supported device.

Here is my workflow on Google Colab
```
!git clone https://github.com/ginofft/flax-BLT.git
!pip install --upgrade jax==0.4.16 jaxlib==0.4.16+cuda11.cudnn86 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
!pip install clu flax orbax-checkpoint optax frozendict livelossplot
```

## Training
I haven't add a `main.py` yet, so the current training loop can be perform in a notebook instance.
```
from src.trainer import BERTLayoutTrainer
from src.utils import attribute_random_masking, get_publaynet_config
import jax.numpy as jnp

config = get_publaynet_config()
config.checkpoint_path = "your/check/point/path"
dataset_config = config.dataset
trainer = BERTLayoutTrainer(config=config, workdir='where/to/store')
```

```
trainer.train()
```

Every para/hyperpara can be changed through the `config` variable. It is recommended to take a look inside `src/utils.py` and customize your training workflow as you see fit.

## Decoding
I haven't add a `main.py` yet, so the current decoding can be perform in a notebook instance.

```
from src.trainer import BERTLayoutTrainer
from src.dataset import LayoutDataset, PaddingCollator
from src.utils import attribute_random_masking, get_publaynet_config
from torch.utils.data import DataLoader
from pathlib import Path
import orbax
import jax.numpy as jnp

config = get_publaynet_config()
dataset_config = config.dataset
trainer = BERTLayoutTrainer(config=config, workdir=None) #workdir is None, as we do not plan to train and save any model
```

To initiate model architecture
```
train_dataset = LayoutDataset('train',
                              config.dataset_path+'/train.json',
                              config=dataset_config)
val_dataset = LayoutDataset('validation',
                            config.dataset_path+'/val.json',
                            config=dataset_config)
collator = PaddingCollator(pad_token_id=train_dataset.pad_idx, seq_len=train_dataset.seq_len)
train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size,
                                collate_fn=collator.collate_padding,
                                shuffle=config.train_shuffle)
val_dataloader = DataLoader(val_dataset, batch_size=config.eval_batch_size,
                            collate_fn=collator.collate_padding,
                            shuffle=config.train_shuffle)

init_batch = jnp.ones((trainer.config.batch_size, train_dataset.seq_len))
init_label = jnp.ones((trainer.config.batch_size, train_dataset.seq_len))
init_batch = dict(inputs=init_batch, labels=init_label)
vocab_size = train_dataset.get_vocab_size()

state = trainer.create_train_state(rng = trainer.rng, inputs = init_batch, vocab_size=vocab_size)
```
To load a trained model
```
target = {'model':state, 'metric_history':dict, 'min_loss': float, 'epoch':int}

orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
checkpoint_path = Path('output/publaynet/best')
if checkpoint_path.exists() == False:
    raise Exception(f"{checkpoint_path} does not exits!!")
ckpt = orbax_checkpointer.restore(checkpoint_path, item=target)

state = ckpt['model']
metric_history = ckpt['metric_history']
min_validation_loss = ckpt['min_loss']
start_epoch = ckpt['epoch']
print("Loaded epoch {} - Loss: {:.6f}".format(start_epoch, min_validation_loss))
```
Take a random batch from validation set
```
vocab_size = train_dataset.get_vocab_size()

# Position of key tokens
pos_info = [[train_dataset.offset_class, train_dataset.number_classes], 
            [train_dataset.offset_width, train_dataset.resolution_w],
            [train_dataset.offset_height, train_dataset.resolution_h],
            [train_dataset.offset_center_x, train_dataset.resolution_w],
            [train_dataset.offset_center_y, train_dataset.resolution_h]]

seq_len = train_dataset.seq_len
possible_logit, offset = trainer._make_possible_mask(vocab_size=vocab_size, 
                                                     pos_info=pos_info,
                                                     seq_len=seq_len)
for data in val_dataloader:
    batch = attribute_random_masking(data, mask_token=train_dataset.mask_idx,
                                    pad_token=train_dataset.pad_idx, layout_dim=trainer.layout_dim)
    break
```
Call your decoder
```
from src.decoder import LayoutDecoder

vocab_size = train_dataset.get_vocab_size()
seq_len = train_dataset.seq_len
layout_dim = config.layout_dim
color_map = train_dataset.COLORS
id_to_label = train_dataset.ID_TO_LABEL
resolution_w = train_dataset.resolution_w
resolution_h = train_dataset.resolution_h

decoder = LayoutDecoder(vocab_size, seq_len, layout_dim, 
                        id_to_label, color_map, 
                        resolution_w, resolution_h, temperature=1.0)
decoder.model = state
seq = decoder.decode(state, batch["masked_inputs"], possible_logit)
```
Rendering
```
index = 16
true_layout = batch["targets"][index]
generated = seq[index][-1]
mask = batch["weights"][index]
decoder.render_two_layouts(true_layout, generated, mask, offset)
```