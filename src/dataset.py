import json
import numpy as np
from typing import Optional, Any, List, Dict
from dataclasses import dataclass
import torch
from PIL import Image, ImageDraw, ImageOps
import frozendict

DEFAULT_RESOLUTION_WIDTH = 32
DEFAULT_RESOLUTION_HEIGHT = 32

class LayoutDataset:
    """Layout Dataset for training using vocabulary similar to NLP task.
    A datapoint is an array [int] where each element is the index of the corresponding vocab

    Attributes
    ----------
    limit : int
        Maximum no. layout elements.
    seq_len : int
        sequen length, currently is 5*limit -> subject to change (for 3d data)
    resolution_w : int
        discrete wth resolution
    resolution_h : int
        discrte height resolution
    data : list[list[int]]
        each element is a layout, each layout is a set of token describing the layout
    ..._idx : int
        special token
    offset_class : int
        beginning token of classes
    offset_x : int
        begining token of the x coordinate (horizontal) dimension
    offset_y : int
        begining token of the y coordinate (horizontal) dimension
    offset_width : int
        begining token of the layout's element width
    offset_height : int
        begining token of the layout's element height 
    name : str
        dataset name 
    """
    def __init__(self,
                 name,
                 path,
                 config: Optional[Any] = None,
                 resolution_w = DEFAULT_RESOLUTION_WIDTH,
                 resolution_h = DEFAULT_RESOLUTION_HEIGHT,
                 limit = 26):
        """Creating a Layout dataset

        Parameters
        ----------
        name : str
            dataset name
        path : pathlib.Path
            Path to JSON file
        config : Any
            A config with certain attributes (or field), which includes: 
            FRAME_WIDTH, FRAME_HEIGHT, COLORS, LABEL_NAMES, NUMBER_LABELS, ID_TO_LABEL, LABEL_TO_ID_
        resolution_w : int
            discrete resolution of width
        resolution_h : int 
            discrete resolution of height
        limit : int
            maximum no. elements
        """
        if config is None:
            raise Exception("Please provide a data config file!!")
        else:
            self._process_config(config)
        with open(path, "r") as f:
            data = json.load(f)
        self.name = name
        self.resolution_w = resolution_w
        self.resolution_h = resolution_h
        self.limit = limit
        self.seq_len = self.limit*5
        self._setup_vocab()
        self.key, self.data = self._convert_data_to_model_format(data)
        
    def _process_config(self, config):
        """Reading config
        """
        self.COLORS = config.COLORS
        self.LABEL_NAMES = config.LABEL_NAMES
        self.number_classes = config.NUMBER_LABELS
        self.LABEL_TO_ID = config.LABEL_TO_ID_
        self.ID_TO_LABEL = config.ID_TO_LABEL

    def _setup_vocab(self):
        """Setup vocabularies, consists of: special tokens, class tokens, position token, width and height tokens
        """
        self.pad_idx, self.bos_idx, self.eos_idx, self.mask_idx = 0, 1, 2, 3
        self.offset_class = 4
        self.offset_x = self.offset_class + self.number_classes
        self.offset_y = self.offset_x + self.resolution_w

        self.offset_width = self.offset_y + self.resolution_h
        self.offset_height = self.offset_width + self.resolution_w

        self.LABEL_TO_ID = frozendict.frozendict({
            label: index + self.offset_class for label, index in self.LABEL_TO_ID.items()
        })
        self.ID_TO_LABEL = frozendict.frozendict({
            index: label for label, index in self.LABEL_TO_ID.items()
        })

    def get_vocab_size(self):
        return self.offset_class + self.number_classes + (self.resolution_w + self.resolution_h)*2
    
    def _convert_data_to_model_format(self, data):
        """Discretize original data into vocab forms

        Parameters
        ---------
        data : List of dictionary, each containg: layout metadata and layout dimensions (scaled to [0,1])
        
        Returns
        -------
        data : list[list[int]]
            discritzed version of original data
        """
        processed_entry = []
        keys = []
        for entries in data: 
            keys.append(entries["name"])
            entries["elements"] = sorted(entries['elements'], 
                                         key=lambda x: self.LABEL_TO_ID.get(x['class'], 
                                                                            len(self.LABEL_TO_ID)))
            elements = []
            for box in entries['elements'][:self.limit]:
                class_id = self.LABEL_TO_ID[box["class"]]
                x = np.clip(box['x'],0,1)
                y = np.clip(box['y'],0,1)
                width = np.clip(box['width'],0,1)
                height = np.clip(box['height'],0,1)

                discrete_x = round(x * (self.resolution_w - 1)) + self.offset_x
                discrete_y = round(y * (self.resolution_h - 1)) + self.offset_y
                discrete_width = round(
                    np.clip(width * (self.resolution_w - 1), 1., self.resolution_w-1)) + self.offset_width
                discrete_height = round(
                    np.clip(height * (self.resolution_h - 1), 1., self.resolution_h-1)) + self.offset_height
                elements.extend([class_id, discrete_width, discrete_height, discrete_x, discrete_y])
            processed_entry.append(np.array(elements))
        return keys, processed_entry

    def render(self, layout):
        img =  Image.new('RGB', (256,256), color=(255,255,255))
        draw = ImageDraw.Draw(img, 'RGBA')
        layout = layout.reshape(-1)
        layout = trim_tokens(layout, pad_token=self.pad_idx)
        layout = layout[:len(layout)//5*5].reshape(-1, 5) # guarrantee layout length is divisible by 5
        
        box = layout[:, 1:]
        box[:, 0] = box[:, 0] - self.offset_width
        box[:, 1] = box[:, 1] - self.offset_height
        box[:, 2] = box[:, 2] - self.offset_x
        box[:, 3] = box[:, 3] - self.offset_y

        box[:, [0,2]] = box[:, [0,2]] / (self.resolution_w-1) * 256
        box[:, [1,3]] = box[:, [1,3]] / (self.resolution_h-1) * 256
        
        box[:, [0,1]] =  box[:, [0,1]] + box[:, [2,3]]
        for i in range(len(layout)):
            x2, y2, x1, y1 = box[i]
            cat = self.ID_TO_LABEL[layout[i][0]]
            col = self.COLORS[cat]
            draw.rectangle([x1, y1, x2, y2],
                           outline=tuple(col) + (200,),
                           fill=tuple(col) + (64,),
                           width=2)

        # Add border around image
        img = ImageOps.expand(img, border=2)
        return img
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
@dataclass
class PaddingCollator():
    """ This class provide methods for collate and padding. Meant to be used with Pytorch's DataLoader

    Attributes
    ----------
    pad_token_id : int
        the token id of [PAD] token
    
    Methods
    -------
    collate_padding(batch)
        take in a batch of tokenized sentences with various length. Pad them all to the longest possible length allowed
    """
    pad_token_id: int
    seq_len: int
    def pad_seq(self, seq:List[int], max_batch_len: int, pad_value:int)->List[int]:
        return seq + (max_batch_len - len(seq)) * [pad_value]

    def collate_padding(self, batch) -> torch.Tensor:
        batch_input = []
        for seq in batch:
            batch_input += [self.pad_seq(list(seq), self.seq_len, self.pad_token_id)]
            
        return np.array(batch_input, dtype=np.int32)
    
@dataclass
class SmartCollator():
    """ This class provide methods for collate and dynamic padding. Meant to be used with Pytorch's DataLoader

    Attributes
    ----------
    pad_token_id : int
        the token id of [PAD] token
    
    Methods
    -------
    collate_dynamic_padding(batch)
        take in a batch of tokenized sentences with various length. Then pad them all to the longest sentence in the batch
    """
    pad_token_id: int
    def pad_seq(self, seq:List[int], max_batch_len: int, pad_value:int)->List[int]:
        return seq + (max_batch_len - len(seq)) * [pad_value]

    def collate_dynamic_padding(self, batch) -> torch.Tensor:
        batch_input = []
        max_size = max([len(ex) for ex in batch])
        for seq in batch:
            batch_input += [self.pad_seq(seq, max_size, self.pad_token_id)]
            
        return np.array(batch_input, dtype=np.int32)

# Utils function
def trim_tokens(layout, pad_token=None):
    if pad_token is not None:
        layout = layout[layout!=pad_token]
    return layout
