import json
import numpy as np
from typing import Optional, Any, List, Dict
from dataclasses import dataclass
import torch

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
        discrete width resolution
    resolution_h : int
        discrte height resolution
    data : list[list[int]]
        each element is a layout, each layout is a set of token describing the layout
    ..._idx : int
        special token
    offset_class : int
        beginning token of classes
    offset_center_x : int
        begining token of the center coordinate - in x (horizontal) dimension
    offset_center_y : int
        begining token of the center coordinate - in y (vertical) dimension
    offset_width : int
        begining token of the layout's element width
    offset_height : int
        begining token of the layout's element height 
    name : str
        dataset name
    add_bos : boolean
        whether or not to add [eos] and [bos] token    
    """
    def __init__(self,
                 name,
                 path,
                 config: Optional[Any] = None,
                 add_bos = False,
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
        add_bos : Boolean
            Whether or not to include [bos] and [eos] token
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
        self.add_bos = add_bos
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
        self.FRAME_WIDTH = config.FRAME_WIDTH
        self.FRAME_HEIGHT = config.FRAME_HEIGHT
        self.COLORS = config.COLORS
        self.LABEL_NAMES = config.LABEL_NAMES
        self.number_classes = config.NUMBER_LABELS
        self.ID_TO_LABEL = config.ID_TO_LABEL
        self.LABEL_TO_ID = config.LABEL_TO_ID_

    def _setup_vocab(self):
        """Setup vocabularies, consists of: special tokens, class tokens, center position tokens, width and height tokens
        """
        self.pad_idx, self.bos_idx, self.eos_idx, self.mask_idx = 0, 1, 2, 3
        self.offset_class = 4
        self.offset_center_x = self.offset_class + self.number_classes
        self.offset_center_y = self.offset_center_x + self.resolution_w

        self.offset_width = self.offset_center_y + self.resolution_h
        self.offset_height = self.offset_width + self.resolution_w

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
            elements = []
            for box in entries['elements'][:self.limit]:
                category_id = self.LABEL_TO_ID[box["class"]]
                center = box["center"]
                width = box["width"]
                height = box["height"]

                class_id = category_id + self.offset_class
                discrete_x = round(center[0] * self.resolution_w - 1) + self.offset_center_x
                discrete_y = round(center[1] * self.resolution_h - 1) + self.offset_center_y
                discrete_width = round(
                    np.clip(width * (self.resolution_w - 1), 0., self.resolution_w-1)) + self.offset_width
                discrete_height = round(
                    np.clip(height * (self.resolution_h - 1), 0., self.resolution_h-1)) + self.offset_height
                elements.extend([class_id, discrete_width, discrete_height, discrete_x, discrete_y])
            if self.add_bos:
                elements = [self.bos_idx] + elements + [self.eos_idx]
            processed_entry.append(elements)
        return keys, processed_entry
    
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
            batch_input += [self.pad_seq(seq, self.seq_len, self.pad_token_id)]
            
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