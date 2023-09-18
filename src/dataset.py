import json
import numpy as np
from typing import Optional, Any, List, Dict
from dataclasses import dataclass
import torch

DEFAULT_RESOLUTION_WIDTH = 32
DEFAULT_RESOLUTION_HEIGHT = 32

class LayoutDataset:
    def __init__(self,
                 name,
                 path,
                 config: Optional[Any] = None,
                 add_bos = False,
                 resolution_w = DEFAULT_RESOLUTION_WIDTH,
                 resolution_h = DEFAULT_RESOLUTION_HEIGHT,
                 limit = 22):
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
        self.seq_len = self.limit * 5
        self._setup_vocab()
        self.data = self._convert_data_to_model_format(data)
        
    def _process_config(self, config):
        self.FRAME_WIDTH = config.FRAME_WIDTH
        self.FRAME_HEIGHT = config.FRAME_HEIGHT
        self.COLORS = config.COLORS
        self.LABEL_NAMES = config.LABEL_NAMES
        self.number_classes = config.NUMBER_LABELS
        self.ID_TO_LABEL = config.ID_TO_LABEL
        self.LABEL_TO_ID = config.LABEL_TO_ID_

    def _setup_vocab(self):
        self.pad_idx, self.bos_idx, self.eos_idx, self.unk_idx = 0, 1, 2, 3
        self.offset_class = 4
        self.offset_center_x = self.offset_class + self.number_classes
        self.offset_center_y = self.offset_center_x + self.resolution_w

        self.offset_width = self.offset_center_y + self.resolution_h
        self.offset_height = self.offset_width + self.resolution_w

    def get_vocab_size(self):
        return self.offset_class + self.number_classes + (self.resolution_w + self.resolution_h)*2
    
    def _convert_data_to_model_format(self, data):
        processed_entry = []
        for entries in data: 
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
                    np.clip(width * (self.resolution_w - 1), 1., self.resolution_w-1)) + self.offset_width
                discrete_height = round(
                    np.clip(height * (self.resolution_h - 1), 1., self.resolution_h-1)) + self.offset_height
                elements.extend([class_id, discrete_width, discrete_height, discrete_x, discrete_y])
            if self.add_bos:
                elements = [self.bos_idx] + elements + [self.eos_idx]
            processed_entry.append(elements)
        return processed_entry
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
@dataclass
class SmartCollator():
    """ This class provide methods for dynamic padding

    Attributes
    ----------
    pad_token_id : int
        the token id of [PAD] token, which are usually set by the tokenizer
    
    Methods
    -------
    collate_dynamic_paddding(batch)
        take in a batch of tokenized sentences with various length. Then pad them all to the longest sentence in the pad
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