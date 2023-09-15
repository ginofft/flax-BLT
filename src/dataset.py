import json

DEFAULT_RESOLUTION_WIDTH = 32
DEFAULT_RESOLUTION_HEIGHT = 32

class LayoutDataset:
    
    def __init__(self, 
                name,
                path,
                add_bos=True,
                shuffle=False,
                resolution_w= DEFAULT_RESOLUTION_WIDTH,
                resolution_h = DEFAULT_RESOLUTION_HEIGHT,
                limit =22):
        """
        Args
        ---
        """
        with open(path, "r") as f:
            data = json.load(f)
        self.add_bos = add_bos
        self.name = name
        self.shuffle = shuffle
        