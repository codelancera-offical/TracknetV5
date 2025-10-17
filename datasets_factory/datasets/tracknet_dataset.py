import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from ..builder import DATASETS, build_pipeline

@DATASETS.register_module
class TrackNetDataset(Dataset):
    def __init__(self, csv_path: str, pipeline: list, data_dir: str = '', 
                 input_height: int = 360, input_width: int = 640):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.data_info = pd.read_csv(csv_path)
        self.pipeline = build_pipeline(pipeline)
        self.input_height = input_height
        self.input_width = input_width
        print(f"Dataset '{self.__class__.__name__}' initialized.")
        print(f"Loaded {Path(csv_path).name}, total samples = {len(self.data_info)}")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx: int) -> dict:
        row_info = self.data_info.iloc[idx].to_dict()

        results = {
            'data_dir': self.data_dir,
            'input_height': self.input_height,
            'input_width': self.input_width,
            'original_info': row_info,
            **row_info
        }
        
        # ✨✨✨ 关键修正：在这里手动创建 'coords' 键 ✨✨✨
        # 我们从原始的列中，组合出后续模块需要的 'coords' 键
        if 'x-coordinate' in row_info and 'y-coordinate' in row_info:
            results['coords'] = (row_info['x-coordinate'], row_info['y-coordinate'])
        
        img_fields = [k for k in row_info.keys() if 'path' in k and k != 'gt_path']
        results['img_fields'] = img_fields
        
        for key in img_fields + ['gt_path']:
             if key in results:
                results[key] = self.data_dir / results[key]

        return self.pipeline(results)