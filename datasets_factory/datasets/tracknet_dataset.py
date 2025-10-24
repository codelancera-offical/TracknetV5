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
        self.data_info = pd.read_csv(csv_path) # 在这里已经把表读进去了
        self.pipeline = build_pipeline(pipeline) # 初始化pipeline
        self.input_height = input_height
        self.input_width = input_width
        print(f"Dataset '{self.__class__.__name__}' initialized.")
        print(f"Loaded {Path(csv_path).name}, total samples = {len(self.data_info)}")

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx: int) -> dict:
        row_info = self.data_info.iloc[idx].to_dict() # 这也是解出来一个字典？

        results = {
            'data_dir': self.data_dir,
            'input_height': self.input_height,
            'input_width': self.input_width,
            'original_info': row_info,
            **row_info # 这个是什么鬼？直接解字典出来？
        }

        """
        **row_info 等效于解包出：
        {
            path:
            path_prev:
            path_next:
            gt_path:
            ...
        }
        """
        
        if 'x-coordinate' in row_info and 'y-coordinate' in row_info:
            results['coords'] = (row_info['x-coordinate'], row_info['y-coordinate']) # 以元组形式追加results中的坐标信息
        
        img_fields = [k for k in row_info.keys() if 'path' in k and k != 'gt_path']
        results['img_fields'] = img_fields # 把 path_prev, path, path_next 这三个 key 抓出来（字符串形式）
        
        for key in img_fields + ['gt_path']:
             if key in results:
                results[key] = self.data_dir / results[key] # 把完整路径组装出来

        # pipeline 主要就访问  x, y, path, path_prev, path_next, gt_path了
        return self.pipeline(results) # 把组合出的路径丢入pipeline，让pipeline根据图片路径组装出输入的数据