import kagglehub
import splitfolders


kagglehub.dataset_download("ryanbadai/clothes-dataset")
splitfolders.ratio("/kaggle/input/clothes-dataset/Clothes_Dataset", output="/content/clothes", ratio=(0.6, 0.2, 0.2), group_prefix=None)