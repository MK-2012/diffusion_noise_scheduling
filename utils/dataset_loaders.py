from os import listdir, path
import numpy as np
from torch import from_numpy, float32, zeros
from torch.utils.data import Dataset
from cv2 import VideoCapture, CAP_PROP_FRAME_COUNT, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT


class UCFDataset(Dataset):
    __slots__ = "root_dir", "class_list", "items", "item_inds"

    def __init__(self, ucf_dataset_directory):
        self.root_dir = ucf_dataset_directory
        self.class_list = np.array(listdir(ucf_dataset_directory))
        
        self.items = np.array(listdir(path.join(ucf_dataset_directory, self.class_list[0])))
        self.items = np.vstack([self.items, [self.class_list[0]] * self.items.shape[0]]).T
        self.item_inds = np.array([0.0] * self.items.shape[0])
        for i in range(1, self.class_list.shape[0]):
            temp = np.array(listdir(path.join(ucf_dataset_directory, self.class_list[i])))
            temp = np.vstack([temp, [self.class_list[i]] * temp.shape[0]]).T
            self.item_inds = np.hstack([self.item_inds, [float(i)] * temp.shape[0]])
            self.items = np.vstack([self.items, temp])

    def __len__(self):
        return self.items.shape[0]
    
    def _capture_video_(self, path):
        cap = VideoCapture(path)
        frameCount = int(cap.get(CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype('uint8'))

        fc = 0
        ret = True

        while (fc < frameCount and ret):
            ret, buf[fc] = cap.read()
            fc += 1

        cap.release()
        
        return np.transpose(buf, (3, 0, 1, 2))

    def __getitem__(self, idx):
        return self._capture_video_(path.join(path.join(self.root_dir,
                                                        self.items[idx, 1]),
                                              self.items[idx, 0])),\
               self.item_inds[idx]


class MovMNISTDataset(Dataset):
    __slots__ = "data"

    def __init__(self, mov_mnist_numpy_path):
        self.data = from_numpy(np.load(mov_mnist_numpy_path)).permute(1, 0, 2, 3).unsqueeze(1)
        self.data = self.data.float() / 255 * 2 - 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        try:
            a = len(idx)
        except:
            a = 1
        return self.data[idx], zeros(a, dtype=float32)


class MovMNISTFrameDataset(Dataset):
    __slots__ = "data"

    def __init__(self, mov_mnist_numpy_path):
        self.data = from_numpy(np.load(mov_mnist_numpy_path))
        self.data = self.data.view(-1, *self.data.shape[-2:]).unsqueeze(1)
        self.data = self.data.float() / 255 * 2 - 1

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        try:
            a = len(idx)
        except:
            a = 1
        return self.data[idx], zeros(a, dtype=float32)