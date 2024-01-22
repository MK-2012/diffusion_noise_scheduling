from os import listdir, path
import numpy as np
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
