import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer

class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder , posefile = None, transform = None, 
                    focalx = 320.0, focaly = 320.0, centerx = 320.0, centery = 240.0, skip_n = 0):
        
        files = listdir(imgfolder)
        self.rgbfiles = [(imgfolder +'/'+ ff) for ff in files if (ff.endswith('.png') or ff.endswith('.jpg'))]
        self.rgbfiles.sort()
        self.imgfolder = imgfolder

        print('Find {} image files in {}'.format(len(self.rgbfiles), imgfolder))

        if posefile is not None and posefile!="":
            poselist = np.loadtxt(posefile).astype(np.float32)
            assert(poselist.shape[1]==7) # position + quaternion
            poses = pos_quats2SEs(poselist)
            self.matrix = pose2motion(poses)
            self.motions     = SEs2ses(self.matrix).astype(np.float32)
            # self.motions = self.motions / self.pose_std
            assert(len(self.motions) == len(self.rgbfiles)) - 1
        else:
            self.motions = None

        self.transform = transform
        self.focalx = focalx
        self.focaly = focaly
        self.centerx = centerx
        self.centery = centery

        self.skip_n = skip_n

        self.N = int(len(self.rgbfiles)/self.skip_n) - 1

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        processed_idx = (self.skip_n + 1) * idx
        next_processed_idx = (self.skip_n + 1) * (idx + 1)

        if next_processed_idx >= len(self) or processed_idx >= len(self):
            raise StopIteration

        imgfile1 = self.rgbfiles[processed_idx].strip()
        imgfile2 = self.rgbfiles[next_processed_idx].strip()
        img1 = cv2.imread(imgfile1)
        img2 = cv2.imread(imgfile2)

        res = {'img1': img1, 'img2': img2 }

        h, w, _ = img1.shape
        intrinsicLayer = make_intrinsics_layer(w, h, self.focalx, self.focaly, self.centerx, self.centery)
        res['intrinsic'] = intrinsicLayer

        if self.transform:
            res = self.transform(res)

        if self.motions is None:
            return res
        else:
            res['motion'] = self.motions[processed_idx]
            return res


