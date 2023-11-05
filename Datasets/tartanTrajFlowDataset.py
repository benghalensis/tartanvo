import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import join
from .transformation import pos_quats2SEs, pose2motion, SEs2ses
from .utils import make_intrinsics_layer

def flow16to32(flow16):
    '''
    flow_32b (float32) [-512.0, 511.984375]
    flow_16b (uint16) [0 - 65535]
    flow_32b = (flow16 -32768) / 64
    '''
    flow32 = flow16[:,:,:2].astype(np.float32)
    flow32 = (flow32 - 32768) / 64.0

    mask8 = flow16[:,:,2].astype(np.uint8)
    return flow32, mask8

class TrajFolderDataset(Dataset):
    """scene flow synthetic dataset. """

    def __init__(self, imgfolder, flow_folder = None, posefile = None, transform = None, 
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

        if flow_folder is not None:
            files = listdir(flow_folder)
            self.flowfiles = [join(flow_folder, ff) for ff in files if ff.endswith('.png')]
            self.flowfiles.sort()
            assert(len(self.flowfiles) - 1 == len(self.rgbfiles)/2 - 1)
        else:
            self.flowfiles = None

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

        if self.flowfiles is not None:
            flowfile = self.flowfiles[idx+1]
            flow16 = cv2.imread(flowfile, cv2.IMREAD_UNCHANGED)
            res['flow'] = flow16to32(flow16)[0]

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


