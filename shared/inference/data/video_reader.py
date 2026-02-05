import os
from os import path

from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torch.nn.functional as Ff
from PIL import Image
import numpy as np

from shared.dataset.range_transform import im_normalization, im_rgb2lab_normalization, ToTensor, RGB2Lab

class VideoReader_221128_TransColorization(Dataset):
    """
    This class is used to read a video, one frame at a time
    """
    def __init__(self, vid_name, image_dir, mask_dir, size=-1, to_save=None, use_all_mask=False, size_dir=None, args=None, exemplar_path=None):
        """
        image_dir - points to a directory of jpg images
        mask_dir - points to a directory of png masks
        size - resize min. side to size. Does nothing if <0.
        to_save - optionally contains a list of file names without extensions 
            where the segmentation mask is required
        use_all_mask - when true, read all available mask in mask_dir.
            Default false. Set to true for YouTubeVOS validation.
        """
        self.vid_name = vid_name
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.to_save = to_save
        self.use_all_mask = use_all_mask
        # print('use_all_mask', use_all_mask);assert 1==0
        if size_dir is None:
            self.size_dir = self.image_dir
        else:
            self.size_dir = size_dir

        flag_reverse = getattr(args, 'reverse', False) if args is not None else False
        self.frames = [img for img in sorted(os.listdir(self.image_dir), reverse=flag_reverse) if (img.endswith('.jpg') or img.endswith('.png')) and not img.startswith('.')]
        self.mask_frames = [
            msk for msk in sorted(os.listdir(self.mask_dir))
            if not msk.startswith('.') and (msk.endswith('.png') or msk.endswith('.jpg') or msk.endswith('.jpeg'))
        ]
        if self.mask_frames:
            self.first_gt_path = path.join(self.mask_dir, self.mask_frames[0])
            self.palette = Image.open(self.first_gt_path).getpalette()
            self.suffix = self.first_gt_path.split('.')[-1]
        else:
            self.first_gt_path = None
            self.palette = None
            self.suffix = None

        self.exemplar_path = exemplar_path

        if size < 0:
            self.im_transform = transforms.Compose([
                RGB2Lab(),
                ToTensor(),
                im_rgb2lab_normalization,
            ])
        else:
            self.im_transform = transforms.Compose([
                transforms.ToTensor(),
                im_normalization,
                transforms.Resize(size, interpolation=InterpolationMode.BILINEAR),
            ])
        self.size = size


    def __getitem__(self, idx):
        frame = self.frames[idx]
        info = {}
        data = {}
        info['frame'] = frame
        info['vid_name'] = self.vid_name
        info['save'] = (self.to_save is None) or (frame[:-4] in self.to_save)

        im_path = path.join(self.image_dir, frame)
        img = Image.open(im_path).convert('RGB')

        if self.image_dir == self.size_dir:
            shape = np.array(img).shape[:2]
        else:
            size_path = path.join(self.size_dir, frame)
            size_im = Image.open(size_path).convert('RGB')
            shape = np.array(size_im).shape[:2]

        gt_path = path.join(self.mask_dir, self.mask_frames[idx]) if idx < len(self.mask_frames) else None 

        img = self.im_transform(img)
        img_l = img[:1,:,:]
        img_lll = img_l.repeat(3,1,1)

        if self.exemplar_path:
            load_mask = (idx == 0)
            mask_path = self.exemplar_path
        else:
            load_mask = self.use_all_mask or (gt_path == self.first_gt_path)
            mask_path = gt_path

        if load_mask and mask_path and path.exists(mask_path):
            mask = Image.open(mask_path).convert('RGB')
            
            # 用 PIL 先 resize 成和 img 尺寸一致
            mask = mask.resize((img.shape[2], img.shape[1]), Image.BILINEAR)

            mask = self.im_transform(mask)

            # keep L channel of reference image in case First frame is not exemplar
            # mask_ab = mask[1:3,:,:]
            # data['mask'] = mask_ab
            data['mask'] = mask

        info['shape'] = shape
        info['need_resize'] = not (self.size < 0)
        data['rgb'] = img_lll
        data['info'] = info

        return data

    def resize_mask(self, mask):
        # mask transform is applied AFTER mapper, so we need to post-process it in eval.py
        h, w = mask.shape[-2:]
        min_hw = min(h, w)
        return Ff.interpolate(mask, (int(h/min_hw*self.size), int(w/min_hw*self.size)), 
                    mode='nearest')

    def get_palette(self):
        return self.palette

    def __len__(self):
        return len(self.frames)
