import os
from os import path
import json

from shared.inference.data.video_reader import VideoReader_221128_TransColorization

class DAVISTestDataset_221128_TransColorization_batch:
    def __init__(self, data_root, imset='2017/val.txt', size=-1, args=None, subset=None, exemplar_path=None):
        self.image_dir = data_root
        self.mask_dir = imset
        self.size_dir = data_root
        self.size = size
        self.exemplar_path = exemplar_path

        self.vid_list =  [clip_name for clip_name in sorted(os.listdir(data_root)) if clip_name != '.DS_Store' and not clip_name.startswith('.')]
        self.ref_img_list = [clip_name for clip_name in sorted(os.listdir(imset)) if clip_name != '.DS_Store' and not clip_name.startswith('.')]

        if subset is not None:
            subset_set = set(subset)
            self.vid_list = [v for v in self.vid_list if v in subset_set]
            self.ref_img_list = [v for v in self.ref_img_list if v in subset_set]

        self.args = args

        # print(lst, len(lst), self.vid_list, self.vid_list_DAVIS2016, path.join(data_root, 'ImageSets', imset));assert 1==0

    def get_datasets(self):
        for video in self.vid_list:
            if video not in self.ref_img_list:
                continue

            # print(self.image_dir, video, path.join(self.image_dir, video));assert 1==0
            yield VideoReader_221128_TransColorization(video, 
                path.join(self.image_dir, video), 
                path.join(self.mask_dir, video),
                size=self.size,
                size_dir=path.join(self.size_dir, video),
                args=self.args,
                exemplar_path=self.exemplar_path
            )

    def __len__(self):
        return len(self.vid_list)
