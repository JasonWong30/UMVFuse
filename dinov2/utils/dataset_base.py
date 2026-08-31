from pathlib import Path
import torch
from torch.utils import data as data
from dinov2.utils.utils import FileClient, paired_random_crop_multi, img2tensor, scandir, read_img_seq, normalize_image_np
import random
from os import path as osp
import glob

class VideoFusionDataset(data.Dataset):

    def __init__(self, opt):
        super(VideoFusionDataset, self).__init__()
        self.opt = opt
        # Keep paths to the high-resolution images.
        self.ir_gt_root = Path(opt["dataroot_gt_ir"])
        self.vi_gt_root = Path(opt["dataroot_gt_vi"])
        self.num_frame = opt["num_frame"]
        
        self.keys = []
        if opt["test_mode"]==True:
            with open(opt["meta_info_file_val"], "r") as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(" ")
                    self.keys.extend(
                        [f"{folder}/{i}/{frame_num}" for i in range(int(frame_num))]
                    )
        elif opt["test_mode"]==False:
            with open(opt["meta_info_file_train"], "r") as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(" ")
                    self.keys.extend(
                        [
                            f"{folder}/{i}/{frame_num}"
                            for i in range(1, int(frame_num) + 1)
                        ]
                    )
        else:
            with open(opt["meta_info_file_val"], "r") as fin:
                for line in fin:
                    folder, frame_num, _ = line.split(" ")
                    self.keys.extend(
                        [
                            f"{folder}/{i}/{frame_num}"
                            for i in range(1, int(frame_num) + 1)
                        ]
                    )            

        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        self.is_lmdb = False

        # temporal augmentation configs
        self.interval_list = opt.get("interval_list", [1])
        self.random_reverse = opt.get("random_reverse", False)
    def __getitem__(self, index):
        
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop("type"), **self.io_backend_opt
            )

        gt_size = self.opt["gt_size"]
        key = self.keys[index]
        clip_name, frame_name, frame_num = key.split("/")  # key example: 000/000000

        # determine the neighboring frames
        interval = random.choice(self.interval_list)
        # ensure not exceeding the borders
        start_frame_idx = int(frame_name) - 1
        if start_frame_idx > int(frame_num) - self.num_frame:
            start_frame_idx = random.randint(0, int(frame_num) - self.num_frame)
        end_frame_idx = start_frame_idx + self.num_frame

        neighbor_list = list(range(start_frame_idx, end_frame_idx, interval))

        # random reverse
        if self.random_reverse and random.random() < 0.5:
            neighbor_list.reverse()

        # get the neighboring GT frames
        img_ir_gts, img_vi_gts = [], []
        # Detect per-clip extension once (uniform .jpg or .png within a subfolder)
        if not self.is_lmdb:
            ir_dir = self.ir_gt_root / clip_name
            vi_dir = self.vi_gt_root / clip_name
            
            ir_ext = ".jpg" if any(ir_dir.glob("*.jpg")) else (".png" if any(ir_dir.glob("*.png")) else ".jpg")
            vi_ext = ".jpg" if any(vi_dir.glob("*.jpg")) else (".png" if any(vi_dir.glob("*.png")) else ".jpg")
        for neighbor in neighbor_list:
            num = neighbor + 1
            if self.is_lmdb:
                img_ir_gt_path = self.ir_gt_root / clip_name / f"{num}.jpg"
                img_vi_gt_path = self.vi_gt_root / clip_name / f"{num}.jpg"
            else:
                img_ir_gt_path = self.ir_gt_root / clip_name / f"{num}{ir_ext}"
                img_vi_gt_path = self.vi_gt_root / clip_name / f"{num}{vi_ext}"

            # get GT images
            img_ir_bytes = self.file_client.get(img_ir_gt_path, "gt")
            img_vi_bytes = self.file_client.get(img_vi_gt_path, "gt")
            img_ir_gt = normalize_image_np(img_ir_bytes, float32=True)
            img_vi_gt = normalize_image_np(img_vi_bytes, float32=True)
            img_ir_gts.append(img_ir_gt)
            img_vi_gts.append(img_vi_gt)
            # img_ir_gts, img_vi_gts = [], []
        # for neighbor in neighbor_list:
            # if self.is_lmdb:
            #     img_ir_gt_path = self.ir_gt_root / clip_name / f"{neighbor+1}.jpg"
            #     img_vi_gt_path = self.vi_gt_root / clip_name / f"{neighbor+1}.jpg"
            # else:
            #     img_ir_gt_path = self.ir_gt_root / clip_name / f"{neighbor+1}.jpg"
            #     img_vi_gt_path = self.vi_gt_root / clip_name / f"{neighbor+1}.jpg"

        #     # get GT images
        #     img_ir_bytes = self.file_client.get(img_ir_gt_path, "gt")
        #     img_vi_bytes = self.file_client.get(img_vi_gt_path, "gt")
        #     img_ir_gt = normalize_image_np(img_ir_bytes, float32=True)
        #     img_vi_gt = normalize_image_np(img_vi_bytes, float32=True)
        #     img_ir_gts.append(img_ir_gt)
        #     img_vi_gts.append(img_vi_gt)
        # randomly crop
        img_ir_gts, img_vi_gts = paired_random_crop_multi(
            img_ir_gts, img_vi_gts, gt_size, gt_path=img_ir_gt_path
        )
        
        # augmentation - flip, rotate
        img_ir_results = img_ir_gts
        img_vi_results = img_vi_gts
        
        # img_ir_results = augment(img_ir_results, self.opt["use_flip"], self.opt["use_rot"])
        # img_vi_results = augment(img_vi_results, self.opt["use_flip"], self.opt["use_rot"])

        img_ir_results = img2tensor(img_ir_results)
        img_vi_results = img2tensor(img_vi_results)
        
        img_ir_gts = torch.stack(img_ir_results, dim=0)
        img_vi_gts = torch.stack(img_vi_results, dim=0)

        # img_gts: (t, c, h, w)
        # key: str
        return {"gt_ir": img_ir_gts, "gt_vi": img_vi_gts, "key": key}

    def __len__(self):
        return len(self.keys)

class VideoTestDataset(data.Dataset):


    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt['cache_data']
        self.ir_gt_root = opt['dataroot_gt_ir']
        self.vi_gt_root = opt['dataroot_gt_vi']
        self.data_info = {'ir_gt_path': [], 'vi_gt_path': [], 'folder': [], 'idx': [], 'border': []}
        
        # file client (io backend)
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        assert self.io_backend_opt['type'] != 'lmdb', 'No need to use lmdb during validation/test.'
       
        self.imgs_ir_gt, self.imgs_vi_gt = {}, {}
        subfolders_ir_gt = sorted(glob.glob(osp.join(self.ir_gt_root, '*')))
        subfolders_vi_gt = sorted(glob.glob(osp.join(self.vi_gt_root, '*')))

        # name=opt['name'].lower()
        if opt['name'].lower() in ['ms3v', 'hdo']:
            for subfolder_ir_gt, subfolder_vi_gt in zip(subfolders_ir_gt, subfolders_vi_gt):
                # get frame list for gt
                subfolder_name = osp.basename(subfolder_ir_gt)
                from natsort import natsorted
                ir_img_paths_gt = natsorted(list(scandir(subfolder_ir_gt, full_path=True)))[:205]
                vi_img_paths_gt = natsorted(list(scandir(subfolder_vi_gt, full_path=True)))[:205]

                max_idx = len(ir_img_paths_gt)

                self.data_info['ir_gt_path'].extend(ir_img_paths_gt)
                self.data_info['vi_gt_path'].extend(vi_img_paths_gt)
                
                self.data_info['folder'].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info['idx'].append(f'{i}/{max_idx}')
                border_l = [0] * max_idx
                for i in range(self.opt['num_frame'] // 2):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info['border'].extend(border_l)

                # cache data or save the frame list
                if self.cache_data:                   
                    self.imgs_ir_gt[subfolder_name] = read_img_seq(ir_img_paths_gt)
                    self.imgs_vi_gt[subfolder_name] = read_img_seq(vi_img_paths_gt)
                else:
                    self.imgs_ir_gt[subfolder_name] = ir_img_paths_gt
                    self.imgs_vi_gt[subfolder_name] = vi_img_paths_gt

        else:
            raise ValueError(f'Non-supported video test dataset: {type(opt["name"])}')
        	

    def __getitem__(self, index):
        folder = self.data_info['folder'][index]
        idx, max_idx = self.data_info['idx'][index].split('/')
        idx, max_idx = int(idx), int(max_idx)
        border = self.data_info['border'][index]

        if self.cache_data:
            ir_img_gt = self.imgs_ir_gt[folder][idx]
            vi_img_gt = self.imgs_vi_gt[folder][idx]
        else:
            ir_img_gt = read_img_seq([self.imgs_ir_gt[folder][idx]])
            ir_img_gt.squeeze_(0)
            vi_img_gt = read_img_seq([self.imgs_vi_gt[folder][idx]])
            vi_img_gt.squeeze_(0)

        return {
            'gt_ir': ir_img_gt,  # (c, h, w)
            'gt_vi': vi_img_gt,  # (c, h, w)
            'folder': folder,  # folder name
            'idx': self.data_info['idx'][index],  # e.g., 0/99
            'border': border,  # 1 for border, 0 for non-border
        }

    def __len__(self):
        return len(self.data_info['ir_gt_path'])
    
class VideoFusionTestDataset(VideoTestDataset):
    
    def __init__(self, opt):
        super(VideoFusionTestDataset, self).__init__(opt)

        ori_folders = sorted(list(self.imgs_ir_gt.keys()))
        ori_num_frames_per_folder = {}
        ir_ori_imgs_gt_paths = {}
        vi_ori_imgs_gt_paths = {}
        now_idx = 0
        for folder in ori_folders:
            if self.cache_data:
                nf = self.imgs_ir_gt[folder].size()[0]
            else:
                nf = len(self.imgs_ir_gt[folder])
            ori_num_frames_per_folder[folder] = nf
            ir_ori_imgs_gt_paths[folder] = self.data_info['ir_gt_path'][now_idx:now_idx + nf]
            vi_ori_imgs_gt_paths[folder] = self.data_info['vi_gt_path'][now_idx:now_idx + nf]
            now_idx = now_idx + nf

        # Split Clips
        num_frame = self.opt['num_frame']
        num_overlap = self.opt['num_overlap']
        clip_data_info = {'ir_gt_path': [], 'vi_gt_path': [], 'folder': [], 'idx': [], 'border': []}
        clip_folders = []
        ir_clip_imgs_gt = {}
        vi_clip_imgs_gt = {}
        def natural_sort_key(s):
            import re
            return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

        for folder in ori_folders:
            num_all = ori_num_frames_per_folder[folder]
            self.imgs_ir_gt[folder] = sorted(self.imgs_ir_gt[folder], key=natural_sort_key)
            self.imgs_vi_gt[folder] = sorted(self.imgs_vi_gt[folder], key=natural_sort_key)
            ir_ori_imgs_gt_paths[folder] = sorted(ir_ori_imgs_gt_paths[folder], key=natural_sort_key)
            vi_ori_imgs_gt_paths[folder] = sorted(vi_ori_imgs_gt_paths[folder], key=natural_sort_key)

            for i in range((num_all - num_overlap) // (num_frame - num_overlap)):
                clip_folder = f'{folder}-{i:03d}'
                clip_folders.append(clip_folder)
                ir_clip_imgs_gt[clip_folder] = \
                    self.imgs_ir_gt[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame]
                vi_clip_imgs_gt[clip_folder] = \
                    self.imgs_vi_gt[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame]
                    
                clip_data_info['ir_gt_path'].extend(
                    ir_ori_imgs_gt_paths[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame])
                clip_data_info['vi_gt_path'].extend(
                    vi_ori_imgs_gt_paths[folder][i * (num_frame - num_overlap):i * (num_frame - num_overlap) + num_frame])
                
                clip_data_info['folder'].extend([clip_folder] * num_frame)
                for i in range(num_frame):
                    clip_data_info['idx'].append(f'{i}/{num_frame}')
                border_l = [0] * num_frame
                for i in range(num_frame // 2):
                    border_l[i] = 1
                    border_l[num_frame - i - 1] = 1
                clip_data_info['border'].extend(border_l)

            if (num_all - num_overlap) % (num_frame - num_overlap) != 0:
                clip_folder = f'{folder}-{((num_all - num_overlap) // (num_frame - num_overlap)):03d}'
                clip_folders.append(clip_folder)
                ir_clip_imgs_gt[clip_folder] = self.imgs_ir_gt[folder][-((num_all - num_overlap) % (num_frame - num_overlap)):]
                vi_clip_imgs_gt[clip_folder] = self.imgs_vi_gt[folder][-((num_all - num_overlap) % (num_frame - num_overlap)):]
                
                clip_data_info['ir_gt_path'].extend(ir_ori_imgs_gt_paths[folder][-num_frame:])
                clip_data_info['vi_gt_path'].extend(vi_ori_imgs_gt_paths[folder][-num_frame:])
                
                clip_data_info['folder'].extend([clip_folder] * num_frame)
                for i in range(num_frame):
                    clip_data_info['idx'].append(f'{i}/{num_frame}')
                border_l = [0] * num_frame
                for i in range(num_frame // 2):
                    border_l[i] = 1
                    border_l[num_frame - i - 1] = 1
                clip_data_info['border'].extend(border_l)

        self.folders = clip_folders
        self.imgs_ir_gt = ir_clip_imgs_gt
        self.imgs_vi_gt = vi_clip_imgs_gt
        self.data_info = clip_data_info

    def __getitem__(self, index):
        folder = self.folders[index]

        if self.cache_data:
            
            ir_imgs_gt = self.imgs_ir_gt[folder]
            vi_imgs_gt = self.imgs_vi_gt[folder]
        else:
            
            ir_img_paths_gt = self.imgs_ir_gt[folder]
            ir_imgs_gt = read_img_seq(ir_img_paths_gt)
            vi_img_paths_gt = self.imgs_vi_gt[folder]
            vi_imgs_gt = read_img_seq(vi_img_paths_gt)

        return {
            'gt_ir': ir_imgs_gt,
            'gt_vi': vi_imgs_gt,
            'folder': folder,
        }

    def __len__(self):
        return len(self.folders)
