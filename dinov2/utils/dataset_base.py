from torch.utils import data as data
from dinov2.utils.utils import scandir, read_img_seq
from os import path as osp
import glob


class VideoTestDataset(data.Dataset):

    def __init__(self, opt):
        super(VideoTestDataset, self).__init__()
        self.opt = opt
        self.cache_data = opt["cache_data"]
        self.ir_gt_root = opt["dataroot_gt_ir"]
        self.vi_gt_root = opt["dataroot_gt_vi"]
        self.data_info = {
            "ir_gt_path": [],
            "vi_gt_path": [],
            "folder": [],
            "idx": [],
            "border": [],
        }
        self.file_client = None
        self.io_backend_opt = opt["io_backend"]
        assert (
            self.io_backend_opt["type"] != "lmdb"
        ), "No need to use lmdb during validation/test."
        (self.imgs_ir_gt, self.imgs_vi_gt) = ({}, {})
        if "meta_info_file_test" in opt:
            with open(opt["meta_info_file_val"], "r") as fin:
                subfolders = [line.split(" ")[0] for line in fin]
                subfolders_gt = [osp.join(self.gt_root, key) for key in subfolders]
        else:
            subfolders_ir_gt = sorted(glob.glob(osp.join(self.ir_gt_root, "*")))
            subfolders_vi_gt = sorted(glob.glob(osp.join(self.vi_gt_root, "*")))
        if opt["name"].lower() in ["ms3v", "hdo"]:
            for subfolder_ir_gt, subfolder_vi_gt in zip(
                subfolders_ir_gt, subfolders_vi_gt
            ):
                subfolder_name = osp.basename(subfolder_ir_gt)
                from natsort import natsorted

                ir_img_paths_gt = natsorted(
                    list(scandir(subfolder_ir_gt, full_path=True))
                )[:205]
                vi_img_paths_gt = natsorted(
                    list(scandir(subfolder_vi_gt, full_path=True))
                )[:205]
                max_idx = len(ir_img_paths_gt)
                self.data_info["ir_gt_path"].extend(ir_img_paths_gt)
                self.data_info["vi_gt_path"].extend(vi_img_paths_gt)
                self.data_info["folder"].extend([subfolder_name] * max_idx)
                for i in range(max_idx):
                    self.data_info["idx"].append(f"{i}/{max_idx}")
                border_l = [0] * max_idx
                for i in range(self.opt["num_frame"] // 2):
                    border_l[i] = 1
                    border_l[max_idx - i - 1] = 1
                self.data_info["border"].extend(border_l)
                if self.cache_data:
                    self.imgs_ir_gt[subfolder_name] = read_img_seq(ir_img_paths_gt)
                    self.imgs_vi_gt[subfolder_name] = read_img_seq(vi_img_paths_gt)
                else:
                    self.imgs_ir_gt[subfolder_name] = ir_img_paths_gt
                    self.imgs_vi_gt[subfolder_name] = vi_img_paths_gt
        else:
            raise ValueError(f"Non-supported video test dataset: {type(opt['name'])}")


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
            ir_ori_imgs_gt_paths[folder] = self.data_info["ir_gt_path"][
                now_idx : now_idx + nf
            ]
            vi_ori_imgs_gt_paths[folder] = self.data_info["vi_gt_path"][
                now_idx : now_idx + nf
            ]
            now_idx = now_idx + nf
        num_frame = self.opt["num_frame"]
        num_overlap = self.opt["num_overlap"]
        clip_data_info = {
            "ir_gt_path": [],
            "vi_gt_path": [],
            "folder": [],
            "idx": [],
            "border": [],
        }
        clip_folders = []
        ir_clip_imgs_gt = {}
        vi_clip_imgs_gt = {}

        def natural_sort_key(s):
            import re

            return [
                int(text) if text.isdigit() else text.lower()
                for text in re.split("(\\d+)", s)
            ]

        for folder in ori_folders:
            num_all = ori_num_frames_per_folder[folder]
            self.imgs_ir_gt[folder] = sorted(
                self.imgs_ir_gt[folder], key=natural_sort_key
            )
            self.imgs_vi_gt[folder] = sorted(
                self.imgs_vi_gt[folder], key=natural_sort_key
            )
            ir_ori_imgs_gt_paths[folder] = sorted(
                ir_ori_imgs_gt_paths[folder], key=natural_sort_key
            )
            vi_ori_imgs_gt_paths[folder] = sorted(
                vi_ori_imgs_gt_paths[folder], key=natural_sort_key
            )
            for i in range((num_all - num_overlap) // (num_frame - num_overlap)):
                clip_folder = f"{folder}-{i:03d}"
                clip_folders.append(clip_folder)
                ir_clip_imgs_gt[clip_folder] = self.imgs_ir_gt[folder][
                    i * (num_frame - num_overlap) : i * (num_frame - num_overlap)
                    + num_frame
                ]
                vi_clip_imgs_gt[clip_folder] = self.imgs_vi_gt[folder][
                    i * (num_frame - num_overlap) : i * (num_frame - num_overlap)
                    + num_frame
                ]
                clip_data_info["ir_gt_path"].extend(
                    ir_ori_imgs_gt_paths[folder][
                        i * (num_frame - num_overlap) : i * (num_frame - num_overlap)
                        + num_frame
                    ]
                )
                clip_data_info["vi_gt_path"].extend(
                    vi_ori_imgs_gt_paths[folder][
                        i * (num_frame - num_overlap) : i * (num_frame - num_overlap)
                        + num_frame
                    ]
                )
                clip_data_info["folder"].extend([clip_folder] * num_frame)
                for i in range(num_frame):
                    clip_data_info["idx"].append(f"{i}/{num_frame}")
                border_l = [0] * num_frame
                for i in range(num_frame // 2):
                    border_l[i] = 1
                    border_l[num_frame - i - 1] = 1
                clip_data_info["border"].extend(border_l)
            if (num_all - num_overlap) % (num_frame - num_overlap) != 0:
                clip_folder = f"{folder}-{(num_all - num_overlap) // (num_frame - num_overlap):03d}"
                clip_folders.append(clip_folder)
                ir_clip_imgs_gt[clip_folder] = self.imgs_ir_gt[folder][
                    -((num_all - num_overlap) % (num_frame - num_overlap)) :
                ]
                vi_clip_imgs_gt[clip_folder] = self.imgs_vi_gt[folder][
                    -((num_all - num_overlap) % (num_frame - num_overlap)) :
                ]
                clip_data_info["ir_gt_path"].extend(
                    ir_ori_imgs_gt_paths[folder][-num_frame:]
                )
                clip_data_info["vi_gt_path"].extend(
                    vi_ori_imgs_gt_paths[folder][-num_frame:]
                )
                clip_data_info["folder"].extend([clip_folder] * num_frame)
                for i in range(num_frame):
                    clip_data_info["idx"].append(f"{i}/{num_frame}")
                border_l = [0] * num_frame
                for i in range(num_frame // 2):
                    border_l[i] = 1
                    border_l[num_frame - i - 1] = 1
                clip_data_info["border"].extend(border_l)
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
        return {"gt_ir": ir_imgs_gt, "gt_vi": vi_imgs_gt, "folder": folder}

    def __len__(self):
        return len(self.folders)
