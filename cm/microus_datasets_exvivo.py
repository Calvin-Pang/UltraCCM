from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import os
import pydicom
import glob
import math
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from .utils import *
import random
from mpi4py import MPI

def load_data(
    root_list,
    batch_size,
    axis_distance,
    scale,
    trim,
    mode,
    self_supervised = False,
    # random_sampling = True,
    # adaptive_sampling = False,
    select_k = None,
    num_workers = 8,
):
    if not self_supervised:
        image_dataset = MicroUSExvivoImageFolder(root_list = root_list,
                                                axis_distance = axis_distance,
                                                scale = scale,
                                                trim = trim,
                                                select_k = select_k,
                                                shard = MPI.COMM_WORLD.Get_rank(),
                                                num_shards = MPI.COMM_WORLD.Get_size(),)
        loader = DataLoader(dataset = image_dataset,
                            batch_size = batch_size,
                            shuffle = (mode == 'train'),
                            num_workers = num_workers,
                            drop_last = True
                            )   
    elif self_supervised:
        image_dataset = MicroUSSagittalImageFolder(root_list = root_list,
                                                   axis_distance = axis_distance,
                                                   scale = scale,
                                                   repeat = 10,
                                                   select_k = select_k,
                                                   ex_vivo = True,
                                                   trim = trim,
                                                   shard = MPI.COMM_WORLD.Get_rank(),
                                                   num_shards = MPI.COMM_WORLD.Get_size(),)
        patch_dataset = MicroUSSagittalPatchWrapper(dataset = image_dataset,
                                                    patch_size = 256,
                                                    random_sampling = True,
                                                    adaptive_sampling = False)
        loader = DataLoader(dataset = patch_dataset,
                            batch_size = batch_size,
                            shuffle = (mode == 'train'),
                            num_workers = num_workers,
                            drop_last = True
                            )
    return loader




class MicroUSExvivoImageFolder(Dataset):

    def __init__(self, 
                 root_list, 
                 axis_distance, 
                 scale = 8, 
                 trim = None, 
                 select_k = None, 
                 shard=0,
                 num_shards=1):
        self.axis_distance = axis_distance
        print('axis_distance', self.axis_distance)
        self.scale = scale
        self.root_list = root_list
        self.select_k = select_k
        self.lr_axial_planes = []
        self.lr_grids_gap_list = []
        self.lr_grids_orig_list = []
        self.hr_axial_planes = []
        self.hr_grids_orig_list = []
        self.hr_grids_uni_list = []
        self.meta_info_list = []

        for case in self.root_list.keys():
            filenames = sorted(glob.glob(os.path.join(case, '*.dcm')))
            num_files = len(filenames)

            (h_s, h_e), (w_s, w_e) = get_imaging_range(filenames[0])
            
            h, w = h_e - h_s, w_e - w_s
            h = (h // 16) * 16 # trim the h to be divisible by 16
            pixelspacing = pydicom.dcmread(filenames[0]).PixelSpacing[0]
            h_expand = int(self.axis_distance / pixelspacing) + h

            imgs = torch.zeros((num_files, h, w))
            thetas = torch.zeros((num_files, 1))
            for idx, filename in tqdm(enumerate(filenames), desc = 'Loading DICOMs...', leave = False):
                dicom = pydicom.dcmread(filename)
                imgs[idx] = torch.tensor(2 * (dicom.pixel_array[h_s : h_e, w_s : w_e][ : h] / 255) - 1, dtype = torch.float32)
                thetas[idx] = dicom.SliceLocation * np.pi / 180
            thetas = thetas - (thetas.max() + thetas.min()) / 2  # theta calibration
            thetas, indices = torch.sort(thetas, dim=0)
            imgs = imgs[indices.squeeze(1)]
            radius = (torch.flip(torch.arange(h), dims = (0,)) + int(self.axis_distance / pixelspacing))

            # default crop
            imgs = imgs[self.root_list[case][0] : self.root_list[case][1]]
            thetas = thetas[self.root_list[case][0] : self.root_list[case][1]]
            num_files = len(imgs)

            # trim the files to be divisible by 16
            num_files  = (num_files // 16) * 16
            imgs = imgs[ : num_files]
            thetas = thetas[ : num_files]
            if trim: # extra trim
                imgs = imgs[trim : ]
                thetas = thetas[trim : ]
            num_files = len(imgs)


            lr_thetas, lr_imgs = thetas[::self.scale], imgs[::self.scale]
            theta_min, theta_max = lr_thetas.min(), lr_thetas.max()
            lr_num_files = len(lr_imgs)
            hr_thetas, hr_imgs = thetas[: -self.scale + 1], imgs[ : -self.scale + 1]
            hr_num_files = len(hr_imgs)

            lr_grids = torch.stack(torch.meshgrid([radius / h_expand, lr_thetas.squeeze(-1)], indexing="ij"), dim=-1)
            hr_grids_orig = torch.stack(torch.meshgrid([radius / h_expand, hr_thetas.squeeze(-1)], indexing="ij"), dim=-1)
            
            lr_grids_gap = lr_grids.permute(2,0,1)
            lr_radius, lr_thetas = lr_grids_gap[0,:,:], lr_grids_gap[1,:,:]
            lr_thetas_pad = F.pad(lr_thetas, (1,1), 'replicate')
            lr_thetas_left_gap = (lr_thetas_pad[:,1:-1] - lr_thetas_pad[:,:-2]).unsqueeze(-3)
            lr_thetas_right_gap = (lr_thetas_pad[:,2:] - lr_thetas_pad[:,1:-1]).unsqueeze(-3)
            lr_grids_gap = torch.cat([lr_radius.unsqueeze(-3), lr_thetas_left_gap, lr_thetas_right_gap], dim = -3)
            
            hr_thetas_uni = torch.linspace(theta_min, theta_max, lr_num_files * self.scale)
            hr_grids_uni = torch.stack(torch.meshgrid([radius / h_expand, hr_thetas_uni], indexing="ij"), dim=-1).permute(2,0,1)
            

            for i in tqdm(range(w), desc = 'Converting to axial data...', leave = False):
                lr_axial_slice = torch.zeros((1, h, lr_num_files)) # [theta, r - dist, intensity]
                for j in range(lr_num_files):
                    lr_axial_slice[:, :, j] = lr_imgs[j][:, w - i - 1]
                self.lr_axial_planes.append(lr_axial_slice)
                self.lr_grids_gap_list.append(lr_grids_gap)
                self.lr_grids_orig_list.append(lr_grids)

                hr_axial_slice = torch.zeros((1, h, hr_num_files))
                for j in range(hr_num_files):
                    hr_axial_slice[:, :, j] = hr_imgs[j][:, w - i - 1]
                self.hr_axial_planes.append(hr_axial_slice)
                self.hr_grids_orig_list.append(hr_grids_orig)
                self.hr_grids_uni_list.append(hr_grids_uni)

                meta_info = {
                    'theta_min': theta_min,
                    'theta_max': theta_max,
                    'orig_size': (h, w),
                    'h_expand': h_expand,
                    'pixelspacing': pixelspacing,
                    'hr_theta_num': lr_num_files * self.scale,
                    'lr_theta_num': lr_num_files,
                    'scale': self.scale,
                    'axis_distance': self.axis_distance
                }
                self.meta_info_list.append(meta_info)

            print('Slice for this case', case,':', num_files, 'theta_min:', theta_min, 'theta_max:', theta_max)
            
        self.lr_axial_planes = self.lr_axial_planes[shard:][::num_shards]
        self.hr_axial_planes = self.hr_axial_planes[shard:][::num_shards]
        self.meta_info_list = self.meta_info_list[shard:][::num_shards]
        self.lr_grids_gap_list = self.lr_grids_gap_list[shard:][::num_shards]
        self.lr_grids_orig_list = self.lr_grids_orig_list[shard:][::num_shards]
        self.hr_grids_orig_list = self.hr_grids_orig_list[shard:][::num_shards]
        self.hr_grids_uni_list = self.hr_grids_uni_list[shard:][::num_shards]
        if select_k: 
            k = random.sample(range(len(self.lr_axial_planes)), select_k)
            self.lr_axial_planes = [self.lr_axial_planes[i] for i in k]
            self.hr_axial_planes = [self.hr_axial_planes[i] for i in k]
            self.meta_info_list = [self.meta_info_list[i] for i in k]
            self.lr_grids_gap_list = [self.lr_grids_gap_list[i] for i in k]
            self.lr_grids_orig_list = [self.lr_grids_orig_list[i] for i in k]
            self.hr_grids_orig_list = [self.hr_grids_orig_list[i] for i in k]
            self.hr_grids_uni_list = [self.hr_grids_uni_list[i] for i in k]

        
    def __len__(self):
        return len(self.lr_axial_planes)

    def __getitem__(self, idx):
        lr_img = self.lr_axial_planes[idx] 
        lr_grids_orig = self.lr_grids_orig_list[idx]
        lr_grids = self.lr_grids_gap_list[idx]
        hr_orig = self.hr_axial_planes[idx]
        hr_grids_orig = self.hr_grids_orig_list[idx]
        hr_grids_uni = self.hr_grids_uni_list[idx]
        meta_info = self.meta_info_list[idx]

        # print(lr_img.shape, lr_grids.shape, meta_info['hr_theta_num'])
        hr_inte = polar_intepolation_1d(lr_img, lr_grids_orig, hr_num = meta_info['hr_theta_num'])
        hr_img = polar_intepolation_1d(hr_orig, hr_grids_orig, hr_num = meta_info['hr_theta_num'])
        return {'lr_img': lr_img,
                'lr_grids': lr_grids,
                'hr_inte': hr_inte,
                'hr_grids': hr_grids_uni,
                'hr_img': hr_img,
                'meta_info': meta_info}


class MicroUSSagittalImageFolder(Dataset):

    def __init__(self, 
                 root_list, 
                 axis_distance, 
                 scale = 8, 
                 repeat = 1, 
                 select_k = None,  
                 ex_vivo = False,   
                 trim = 0,    
                 shard=0,
                 num_shards=1):
        super().__init__()
        self.axis_distance = axis_distance
        self.repeat = repeat
        self.scale = scale
        self.roots = root_list if not ex_vivo else root_list.keys()
        self.img_list = []
        self.meta_list = []
        for n, root_path in enumerate(self.roots):
            filenames = sorted(glob.glob(os.path.join(root_path, '*.dcm')))
            if not ex_vivo: filenames = trim_files(filenames)
            num_files = len(filenames)
            (h_s, h_e), (w_s, w_e) = get_imaging_range(filenames[0])
            
            h, w = h_e - h_s, w_e - w_s
            pixelspacing = pydicom.dcmread(filenames[0]).PixelSpacing[0]
            h_expand = int(axis_distance / pixelspacing) + h

            imgs = torch.zeros((num_files, h, w))
            thetas = torch.zeros((num_files, 1))
            
            for idx, filename in tqdm(enumerate(filenames), desc = 'Loading DICOMs of case ' + str(n+1) + '...', leave = False):
                dicom = pydicom.dcmread(filename)
                imgs[idx] = torch.tensor(2 * (dicom.pixel_array[h_s : h_e, w_s : w_e] / 255) - 1, dtype = torch.float32)
                thetas[idx] = dicom.SliceLocation * np.pi / 180
            thetas, indices = torch.sort(thetas, dim=0)
            imgs = imgs[indices.squeeze(1)]
            
            
            if ex_vivo: 
                imgs = imgs[root_list[root_path][0] : root_list[root_path][1]]
                thetas = thetas[root_list[root_path][0] : root_list[root_path][1]]

                num_files  = (num_files // 16) * 16
                imgs = imgs[ : num_files]
                thetas = thetas[ : num_files]

                if trim: 
                    imgs = imgs[trim : ]
                    thetas = thetas[trim : ]
                thetas, imgs = thetas[::self.scale], imgs[::self.scale]
                num_files = len(imgs)

            thetas = thetas - (thetas.max() + thetas.min()) / 2  # theta calibration
            theta_min, theta_max = thetas.min(), thetas.max()

        
            self.img_list += [imgs[i] for i in range(num_files)]

            meta_info = {
                'theta_min': theta_min,
                'theta_max': theta_max,
                'orig_size': (h, w),
                'h_expand': h_expand,
                'pixelspacing': pixelspacing,
                'hr_theta_num': num_files * scale,
                'lr_theta_num': num_files,
                'scale': scale,
                'axis_distance': axis_distance
            }
            print('Slice for this case', root_path,':', num_files, 'theta_min:', theta_min, 'theta_max:', theta_max)
            self.meta_list += [meta_info for _ in range(num_files)]
            
        self.img_list = self.img_list[shard:][::num_shards]
        self.meta_list = self.meta_list[shard:][::num_shards]
        if select_k: 
            self.img_list = self.img_list[:select_k]
            self.meta_list = self.meta_list[:select_k]
            # self.imgs = random.choices(self.imgs, k = select_k)
        
        
    def __len__(self):
        return len(self.img_list) * self.repeat

    def __getitem__(self, idx):
        x = self.img_list[idx % len(self.img_list)]  
        meta_info = self.meta_list[idx % len(self.img_list)]  
        return x, meta_info



class MicroUSSagittalPatchWrapper(Dataset):

    def __init__(self, dataset, patch_size = 256, random_sampling = True, adaptive_sampling = False):
        self.dataset = dataset
        self.patch_size = patch_size
        self.scale = self.dataset.scale
        self.random_sampling = random_sampling
        self.adaptive_sampling = adaptive_sampling
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, meta_info = self.dataset[idx]  # 833, 1372
        theta_max, theta_min = 1.57, -1.57 #meta_info['theta_max'], meta_info['theta_min']
        axis_distance, pixelspacing = meta_info['axis_distance'], meta_info['pixelspacing']
        h, w = meta_info['orig_size']
        h_expand = meta_info['h_expand']
        
        # hr_theta_num, lr_theta_num = meta_info['hr_theta_num'], meta_info['lr_theta_num'] # 8*N, N
        lr_theta_num = random.randint(150,500)
        hr_patch_theta_num, lr_patch_theta_num = self.patch_size, int(self.patch_size / self.scale) # 256, 32
        
        if self.random_sampling:
            lr_thetas = torch.sort(torch.rand(lr_theta_num - 2) * (theta_max - theta_min) + theta_min).values
            lr_thetas = torch.cat((torch.tensor([theta_min]), lr_thetas, torch.tensor([theta_max])))
        else:
            lr_thetas = torch.linspace(theta_min, theta_max, lr_theta_num)

        exacting = True
        while exacting:    
            lr_patch_thetas_start = random.randint(0, lr_theta_num - lr_patch_theta_num)
            lr_thetas_patch = lr_thetas[lr_patch_thetas_start : lr_patch_thetas_start + lr_patch_theta_num] # 32
            patch_theta_max, patch_theta_min = lr_thetas_patch.max(), lr_thetas_patch.min()
            hr_thetas_patch = torch.linspace(patch_theta_min, patch_theta_max, hr_patch_theta_num)
            
            radius = (torch.flip(torch.arange(h), dims = (0,)) + int(axis_distance / pixelspacing))# / h_expand
            if self.adaptive_sampling:
                focus_factor = 2.0
                prob = (radius[ : len(radius) - hr_patch_theta_num].float() ** focus_factor)
                prob /= prob.sum()
                patch_radius_start = torch.multinomial(prob, num_samples = 1).item()
            else:
                patch_radius_start = random.randint(0, len(radius) - hr_patch_theta_num)
            
            radius_patch = radius[patch_radius_start : patch_radius_start + hr_patch_theta_num].float()
            
            lr_patch, hr_patch, valid_patch = extract_patches(img, lr_thetas_patch, hr_thetas_patch, radius_patch)
            if valid_patch == True: exacting = False


        hr_grids = torch.stack(torch.meshgrid([radius_patch / h_expand, hr_thetas_patch], indexing="ij"), dim=-1)
        lr_grids = torch.stack(torch.meshgrid([radius_patch / h_expand, lr_thetas_patch], indexing="ij"), dim=-1)
        hr_inte = polar_intepolation_1d(lr_patch, lr_grids, self.patch_size)
        
        hr_grids = hr_grids.permute(2,0,1)
        lr_grids = lr_grids.permute(2,0,1)
        lr_radius, lr_thetas = lr_grids[0,:,:], lr_grids[1,:,:]
        lr_thetas_pad = F.pad(lr_thetas, (1,1), 'replicate')
        lr_thetas_left_gap = (lr_thetas_pad[:,1:-1] - lr_thetas_pad[:,:-2]).unsqueeze(-3)
        lr_thetas_right_gap = (lr_thetas_pad[:,2:] - lr_thetas_pad[:,1:-1]).unsqueeze(-3)
        lr_grids = torch.cat([lr_radius.unsqueeze(-3), lr_thetas_left_gap, lr_thetas_right_gap], dim = -3)
        # hr_inte = transforms.ToTensor()(transforms.ToPILImage()(lr_patch).resize((self.patch_size, self.patch_size)))

        return {'lr_img': lr_patch,
                'lr_grids': lr_grids,
                'hr_img': hr_patch,
                'hr_grids': hr_grids,
                'hr_inte': hr_inte,
                'meta_info': meta_info}


class MicroUSAxialImageFolder(Dataset):

    def __init__(self, root_path, axis_distance, scale = 8):
        self.axis_distance = axis_distance
        print('axis_distance', self.axis_distance)
        self.scale = scale
        self.filenames = sorted(glob.glob(os.path.join(root_path, '*.dcm')))
        self.filenames = trim_files(self.filenames)
        self.num_files = len(self.filenames)

        (h_s, h_e), (w_s, w_e) = get_imaging_range(self.filenames[0])
        
        self.h, self.w = h_e - h_s, w_e - w_s
        self.h = (self.h // 16) * 16
        self.pixelspacing = pydicom.dcmread(self.filenames[0]).PixelSpacing[0]
        self.h_expand = int(axis_distance / self.pixelspacing) + self.h

        self.imgs = torch.zeros((self.num_files, self.h, self.w))
        thetas = torch.zeros((self.num_files, 1))
        for idx, filename in tqdm(enumerate(self.filenames), desc = 'Loading DICOMs...', leave = False):
            dicom = pydicom.dcmread(filename)
            self.imgs[idx] = torch.tensor(2 * (dicom.pixel_array[h_s : h_e, w_s : w_e][ : self.h] / 255) - 1, dtype = torch.float32)
            thetas[idx] = dicom.SliceLocation * np.pi / 180
        thetas = thetas - (thetas.max() + thetas.min()) / 2  # theta calibration
        self.theta_min, self.theta_max = thetas.min(), thetas.max()
        thetas, indices = torch.sort(thetas, dim=0)
        self.imgs = self.imgs[indices.squeeze(1)]

        self.axial_planes = []
        radius = (torch.flip(torch.arange(self.h), dims = (0,)) + int(self.axis_distance / self.pixelspacing))
        self.lr_grids = torch.stack(torch.meshgrid([radius / self.h_expand, thetas.squeeze(-1)], indexing="ij"), dim=-1)
        
        lr_grids_gap = self.lr_grids.permute(2,0,1)
        lr_radius, lr_thetas = lr_grids_gap[0,:,:], lr_grids_gap[1,:,:]
        lr_thetas_pad = F.pad(lr_thetas, (1,1), 'replicate')
        lr_thetas_left_gap = (lr_thetas_pad[:,1:-1] - lr_thetas_pad[:,:-2]).unsqueeze(-3)
        lr_thetas_right_gap = (lr_thetas_pad[:,2:] - lr_thetas_pad[:,1:-1]).unsqueeze(-3)
        self.lr_grids_gap = torch.cat([lr_radius.unsqueeze(-3), lr_thetas_left_gap, lr_thetas_right_gap], dim = -3)
        
        
        hr_thetas = torch.linspace(self.theta_min, self.theta_max, self.num_files * self.scale)
        self.hr_grids = torch.stack(torch.meshgrid([radius / self.h_expand, hr_thetas], indexing="ij"), dim=-1)
        for i in tqdm(range(self.w), desc = 'Converting to axial data...', leave = False):
            axial_slice = torch.zeros((1, self.h, self.num_files)) # [theta, r - dist, intensity]
            for j in range(self.num_files):
                axial_slice[:, :, j] = self.imgs[j][:, self.w - i - 1]
            self.axial_planes.append(axial_slice)

        self.meta_info = {
            'theta_min': self.theta_min,
            'theta_max': self.theta_max,
            'orig_size': (self.h, self.w),
            'h_expand': self.h_expand,
            'pixelspacing': self.pixelspacing,
            'hr_theta_num': len(self.imgs) * self.scale,
            'lr_theta_num': len(self.imgs),
            'scale': self.scale,
            'axis_distance': self.axis_distance
        }
        
    def __len__(self):
        return len(self.axial_planes)

    def __getitem__(self, idx):
        lr_img = self.axial_planes[idx] 
        hr_inte = polar_intepolation_1d(lr_img, self.lr_grids, hr_num = self.num_files * self.scale)
        return {'lr_img': lr_img,
                'lr_grids': self.lr_grids_gap,
                'hr_inte': hr_inte,
                'hr_grids': self.hr_grids,
                'meta_info': self.meta_info}
