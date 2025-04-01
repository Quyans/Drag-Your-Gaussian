import torch    
import torch.nn as nn
import torch.nn.functional as F

from plyfile import PlyData, PlyElement
import numpy as np
from utils import export_points_for_gaussians_with_two_masks

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def inverse_softplus(x, beta=1):
    return (torch.exp(beta * x) - 1).log() / beta

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def load_ply(path):
    #self.spatial_lr_scale = 1
    plydata = PlyData.read(path)
    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
    print("Number of points at loading : ", xyz.shape[0])
    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])
    
    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    #assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
    max_sh_degree = int(((len(extra_f_names) + 3) / 3) ** 0.5 - 1)
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (max_sh_degree + 1) ** 2 - 1))
    
    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])
    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    xyz = torch.from_numpy(xyz).float().cuda()
    features_dc = torch.from_numpy(features_dc).transpose(1, 2).float().cuda()
    features_rest = torch.from_numpy(features_extra).transpose(1, 2).float().cuda()
    features = torch.cat((features_dc, features_rest), dim=1)

    opacities = torch.sigmoid(torch.from_numpy(opacities)).float().cuda()
    scales = torch.exp(torch.from_numpy(scales)).float().cuda()
    rots = torch.nn.functional.normalize(torch.from_numpy(rots)).float().cuda()

    return xyz, features, opacities, scales, rots


def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

class GaussiansManager():
    def __init__(self, xyz, features, opacity, scales, rotations, lrs, edit_mask,only_optimize_3d_mask, use_knn=False, knn_coeff=0.3, other_coeff=0, knn_number=5):
        self.only_optimize_3d_mask = only_optimize_3d_mask

        self.use_knn = use_knn
        from pytorch3d.ops import knn_points
        knn_index = knn_points(
            xyz[edit_mask].unsqueeze(0),
            xyz.unsqueeze(0),
            K=knn_number,
        ).idx.squeeze(0)
        # self.knn_index [N,10]

        unique_indices = torch.unique(knn_index)
        sorted_indices = torch.sort(unique_indices).values
        knn_points_mask = torch.zeros_like(edit_mask, dtype=torch.bool)
        knn_points_mask[sorted_indices] = True

        inter_points_mask = edit_mask & knn_points_mask
        knn_novel_points_mask = knn_points_mask & ~inter_points_mask
        # 根据unique_selected_indices设置新mask中相应的值为True

        #print("Novel KNN points num: ", knn_novel_points_mask.sum())
        #NOTE:output KNN.ply
        #export_points_for_gaussians_with_two_masks(f"logs/KNN_points.ply", xyz, edit_mask, knn_novel_points_mask)
        
        other_mask = torch.ones_like(edit_mask, dtype=torch.bool, device=edit_mask.device)
        other_mask[edit_mask | knn_novel_points_mask] = False
        self.masks_lens_group = [edit_mask.sum(), knn_novel_points_mask.sum(), other_mask.sum()]


        """
        # self.register_buffer("xyz", nn.Parameter(xyz.contiguous().float().detach().clone().requires_grad_(True)))
        self._xyz = nn.Parameter(xyz.contiguous().float().detach().clone().requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,0:1,:].contiguous().float().detach().clone().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,1:,:].contiguous().float().detach().clone().requires_grad_(True))
    
        #设低原gs opacity
        #self._opacity_origin = opacity[0][edit_mask]
        #opacity[edit_mask] = torch.tensor(0.05)
        self._opacity = nn.Parameter(inverse_sigmoid(opacity).contiguous().float().detach().clone().requires_grad_(True))
        self._scales = nn.Parameter(torch.log(scales + 1e-8).contiguous().float().detach().clone().requires_grad_(True))
        self._rotations = nn.Parameter(rotations.contiguous().float().detach().clone().requires_grad_(True))
        """
        self._xyz_3dmask = nn.Parameter(xyz[edit_mask].contiguous().float().detach().clone().requires_grad_(True))
        self._features_dc_3dmask = nn.Parameter(features[edit_mask,0:1,:].contiguous().float().detach().clone().requires_grad_(True))
        self._features_rest_3dmask = nn.Parameter(features[edit_mask,1:,:].contiguous().float().detach().clone().requires_grad_(True))
        self._opacity_3dmask = nn.Parameter(inverse_sigmoid(opacity[edit_mask]).contiguous().float().detach().clone().requires_grad_(True))
        self._scales_3dmask = nn.Parameter(torch.log(scales[edit_mask] + 1e-8).contiguous().float().detach().clone().requires_grad_(True))
        self._rotations_3dmask = nn.Parameter(rotations[edit_mask].contiguous().float().detach().clone().requires_grad_(True))
        
        self._xyz_knn = nn.Parameter(xyz[knn_novel_points_mask].contiguous().float().detach().clone().requires_grad_(True))
        self._features_dc_knn = nn.Parameter(features[knn_novel_points_mask,0:1,:].contiguous().float().detach().clone().requires_grad_(True))
        self._features_rest_knn = nn.Parameter(features[knn_novel_points_mask,1:,:].contiguous().float().detach().clone().requires_grad_(True))
        self._opacity_knn = nn.Parameter(inverse_sigmoid(opacity[knn_novel_points_mask]).contiguous().float().detach().clone().requires_grad_(True))
        self._scales_knn = nn.Parameter(torch.log(scales[knn_novel_points_mask] + 1e-8).contiguous().float().detach().clone().requires_grad_(True))
        self._rotations_knn = nn.Parameter(rotations[knn_novel_points_mask].contiguous().float().detach().clone().requires_grad_(True))
        
        self._xyz_other = nn.Parameter(xyz[other_mask].contiguous().float().detach().clone().requires_grad_(True))
        self._features_dc_other = nn.Parameter(features[other_mask,0:1,:].contiguous().float().detach().clone().requires_grad_(True))
        self._features_rest_other = nn.Parameter(features[other_mask,1:,:].contiguous().float().detach().clone().requires_grad_(True))
        self._opacity_other = nn.Parameter(inverse_sigmoid(opacity[other_mask]).contiguous().float().detach().clone().requires_grad_(True))
        self._scales_other = nn.Parameter(torch.log(scales[other_mask] + 1e-8).contiguous().float().detach().clone().requires_grad_(True))
        self._rotations_other = nn.Parameter(rotations[other_mask].contiguous().float().detach().clone().requires_grad_(True))
        
        self.device = self._xyz.device
        
        if self.use_knn:
            knn_coeff = knn_coeff
            other_coeff = other_coeff
        else:
            if self.only_optimize_3d_mask:
                knn_coeff = 0
                other_coeff = 0
            else:
                knn_coeff = 1
                other_coeff = 1
        self.optimizer = torch.optim.Adam([{"name": "xyz_3dmask", 'params': [self._xyz_3dmask], 'lr': lrs['xyz']},
                                            {"name": "features_dc_3dmask", 'params': [self._features_dc_3dmask], 'lr': lrs['features']},
                                            {"name": "features_rest_3dmask", 'params': [self._features_rest_3dmask], 'lr': lrs['features'] / 20},
                                            {"name": "opacity_3dmask", 'params': [self._opacity_3dmask], 'lr': lrs['opacity']},
                                            {"name": "scales_3dmask", 'params': [self._scales_3dmask], 'lr': lrs['scales']},
                                            {"name": "rotations_3dmask", 'params': [self._rotations_3dmask], 'lr': lrs['rotations']},
                                            
                                            {"name": "xyz_knn", 'params': [self._xyz_knn], 'lr': lrs['xyz']*knn_coeff},
                                            {"name": "features_dc_knn", 'params': [self._features_dc_knn], 'lr': lrs['features']*knn_coeff },
                                            {"name": "features_rest_knn", 'params': [self._features_rest_knn], 'lr': lrs['features'] / 20*knn_coeff},
                                            {"name": "opacity_knn", 'params': [self._opacity_knn], 'lr': lrs['opacity']*knn_coeff},
                                            {"name": "scales_knn", 'params': [self._scales_knn], 'lr': lrs['scales']*knn_coeff},
                                            {"name": "rotations_knn", 'params': [self._rotations_knn], 'lr': lrs['rotations']*knn_coeff},
                                            
                                            {"name": "xyz_other", 'params': [self._xyz_other], 'lr': lrs['xyz']*other_coeff},
                                            {"name": "features_dc_other", 'params': [self._features_dc_other], 'lr': lrs['features']*other_coeff },
                                            {"name": "features_rest_other", 'params': [self._features_rest_other], 'lr': lrs['features'] / 20*other_coeff},
                                            {"name": "opacity_other", 'params': [self._opacity_other], 'lr': lrs['opacity']*other_coeff},
                                            {"name": "scales_other", 'params': [self._scales_other], 'lr': lrs['scales']*other_coeff},
                                            {"name": "rotations_other", 'params': [self._rotations_other], 'lr': lrs['rotations']*other_coeff},
                                            ], betas=(0.9, 0.99), lr=0.0, eps=1e-15)

        # 目前没用到
        # self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale* scale_small,
        #                                             lr_final=training_args.position_lr_final*self.spatial_lr_scale* scale_small,
        #                                             lr_delay_mult=training_args.position_lr_delay_mult,
        #                                             max_steps=training_args.position_lr_max_steps)
        
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device=self.device)
        self.denom = torch.zeros((self._xyz.shape[0], 1), device=self.device)
        self.max_radii2D = torch.zeros((self._xyz.shape[0],), device=self.device)
        self.is_visible = torch.zeros((self._xyz.shape[0],), device=self.device)

        self.percent_dense = 0.003
        
        # compute bbox of the scene
        self.xyz_max, _ = torch.max(self._xyz.detach(), dim=0)
        self.xyz_min, _ = torch.min(self._xyz.detach(), dim=0)
        self.bounds = self.xyz_max - self.xyz_min
        # normalized xyz into [-1, 1]
        # normalized_points = 2 * (points - xyz_min) / bounds - 1

    # 设置函数为成员函数，直接通过名字self._xyz调用
    @property
    def _xyz(self):
        return torch.cat((self._xyz_3dmask, self._xyz_knn, self._xyz_other), dim=0)
    
    @property
    def _features_dc(self):
        return torch.cat((self._features_dc_3dmask, self._features_dc_knn, self._features_dc_other), dim=0)
    
    @property
    def _features_rest(self):
        return torch.cat((self._features_rest_3dmask, self._features_rest_knn, self._features_rest_other), dim=0)
    
    @property
    def _opacity(self):
        return torch.cat((self._opacity_3dmask, self._opacity_knn, self._opacity_other), dim=0)
    
    @property
    def _scales(self):
        return torch.cat((self._scales_3dmask, self._scales_knn, self._scales_other), dim=0)
    
    @property
    def _rotations(self):
        return torch.cat((self._rotations_3dmask, self._rotations_knn, self._rotations_other), dim=0)
    
    
    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                # print("xyz lr : ", lr)
            # if param_group["name"] == "f_dc":
            #     # import pdb; pdb.set_trace()
            #     param_group['lr'] = param_group['lr'] * ((0.5) ** (1.0 / 1200.0))
            # if param_group["name"] == "f_rest":
            #     # import pdb; pdb.set_trace()
            #     param_group['lr'] = param_group['lr'] * ((0.5) ** (1.0 / 1200.0))

    
    def __call__(self):
        xyz = self._xyz 
        #NOTE: dc rest
        #features = self._features
        features = torch.cat((self._features_dc, self._features_rest), dim=1)
        opacity = torch.sigmoid(self._opacity)
        scales = torch.exp(self._scales)
        rotations = torch.nn.functional.normalize(self._rotations, dim=-1)
        return xyz, features, opacity, scales, rotations

    def normalize_xyz(self, xyz):
        return 2 * (xyz - self.xyz_min) / self.bounds - 1
    
    # def get_3dmask(self):
    #     return self.masks_group[0]
    # def get_group_mask(self):
    #     return self.masks_group[0], self.masks_group[1], self.masks_group[2]
    
    def get_3dmask(self):
        ind_3d = self.masks_lens_group[0]
        ind_knn = self.masks_lens_group[0]+self.masks_lens_group[1]
        ind_other = self.masks_lens_group[0]+self.masks_lens_group[1]+self.masks_lens_group[2]
        
        mask_3d = torch.zeros(self._xyz.shape[0],device=self._xyz.device,dtype=torch.bool)
        mask_3d[:ind_3d] = True
        return mask_3d

    def get_knnmask(self):
        ind_3d = self.masks_lens_group[0]
        ind_knn = self.masks_lens_group[0]+self.masks_lens_group[1]
        ind_other = self.masks_lens_group[0]+self.masks_lens_group[1]+self.masks_lens_group[2]
        
        mask_knn = torch.zeros(self._xyz.shape[0],device=self._xyz.device,dtype=torch.bool)
        mask_knn[ind_3d:ind_knn] = True
        return mask_knn
    
    def get_othermask(self):
        ind_3d = self.masks_lens_group[0]
        ind_knn = self.masks_lens_group[0]+self.masks_lens_group[1]
        ind_other = self.masks_lens_group[0]+self.masks_lens_group[1]+self.masks_lens_group[2]
        
        mask_other = torch.zeros(self._xyz.shape[0],device=self._xyz.device,dtype=torch.bool)
        mask_other[ind_knn:ind_other] = True
        return mask_other
    
    def get_3dmask_len(self):
        return self.masks_lens_group[0]
    def get_group_mask_len(self):
        return self.masks_lens_group[0], self.masks_lens_group[1], self.masks_lens_group[2]

    @torch.no_grad()
    def densify_and_prune(self, max_grad=4, extent=2, opacity_threshold=0.001, K=1, max_num_points=500000, max_screen_size=2):

        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        
        if self.only_optimize_3d_mask:
            if self.use_knn:
                #仅允许3d mask内的gaussians进行密集化
                mask_3d, mask_knn, mask_other = self.get_group_mask_len()
                grads[mask_3d+mask_knn:] = 0.0
            else:
                #仅允许3d mask内的gaussians进行密集化
                mask_3d, mask_knn, mask_other = self.get_group_mask_len()
                grads[mask_3d:] = 0.0

        if True:
            self.densify_and_clone(grads, max_grad, scene_extent=extent)
            self.densify_and_split(grads, max_grad, scene_extent=extent)

        prune_mask = (torch.sigmoid(self._opacity) < opacity_threshold).squeeze() # or torch.exp(self._scales.max(dim=1).values) < 0.001
        mask_3d, mask_knn, mask_other = self.get_group_mask_len()

        self.prune_points(prune_mask[:mask_3d], prune_mask[mask_3d: mask_3d+mask_knn], prune_mask[mask_3d+mask_knn:mask_3d+mask_knn+mask_other])
        
        torch.cuda.empty_cache()

    """
    def prune_points(self, mask):
        valid_points_mask = ~mask

        optimizable_tensors = self.prune_optimizer(valid_points_mask)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scales = optimizable_tensors["scales"]
        self._rotations = optimizable_tensors["rotations"]

        #prune mask
        self.edit_mask = self.edit_mask[valid_points_mask]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.is_visible = self.is_visible[valid_points_mask]

    def prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors
    """
    def prune_points(self, mask_3d, mask_knn, mask_other):
        valid_points_mask_3d = ~mask_3d
        valid_points_mask_knn = ~mask_knn
        valid_points_mask_other = ~mask_other

        optimizable_tensors = self.prune_optimizer(valid_points_mask_3d, valid_points_mask_knn, valid_points_mask_other)
        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["features_dc"]
        # self._features_rest = optimizable_tensors["features_rest"]
        # self._opacity = optimizable_tensors["opacity"]
        # self._scales = optimizable_tensors["scales"]
        # self._rotations = optimizable_tensors["rotations"]
        
        self._xyz_3dmask = optimizable_tensors["xyz_3dmask"]
        self._xyz_knn = optimizable_tensors["xyz_knn"]
        self._xyz_other = optimizable_tensors["xyz_other"]
        
        self._features_dc_3dmask = optimizable_tensors["features_dc_3dmask"]
        self._features_dc_knn = optimizable_tensors["features_dc_knn"]
        self._features_dc_other = optimizable_tensors["features_dc_other"]
        
        self._features_rest_3dmask = optimizable_tensors["features_rest_3dmask"]
        self._features_rest_knn = optimizable_tensors["features_rest_knn"]
        self._features_rest_other = optimizable_tensors["features_rest_other"]
        
        self._opacity_3dmask = optimizable_tensors["opacity_3dmask"]
        self._opacity_knn = optimizable_tensors["opacity_knn"]
        self._opacity_other = optimizable_tensors["opacity_other"]

        self._scales_3dmask = optimizable_tensors["scales_3dmask"]
        self._scales_knn = optimizable_tensors["scales_knn"]
        self._scales_other = optimizable_tensors["scales_other"]
        
        self._rotations_3dmask = optimizable_tensors["rotations_3dmask"]
        self._rotations_knn = optimizable_tensors["rotations_knn"]
        self._rotations_other = optimizable_tensors["rotations_other"]
        
        #prune mask
        # self.edit_mask = self.edit_mask[valid_points_mask]
                
        # total_number = valid_points_mask_3d.shape[0] + valid_points_mask_knn.shape[0] + valid_points_mask_other.shape[0]
        # zero_mask = torch.zeros((total_number,), device=self.device, dtype=torch.bool)
        

        self.masks_lens_group = [valid_points_mask_3d.sum(), valid_points_mask_knn.sum(), valid_points_mask_other.sum()]
        
        # self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        # self.denom = self.denom[valid_points_mask]
        # self.max_radii2D = self.max_radii2D[valid_points_mask]
        # self.is_visible = self.is_visible[valid_points_mask]
        self.xyz_gradient_accum = self.xyz_gradient_accum[torch.cat((valid_points_mask_3d, valid_points_mask_knn, valid_points_mask_other), dim=0)]
        self.denom = self.denom[torch.cat((valid_points_mask_3d, valid_points_mask_knn, valid_points_mask_other), dim=0)]
        self.max_radii2D = self.max_radii2D[torch.cat((valid_points_mask_3d, valid_points_mask_knn, valid_points_mask_other), dim=0)]
        self.is_visible = self.is_visible[torch.cat((valid_points_mask_3d, valid_points_mask_knn, valid_points_mask_other), dim=0)]

    def prune_optimizer(self, mask_3d, mask_knn, mask_other):
        
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            group_name = group.get('name', '')
            if group_name.endswith('_3dmask'):                     
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask_3d]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask_3d]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask_3d].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask_3d].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
                    
        for group in self.optimizer.param_groups:
            group_name = group.get('name', '')
            if group_name.endswith('_knn'):                     
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask_knn]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask_knn]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask_knn].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask_knn].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
                    
        for group in self.optimizer.param_groups:
            group_name = group.get('name', '')
            if group_name.endswith('_other'):                     
                stored_state = self.optimizer.state.get(group['params'][0], None)
                if stored_state is not None:
                    stored_state["exp_avg"] = stored_state["exp_avg"][mask_other]
                    stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask_other]

                    del self.optimizer.state[group['params'][0]]
                    group["params"][0] = nn.Parameter((group["params"][0][mask_other].requires_grad_(True)))
                    self.optimizer.state[group['params'][0]] = stored_state

                    optimizable_tensors[group["name"]] = group["params"][0]
                else:
                    group["params"][0] = nn.Parameter(group["params"][0][mask_other].requires_grad_(True))
                    optimizable_tensors[group["name"]] = group["params"][0]
                    
        return optimizable_tensors
    """
    def add_points(self, params):

        num_points = params["xyz"].shape[0]

        optimizable_tensors = self.cat_tensors_to_optimizer(params)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["features_dc"]
        self._features_rest = optimizable_tensors["features_rest"]

        self._opacity = optimizable_tensors["opacity"]
        self._scales = optimizable_tensors["scales"]
        self._rotations = optimizable_tensors["rotations"]

        #add mask
        # self.edit_mask = torch.cat([self.edit_mask,torch.ones(num_points,dtype=torch.bool,device=self.edit_mask.device)],dim=0)
        # 让mask_group的第一个进行这个操作，剩下的填充为False
        # 现在相当于是新加的点都是3d_mask的
        self.masks_group[0] = torch.cat([self.masks_group[0],torch.ones(num_points,dtype=torch.bool,device=self.masks_group[0].device)],dim=0)
        for i in range(1,len(self.masks_group)):
            self.masks_group[i] = torch.cat([self.masks_group[i],torch.zeros(num_points,dtype=torch.bool,device=self.masks_group[i].device)],dim=0)
        
        
       
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((num_points, 1), device=self.device)])
        self.denom = torch.cat([self.denom, torch.zeros((num_points, 1), device=self.device)])
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros((num_points,), device=self.device)])
        self.is_visible = torch.cat([self.is_visible, torch.zeros((num_points,), device=self.device)])
    """
    def add_points(self, params):

        num_points = params["xyz_3dmask"].shape[0] + params["xyz_knn"].shape[0] + params["xyz_other"].shape[0]

        optimizable_tensors = self.cat_tensors_to_optimizer(params)
        # self._xyz = optimizable_tensors["xyz"]
        # self._features_dc = optimizable_tensors["features_dc"]
        # self._features_rest = optimizable_tensors["features_rest"]

        # self._opacity = optimizable_tensors["opacity"]
        # self._scales = optimizable_tensors["scales"]
        # self._rotations = optimizable_tensors["rotations"]
        
        self._xyz_3dmask = optimizable_tensors["xyz_3dmask"]
        self._xyz_knn = optimizable_tensors["xyz_knn"]
        self._xyz_other = optimizable_tensors["xyz_other"]
        
        self._features_dc_3dmask = optimizable_tensors["features_dc_3dmask"]
        self._features_dc_knn = optimizable_tensors["features_dc_knn"]
        self._features_dc_other = optimizable_tensors["features_dc_other"]
        
        self._features_rest_3dmask = optimizable_tensors["features_rest_3dmask"]
        self._features_rest_knn = optimizable_tensors["features_rest_knn"]
        self._features_rest_other = optimizable_tensors["features_rest_other"]
        
        self._opacity_3dmask = optimizable_tensors["opacity_3dmask"]
        self._opacity_knn = optimizable_tensors["opacity_knn"]
        self._opacity_other = optimizable_tensors["opacity_other"]
        
        self._scales_3dmask = optimizable_tensors["scales_3dmask"]
        self._scales_knn = optimizable_tensors["scales_knn"]
        self._scales_other = optimizable_tensors["scales_other"]
        
        self._rotations_3dmask = optimizable_tensors["rotations_3dmask"]
        self._rotations_knn = optimizable_tensors["rotations_knn"]
        self._rotations_other = optimizable_tensors["rotations_other"]
        
        

        #add mask
        # self.edit_mask = torch.cat([self.edit_mask,torch.ones(num_points,dtype=torch.bool,device=self.edit_mask.device)],dim=0)
        # 让mask_group的第一个进行这个操作，剩下的填充为False
        # 现在相当于是新加的点都是3d_mask的
        
        self.masks_lens_group = [self.masks_lens_group[0]+params["xyz_3dmask"].shape[0], self.masks_lens_group[1]+params["xyz_knn"].shape[0], self.masks_lens_group[2]+params["xyz_other"].shape[0]]
            
        self.xyz_gradient_accum = torch.cat([self.xyz_gradient_accum, torch.zeros((num_points, 1), device=self.device)])
        self.denom = torch.cat([self.denom, torch.zeros((num_points, 1), device=self.device)])
        self.max_radii2D = torch.cat([self.max_radii2D, torch.zeros((num_points,), device=self.device)])
        self.is_visible = torch.cat([self.is_visible, torch.zeros((num_points,), device=self.device)])

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors
    """
    def densify_and_split(self, grads, grad_threshold=0.04, scene_extent=2, N=2):
        n_init_points = self._xyz.shape[0]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(torch.exp(self._scales), dim=1).values > self.percent_dense * scene_extent
        )

        stds = torch.exp(self._scales[selected_pts_mask]).repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self.device)
        samples = torch.normal(mean=means, std=stds)

        rots = build_rotation(self._rotations[selected_pts_mask]).repeat(N, 1, 1)

        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scales = torch.log(torch.exp(self._scales[selected_pts_mask]) / (0.8*N)).repeat(N, 1)
        new_rotations = self._rotations[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        params = {  "xyz": new_xyz,
                    "features_dc": new_features_dc,
                    "features_rest": new_features_rest,
                    "opacity": new_opacity,
                    "scales" : new_scales,
                    "rotations" : new_rotations }
        
        self.add_points(params)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        self.prune_points(prune_filter)
    """
    def densify_and_split(self, grads, grad_threshold=0.04, scene_extent=2, N=2):
        n_init_points = self._xyz.shape[0]

        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(torch.exp(self._scales), dim=1).values > self.percent_dense * scene_extent
        )

        # stds = torch.exp(self._scales[selected_pts_mask]).repeat(N, 1)
        # means = torch.zeros((stds.size(0), 3), device=self.device)
        # samples = torch.normal(mean=means, std=stds)
        # rots = build_rotation(self._rotations[selected_pts_mask]).repeat(N, 1, 1)
        
        # new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        # new_scales = torch.log(torch.exp(self._scales[selected_pts_mask]) / (0.8*N)).repeat(N, 1)
        # new_rotations = self._rotations[selected_pts_mask].repeat(N, 1)
        # new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        # new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        # new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)


        # stds_3dmask = torch.exp(self._scales[selected_pts_mask & self.masks_group[0]]).repeat(N, 1)
        # stds_knn = torch.exp(self._scales[selected_pts_mask & self.masks_group[1]]).repeat(N, 1)
        # stds_other = torch.exp(self._scales[selected_pts_mask & self.masks_group[2]]).repeat(N, 1)
        
        ind_3dmask = self.masks_lens_group[0]
        ind_knn = self.masks_lens_group[0]+self.masks_lens_group[1]
        ind_other = self.masks_lens_group[0]+self.masks_lens_group[1]+self.masks_lens_group[2]
        
        stds_3dmask = torch.exp(self._scales[:ind_3dmask][selected_pts_mask[:ind_3dmask]]).repeat(N, 1)
        stds_knn = torch.exp(self._scales[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]]).repeat(N, 1)
        stds_other = torch.exp(self._scales[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]]).repeat(N, 1)
        
        means_3dmask = torch.zeros((stds_3dmask.size(0), 3), device=self.device)
        means_knn = torch.zeros((stds_knn.size(0), 3), device=self.device)
        means_other = torch.zeros((stds_other.size(0), 3), device=self.device)

        samples_3dmask = torch.normal(mean=means_3dmask, std=stds_3dmask)
        samples_knn = torch.normal(mean=means_knn, std=stds_knn)
        samples_other = torch.normal(mean=means_other, std=stds_other)

        rots_3dmask = build_rotation(self._rotations[:ind_3dmask][selected_pts_mask[:ind_3dmask]]).repeat(N, 1, 1)
        rots_knn = build_rotation(self._rotations[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]]).repeat(N, 1, 1)
        rots_other = build_rotation(self._rotations[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]]).repeat(N, 1, 1)
        
        new_xyz_3dmask = torch.bmm(rots_3dmask, samples_3dmask.unsqueeze(-1)).squeeze(-1) + self._xyz[:ind_3dmask][selected_pts_mask[:ind_3dmask]].repeat(N, 1)
        new_xyz_knn = torch.bmm(rots_knn, samples_knn.unsqueeze(-1)).squeeze(-1) + self._xyz[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]].repeat(N, 1)
        new_xyz_other = torch.bmm(rots_other, samples_other.unsqueeze(-1)).squeeze(-1) + self._xyz[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]].repeat(N, 1)

        new_scales_3dmask = torch.log(torch.exp(self._scales[:ind_3dmask][selected_pts_mask[:ind_3dmask]]) / (0.8*N)).repeat(N, 1)
        new_scales_knn = torch.log(torch.exp(self._scales[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]]) / (0.8*N)).repeat(N, 1)
        new_scales_other = torch.log(torch.exp(self._scales[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]]) / (0.8*N)).repeat(N, 1)
        
        new_rotations_3dmask = self._rotations[:ind_3dmask][selected_pts_mask[:ind_3dmask]].repeat(N, 1)
        new_rotations_knn = self._rotations[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]].repeat(N, 1)
        new_rotations_other = self._rotations[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]].repeat(N, 1)
        
        new_features_dc_3dmask = self._features_dc[:ind_3dmask][selected_pts_mask[:ind_3dmask]].repeat(N, 1, 1)
        new_features_dc_knn = self._features_dc[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]].repeat(N, 1, 1)
        new_features_dc_other = self._features_dc[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]].repeat(N, 1, 1)
        
        new_features_rest_3dmask = self._features_rest[:ind_3dmask][selected_pts_mask[:ind_3dmask]].repeat(N, 1, 1)
        new_features_rest_knn = self._features_rest[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]].repeat(N, 1, 1)
        new_features_rest_other = self._features_rest[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]].repeat(N, 1, 1)
        
        new_opacity_3dmask = self._opacity[:ind_3dmask][selected_pts_mask[:ind_3dmask]].repeat(N, 1)
        new_opacity_knn = self._opacity[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]].repeat(N, 1)
        new_opacity_other = self._opacity[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]].repeat(N, 1)

        params = {  "xyz_3dmask": new_xyz_3dmask,
                    "features_dc_3dmask": new_features_dc_3dmask,
                    "features_rest_3dmask": new_features_rest_3dmask,
                    "opacity_3dmask": new_opacity_3dmask,
                    "scales_3dmask" : new_scales_3dmask,
                    "rotations_3dmask" : new_rotations_3dmask,
                    
                    "xyz_knn": new_xyz_knn,
                    "features_dc_knn": new_features_dc_knn,
                    "features_rest_knn": new_features_rest_knn,
                    "opacity_knn": new_opacity_knn,
                    "scales_knn": new_scales_knn,
                    "rotations_knn": new_rotations_knn,
                    
                    "xyz_other": new_xyz_other,
                    "features_dc_other": new_features_dc_other,
                    "features_rest_other": new_features_rest_other,
                    "opacity_other": new_opacity_other,
                    "scales_other": new_scales_other,
                    "rotations_other": new_rotations_other
                    }
        
        self.add_points(params)
        # 这里应该是所有的分别cat
        filter_3dmask = torch.cat((selected_pts_mask[:ind_3dmask], torch.zeros(N * selected_pts_mask[:ind_3dmask].sum(), device=self.device, dtype=bool)))
        filter_knn = torch.cat((selected_pts_mask[ind_3dmask:ind_knn], torch.zeros(N * selected_pts_mask[ind_3dmask:ind_knn].sum(), device=self.device, dtype=bool)))
        filter_other = torch.cat((selected_pts_mask[ind_knn:ind_other], torch.zeros(N * selected_pts_mask[ind_knn:ind_other].sum(), device=self.device, dtype=bool)))
        
        # prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self.device, dtype=bool)))
        # prune_filter = torch.cat((filter_3dmask, filter_knn, filter_other))
        self.prune_points(filter_3dmask, filter_knn, filter_other)
    """
    def densify_and_clone(self, grads, grad_threshold=0.02, scene_extent=2):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(torch.exp(self._scales), dim=1).values <= self.percent_dense * scene_extent
        )
        
        new_xyz = self._xyz[selected_pts_mask]
        #NOTE: dc rest
        #new_features = self._features[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacity = self._opacity[selected_pts_mask]
        new_scales = self._scales[selected_pts_mask]
        new_rotations = self._rotations[selected_pts_mask]
        #NOTE: dc rest
        
        params = {  "xyz": new_xyz,
                    "features_dc": new_features_dc,
                    "features_rest": new_features_rest,
                    "opacity": new_opacity,
                    "scales" : new_scales,
                    "rotations" : new_rotations }

        self.add_points(params)
    """
    def densify_and_clone(self, grads, grad_threshold=0.02, scene_extent=2):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
            torch.max(torch.exp(self._scales), dim=1).values <= self.percent_dense * scene_extent
        )
        
        # self._xyz_3dmask, self._xyz_knn, self._xyz_other
        
        ind_3dmask = self.masks_lens_group[0]
        ind_knn = self.masks_lens_group[0]+self.masks_lens_group[1]
        ind_other = self.masks_lens_group[0]+self.masks_lens_group[1]+self.masks_lens_group[2]
        
        new_xyz_3dmask = self._xyz[:ind_3dmask][selected_pts_mask[:ind_3dmask]]
        new_xyz_knn = self._xyz[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]]
        new_xyz_other = self._xyz[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]]
        
        new_features_dc_3dmask = self._features_dc[:ind_3dmask][selected_pts_mask[:ind_3dmask]]
        new_features_dc_knn = self._features_dc[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]]
        new_features_dc_other = self._features_dc[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]]
        
        new_features_rest_3dmask = self._features_rest[:ind_3dmask][selected_pts_mask[:ind_3dmask]]
        new_features_rest_knn = self._features_rest[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]]
        new_features_rest_other = self._features_rest[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]]
        
        new_opacity_3dmask = self._opacity[:ind_3dmask][selected_pts_mask[:ind_3dmask]]
        new_opacity_knn = self._opacity[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]]
        new_opacity_other = self._opacity[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]]
        
        new_scales_3dmask = self._scales[:ind_3dmask][selected_pts_mask[:ind_3dmask]]
        new_scales_knn = self._scales[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]]
        new_scales_other = self._scales[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]]
        
        new_rotations_3dmask = self._rotations[:ind_3dmask][selected_pts_mask[:ind_3dmask]]
        new_rotations_knn = self._rotations[ind_3dmask:ind_knn][selected_pts_mask[ind_3dmask:ind_knn]]
        new_rotations_other = self._rotations[ind_knn:ind_other][selected_pts_mask[ind_knn:ind_other]]
        
        
        # params = {  "xyz": new_xyz,
        #             "features_dc": new_features_dc,
        #             "features_rest": new_features_rest,
        #             "opacity": new_opacity,
        #             "scales" : new_scales,
        #             "rotations" : new_rotations }
        params = {  "xyz_3dmask": new_xyz_3dmask,
                    "features_dc_3dmask": new_features_dc_3dmask,
                    "features_rest_3dmask": new_features_rest_3dmask,
                    "opacity_3dmask": new_opacity_3dmask,
                    "scales_3dmask" : new_scales_3dmask,
                    "rotations_3dmask" : new_rotations_3dmask ,
                    
                    "xyz_knn": new_xyz_knn,
                    "features_dc_knn": new_features_dc_knn,
                    "features_rest_knn": new_features_rest_knn,
                    "opacity_knn": new_opacity_knn,
                    "scales_knn": new_scales_knn,
                    "rotations_knn": new_rotations_knn,
                    
                    "xyz_other": new_xyz_other,
                    "features_dc_other": new_features_dc_other,
                    "features_rest_other": new_features_rest_other,
                    "opacity_other": new_opacity_other,
                    "scales_other": new_scales_other,
                    "rotations_other": new_rotations_other
                    }

        self.add_points(params)
    
    
    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def reset_opacity(self):
        #opacities_new = torch.min(self._opacity, inverse_sigmoid(torch.ones_like(self._opacity)*0.05))
        opacities_new = self._opacity
        opacities_new[self.get_3dmask()] = torch.min(self._opacity[self.get_3dmask()], inverse_sigmoid(torch.ones_like(self._opacity[self.get_3dmask()])*0.05))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def decrease_opacity(self):
        opacities_new = self._opacity
        opacities_new[self.get_3dmask()] = opacities_new[self.get_3dmask()] - inverse_sigmoid(torch.ones_like(opacities_new[self.get_3dmask()])*0.001)       
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors