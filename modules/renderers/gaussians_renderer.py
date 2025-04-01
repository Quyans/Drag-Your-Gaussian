import os
import math
import numpy as np

import torch
from torch import nn

from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

import torch.nn.functional as F

from modules.utils.graphics_utils import getWorld2View2, getProjectionMatrix, focal2fov, fov2focal,getWorld2View2_tensor,getWorld2View_tensor

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def inverse_softplus(x, beta=1):
    return (torch.exp(beta * x) - 1).log() / beta

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 1 / tanHalfFovX
    P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from 
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (... x 3 x 3)
    Returns:
        q: quaternion of shape (... x 4)
    """
    prefix_shape = M.shape[:-2]
    Ms = M.reshape(-1, 3, 3)

    trs = 1 + Ms[:, 0, 0] + Ms[:, 1, 1] + Ms[:, 2, 2]

    Qs = []

    for i in range(Ms.shape[0]):
        M = Ms[i]
        tr = trs[i]
        if tr > 0:
            r = torch.sqrt(tr) / 2.0
            x = ( M[ 2, 1] - M[ 1, 2] ) / ( 4 * r )
            y = ( M[ 0, 2] - M[ 2, 0] ) / ( 4 * r )
            z = ( M[ 1, 0] - M[ 0, 1] ) / ( 4 * r )
        elif ( M[ 0, 0] > M[ 1, 1]) and (M[ 0, 0] > M[ 2, 2]):
            S = torch.sqrt(1.0 + M[ 0, 0] - M[ 1, 1] - M[ 2, 2]) * 2 # S=4*qx 
            r = (M[ 2, 1] - M[ 1, 2]) / S
            x = 0.25 * S
            y = (M[ 0, 1] + M[ 1, 0]) / S 
            z = (M[ 0, 2] + M[ 2, 0]) / S 
        elif M[ 1, 1] > M[ 2, 2]: 
            S = torch.sqrt(1.0 + M[ 1, 1] - M[ 0, 0] - M[ 2, 2]) * 2 # S=4*qy
            r = (M[ 0, 2] - M[ 2, 0]) / S
            x = (M[ 0, 1] + M[ 1, 0]) / S
            y = 0.25 * S
            z = (M[ 1, 2] + M[ 2, 1]) / S
        else:
            S = torch.sqrt(1.0 + M[ 2, 2] - M[ 0, 0] -  M[ 1, 1]) * 2 # S=4*qz
            r = (M[ 1, 0] - M[ 0, 1]) / S
            x = (M[ 0, 2] + M[ 2, 0]) / S
            y = (M[ 1, 2] + M[ 2, 1]) / S
            z = 0.25 * S
        Q = torch.stack([r, x, y, z], dim=-1)
        Qs += [Q]

    return torch.stack(Qs, dim=0).reshape(*prefix_shape, 4)

def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    From Pytorch3d
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


class Simple_Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, h, w,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", qvec=None,
                 render_res_factor=1
                 ):
        super(Simple_Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.qvec = qvec

        self.render_res_factor = render_res_factor

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.image_width = w
        self.image_height = h

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()

        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
   
    def world_to_pixel(self, p_world):
        p_world_homogeneous = torch.tensor([*p_world, 1.0], device=self.data_device).float()

        p_camera = p_world_homogeneous @ self.world_view_transform
        if not (self.znear < p_camera[2] < self.zfar):
            return None,None

        p_clip = p_camera @ self.projection_matrix

        p_ndc = p_clip[:3] / p_clip[3]

        if not (-1.0 <= p_ndc[0] <= 1.0 and -1.0 <= p_ndc[1] <= 1.0):
            return None,None
        u = (p_ndc[0] + 1.0) / 2.0 * (self.image_width//self.render_res_factor)
        v = (p_ndc[1] + 1.0) / 2.0 * (self.image_height//self.render_res_factor)

        return torch.tensor([u.item(), v.item()]),p_camera[2]
    
    def project(self, points3d):
        # TODO: should be equivalent to full_proj_transform.T
        if isinstance(points3d, list):
            points3d = torch.stack(points3d, dim=0)
        w2c = self.world_view_transform.T
        R = w2c[:3, :3]
        T = w2c[:3, 3]
        points3d_camera = torch.einsum("ij,bj->bi", R, points3d) + T[None, ...]
        xy = points3d_camera[..., :2] / points3d_camera[..., 2:]
        ij = (
            xy
            * torch.tensor(
                [
                    fov2focal(self.FoVx, self.image_width),
                    fov2focal(self.FoVy, self.image_height),
                ],
                dtype=torch.float32,
                device=xy.device,
            )
            + torch.tensor(
                [self.image_width, self.image_height],
                dtype=torch.float32,
                device=xy.device,
            )
            / 2
        ).to(torch.long)
    
        return ij
    
    def unproject(self, points2d, depth):
        origin = self.camera_center
        w2c = self.world_view_transform.T
        R = w2c[:3, :3].T
    
        if isinstance(points2d, (list, tuple)):
            points2d = torch.stack(points2d, dim=0)
    
        points2d[0] *= self.image_width
        points2d[1] *= self.image_height
        points2d = points2d.to(w2c.device)
        points2d = points2d.to(torch.long)
    
        directions = (
            points2d
            - torch.tensor(
                [self.image_width, self.image_height],
                dtype=torch.float32,
                device=w2c.device,
            )
            / 2
        ) / torch.tensor(
            [
                fov2focal(self.FoVx, self.image_width),
                fov2focal(self.FoVy, self.image_height),
            ],
            dtype=torch.float32,
            device=w2c.device,
        )
        padding = torch.ones_like(directions[..., :1])
        directions = torch.cat([directions, padding], dim=-1)
        if directions.ndim == 1:
            directions = directions[None, ...]
        directions = torch.einsum("ij,bj->bi", R, directions)
        directions = F.normalize(directions, dim=-1)
    
        points3d = (
            directions * depth[0][points2d[..., 1], points2d[..., 0]] + origin[None, ...]
        )
    
        return points3d

class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar,image_name, device='cpu', render_res_factor=1):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width * render_res_factor
        self.image_height = height * render_res_factor
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        self.image_name=image_name
        self.device = torch.device(device)
        self.render_res_factor = render_res_factor

        # opengl2colmap
        c2w[:3, 1:3] *= -1

        w2c = np.linalg.inv(c2w)

        self.world_view_transform = torch.tensor(w2c).transpose(0, 1).to(device)
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
            .to(device)
        )
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = -torch.tensor(c2w[:3, 3]).to(device)
        
        self.intrinsics = self.image_height / (2 * math.tan(self.FoVx / 2)), self.image_width / (2 * math.tan(self.FoVy / 2)), self.image_height / 2, self.image_width / 2
    
    def world_to_pixel(self, p_world):
        p_world_homogeneous = torch.tensor([*p_world, 1.0], device=self.device).float()

        p_camera = p_world_homogeneous @ self.world_view_transform
        if not (self.znear < p_camera[2] < self.zfar):
            return None,None

        p_clip = p_camera @ self.projection_matrix

        p_ndc = p_clip[:3] / p_clip[3]

        if not (-1.0 <= p_ndc[0] <= 1.0 and -1.0 <= p_ndc[1] <= 1.0):
            return None,None
        u = (p_ndc[0] + 1.0) / 2.0 * (self.image_width//self.render_res_factor)
        #???理论上是 v = (1.0 - p_ndc[1]) / 2.0 * self.image_height，但改完后不知道为啥对了
        v = (p_ndc[1] + 1.0) / 2.0 * (self.image_height//self.render_res_factor)

        return torch.tensor([u.item(), v.item()]),p_camera[2]
    
class GaussianConverter(nn.Module):
    def __init__(self):
        super().__init__()

        self.gaussian_channels = [3, 2, 1, 1, 3, 4]

        self.register_buffer("opacity_offset", inverse_sigmoid(torch.Tensor([0.01]))[0], persistent=False)
        self.register_buffer("scales_offset", torch.log(torch.Tensor([1/100]))[0], persistent=False)
        self.register_buffer("rotations_offset", torch.Tensor([1., 0, 0, 0]).reshape(1, 4), persistent=False)
        
        self.register_buffer("muls", torch.Tensor([0.01] * 3 + [0.01] * 2 + [0.05] * 1 + [0.05] * 1 + [0.005] * 3 + [0.005] * 4).reshape(1, -1)) 
        self.muls = self.muls / self.muls.max()

    def forward(self, local_gaussian_params, cameras):

        # B x N x H x W x C
        B, N, C, h, w = local_gaussian_params.shape
        local_gaussian_params = local_gaussian_params.permute(0, 1, 3, 4, 2).reshape(-1, sum(self.gaussian_channels))
        local_gaussian_params = local_gaussian_params * self.muls

        features, uv_offset, depth, opacity, scales, rotations = local_gaussian_params.split(self.gaussian_channels, dim=-1)

        cameras = cameras.flatten(0, 1)
        device = cameras.device
        BN = cameras.shape[0]
        c2w = torch.eye(4)[None].to(device).repeat(BN, 1, 1)
        c2w[:, :3, :] = cameras[:, :12].reshape(BN, 3, 4)
        fx, fy, cx, cy, H, W = cameras[:, 12:].chunk(6, -1)

        fx, cx = fx * h / H, cx * h / H
        fy, cy = fy * w / W, cy * w / W

        inds = torch.arange(0, h*w, device=device).expand(BN, h*w)
        
        i = inds % w + 0.5
        j = torch.div(inds, w, rounding_mode='floor') + 0.5

        u = i / cx + uv_offset[..., 0].reshape(BN, h*w)
        v = j / cy + uv_offset[..., 1].reshape(BN, h*w)

        zs = - torch.ones_like(i)
        xs = - (u - 1) * cx / fx * zs
        ys = (v - 1) * cy / fy * zs
        directions = torch.stack((xs, ys, zs), dim=-1)

        # B x N x 3 & B x 3 x 3
        rays_d = F.normalize(directions @ c2w[:, :3, :3].transpose(-1, -2), dim=-1)

        rays_o = c2w[..., :3, 3] # [B, 3]
        rays_o = rays_o[..., None, :].expand_as(rays_d)

        rays_o = rays_o.reshape(BN*h*w, 3)
        rays_d = rays_d.reshape(BN*h*w, 3)

        depth = depth.reshape(BN*h*w, 1) + 1.85
        xyz = (rays_o + depth * rays_d)
        features = features.reshape(BN*h*w, -1, 3) / (2 * 0.28209479177387814)
        opacity = torch.sigmoid(opacity + self.opacity_offset)
        scales = torch.exp(scales + self.scales_offset)
        rotations = torch.nn.functional.normalize(rotations + self.rotations_offset, dim=-1)
    
        return xyz.reshape(B, -1, 3), features.reshape(B, -1, 1, 3), opacity.reshape(B, -1, 1), scales.reshape(B, -1, 3), rotations.reshape(B, -1, 4)

class GaussianRenderer(nn.Module):
    def __init__(self, sh_degree=0, background=[0, 0, 0]):
        super().__init__()
        self.sh_degree = sh_degree
        
        self.register_buffer("bg_color", torch.tensor(background).float())

    def render(
        self,
        xyz,
        features,
        opacity,
        scales,
        rotations,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color='random',
        max_depth=100,
        save_radii_viewspace_points=True
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        viewspace_points = (
            torch.zeros_like(
                xyz,
                dtype=xyz.dtype,
                requires_grad=True,
                device=xyz.device,
            )
            + 0
        )
        try:
            viewspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        if bg_color == 'random':
            bg_color = torch.rand_like(self.bg_color)
        elif bg_color is None:
            bg_color = self.bg_color
        else:
            bg_color = bg_color

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = xyz
        means2D = viewspace_points

        shs = features

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_mask, _ = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
            filter_switch=False,
            filter_rectangle=None,
        )


        rendered_image = rendered_image.clamp(0, 1)
        rendered_mask = rendered_mask.clamp(0, 1)
        rendered_depth = rendered_depth + max_depth * (1 - rendered_mask)
        
        if save_radii_viewspace_points:
            self.radii.append(radii)
            self.viewspace_points.append(viewspace_points)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
   
        return rendered_image, rendered_depth, rendered_mask
        
    def render_gui(
        self,
        xyz,
        features,
        opacity,
        scales,
        rotations,
        edit_mask,
        viewpoint_camera,
        scaling_modifier=1.0,
        bg_color='random',
        max_depth=100,
        blend_alpha=0.0,
        save_radii_viewspace_points=False,
        filter_switch=False,
        filter_rectangle=None,
    ):
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        viewspace_points = (
            torch.zeros_like(
                xyz,
                dtype=xyz.dtype,
                requires_grad=True,
                device=xyz.device,
            )
            + 0
        )
        try:
            viewspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        if bg_color == 'random':
            bg_color = torch.rand_like(self.bg_color)
        elif bg_color is None:
            bg_color = self.bg_color
        else:
            bg_color = bg_color

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = xyz
        means2D = viewspace_points

        shs = features

        if edit_mask is not None:
            if shs is not None and not edit_mask.all():
                shs = shs.clone()
                shs[edit_mask] =(1 - blend_alpha) * shs[edit_mask] +  blend_alpha * torch.tensor([255,0,0], device=shs.device,dtype=shs.dtype).reshape(-1,3)
    

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, rendered_depth, rendered_mask, mask_3d = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
            filter_switch=filter_switch,
            filter_rectangle=filter_rectangle,
        )


        rendered_image = rendered_image.clamp(0, 1)
        rendered_mask = rendered_mask.clamp(0, 1)
        rendered_depth = rendered_depth + max_depth * (1 - rendered_mask)
        
        if save_radii_viewspace_points:
            self.radii.append(radii)
            self.viewspace_points.append(viewspace_points)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
   
        return rendered_image, rendered_depth, rendered_mask, mask_3d

    def convert_camera_parameters_into_viewpoint_cameras(self, cameras, h=None, w=None):
        device = cameras.device
        cameras = cameras.cpu()
        c2w = torch.eye(4)
        c2w[:3, :] = cameras[:12].reshape(3, 4)
        fx, fy, cx, cy, H, W = cameras[12:].chunk(6, -1) # each
        
        if h is not None:
            fx, cx = fx * h / H, cx * h / H
        else:
            h = int(H[0].item())
        if w is not None:
            fy, cy = fy * w / W, cy * w / W
        else:
            w = int(W[0].item())
        
        fovy = 2 * torch.atan(0.5 * w / fy)
        fovx = 2 * torch.atan(0.5 * h / fx)
        cam = MiniCam(c2w.numpy(), w, h, fovy.numpy(), fovx.numpy(), 0.1, 100, device=device)
        return cam

    @torch.cuda.amp.autocast(enabled=False)
    def forward(
        self,
        cameras,
        gaussians,
        scaling_modifier=1.0,
        bg_color=None,
        save_radii_viewspace_points=True,
        down_sample_res=True,
    ):
        B = len(cameras)     
        #xyz, features, opacity, scales, rotations = gaussians
        #gs = xyz[0], features[0], opacity[0], scales[0], rotations[0]
        if save_radii_viewspace_points:
            self.radii = []
            self.viewspace_points = []
            
        images = []
        depths = []
        masks = []
        for i in range(B):
            viewpoint_camera = cameras[i]
            rendered_image, rendered_depth, rendered_mask= self.render(*gaussians, viewpoint_camera, scaling_modifier=scaling_modifier, bg_color=bg_color,
                                                                                     save_radii_viewspace_points=save_radii_viewspace_points)
            
            images.append(rendered_image)
            depths.append(rendered_depth)
            masks.append(rendered_mask)
       
        images = torch.stack(images, dim=0) * 2 - 1
        depths =  torch.stack(depths, dim=0)
        masks = torch.stack(masks, dim=0)

        torch.cuda.synchronize()
        if down_sample_res:
            images = F.interpolate(images, (cameras[0].image_height // cameras[0].render_res_factor, cameras[0].image_width // cameras[0].render_res_factor), mode="bilinear", align_corners=False)
            depths = F.interpolate(depths, (cameras[0].image_height // cameras[0].render_res_factor, cameras[0].image_width // cameras[0].render_res_factor), mode="bilinear", align_corners=False)
            masks = F.interpolate(masks, (cameras[0].image_height // cameras[0].render_res_factor, cameras[0].image_width // cameras[0].render_res_factor), mode="bilinear", align_corners=False)

        return images, depths, masks

    # @torch.cuda.amp.autocast(enabled=False)
    # def forward(
    #     self,
    #     cameras,
    #     gaussians,
    #     scaling_modifier=1.0,
    #     bg_color=None,
    #     h=256,
    #     w=256,
    # ):
    #     B, N = cameras.shape[:2]        
    #     xyz, features, opacity, scales, rotations = gaussians

    #     self.radii = []
    #     self.viewspace_points = []
            
    #     images = []
    #     depths = []
    #     masks = []
    #     for i in range(B):
    #         gs = xyz[i], features[i], opacity[i], scales[i], rotations[i]
    #         for j in range(N):
    #             viewpoint_camera = self.convert_camera_parameters_into_viewpoint_cameras(cameras[i, j], h=h, w=w)

    #             rendered_image, rendered_depth, rendered_mask = self.render(*gs, viewpoint_camera, scaling_modifier=scaling_modifier, bg_color=bg_color)
            
    #             images.append(rendered_image)
    #             depths.append(rendered_depth)
    #             masks.append(rendered_mask)
       
    #     images = torch.stack(images, dim=0).unflatten(0, (B, N)) * 2 - 1
    #     depths =  torch.stack(depths, dim=0).unflatten(0, (B, N))
    #     masks = torch.stack(masks, dim=0).unflatten(0, (B, N))

    #     torch.cuda.synchronize()
        
    #     return images, depths, masks, None, None
