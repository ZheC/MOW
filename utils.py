#!/usr/bin/env python3

"""Utility functions."""

import cv2
import math
import matplotlib.pyplot as plt
import neural_renderer as nr
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image


def compute_R_init(B):
    """Computes B random rotation matrices (source: Zhang et al. 2020)."""
    x1, x2, x3 = torch.split(torch.rand(3 * B).cuda(), B)
    tau = 2 * math.pi
    R = torch.stack(
        (  # B x 3 x 3
            torch.stack(
                (torch.cos(tau * x1), torch.sin(tau * x1), torch.zeros_like(x1)), 1
            ),
            torch.stack(
                (-torch.sin(tau * x1), torch.cos(tau * x1), torch.zeros_like(x1)), 1
            ),
            torch.stack(
                (torch.zeros_like(x1), torch.zeros_like(x1), torch.ones_like(x1)), 1
            ),
        ),
        1,
    )
    v = torch.stack(
        (  # B x 3
            torch.cos(tau * x2) * torch.sqrt(x3),
            torch.sin(tau * x2) * torch.sqrt(x3),
            torch.sqrt(1 - x3),
        ),
        1,
    )
    identity = torch.eye(3).repeat(B, 1, 1).cuda()
    H = identity - 2 * v.unsqueeze(2) * v.unsqueeze(1)
    rotation_matrices = -torch.matmul(H, R)
    return rotation_matrices


def compute_bbox_proj(verts, f, img_size=256):
    """Computes bbox of projected verts (source: Zhang et al. 2020)."""
    xy = verts[:, :, :2]
    z = verts[:, :, 2:]
    proj = f * xy / z + 0.5  # [0, 1]
    proj = proj * img_size  # [0, img_size]
    u, v = proj[:, :, 0], proj[:, :, 1]
    x1, x2 = u.min(1).values, u.max(1).values
    y1, y2 = v.min(1).values, v.max(1).values
    return torch.stack((x1, y1, x2 - x1, y2 - y1), 1)


def compute_t_init(bbox_target, vertices, f=1, img_size=256):
    """Computes initial translation (source: Zhang et al. 2020)."""
    bbox_mask = np.array(bbox_target)
    mask_center = bbox_mask[:2] + bbox_mask[2:] / 2
    diag_mask = np.sqrt(bbox_mask[2] ** 2 + bbox_mask[3] ** 2)
    B = vertices.shape[0]
    x = torch.zeros(B).cuda()
    y = torch.zeros(B).cuda()
    z = 2.5 * torch.ones(B).cuda()
    for _ in range(50):
        translation = torch.stack((x, y, z), -1).unsqueeze(1)
        v = vertices + translation
        bbox_proj = compute_bbox_proj(v, f=1, img_size=img_size)
        diag_proj = torch.sqrt(torch.sum(bbox_proj[:, 2:] ** 2, 1))
        delta_z = z * (diag_proj / diag_mask - 1)
        z = z + delta_z
        proj_center = bbox_proj[:, :2] + bbox_proj[:, 2:] / 2
        x += (mask_center[0] - proj_center[:, 0]) * z / f / img_size
        y += (mask_center[1] - proj_center[:, 1]) * z / f / img_size
    return torch.stack((x, y, z), -1).unsqueeze(1)


def matrix_to_rot6d(rotmat):
    """Converts rot 3x3 mat to 6D (source: Zhang et al. 2020)."""
    return rotmat.view(-1, 3, 3)[:, :, :2]


def rot6d_to_matrix(rot_6d):
    """Converts 6D rot to 3x3 mat (source: Zhang et al. 2020)."""
    rot_6d = rot_6d.view(-1, 3, 2)
    a1 = rot_6d[:, :, 0]
    a2 = rot_6d[:, :, 1]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum("bi,bi->b", b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2)
    return torch.stack((b1, b2, b3), dim=-1)



def center_vertices(vertices, faces, flip_y=True):
    """Centroid-align vertices."""
    vertices = vertices - vertices.mean(dim=0, keepdim=True)
    if flip_y:
        vertices[:, 1] *= -1
        faces = faces[:, [2, 1, 0]]
    return vertices, faces




class PerspectiveRenderer(object):
    """Perspective renderer (source: Zhang et al. 2020)."""

    colors = {
        "blue": [0.65098039, 0.74117647, 0.85882353],
        "red": [251 / 255.0, 128 / 255.0, 114 / 255.0],
    }

    class Model(object):
        def __init__(
            self,
            renderer,
            vertices,
            faces,
            textures=None,
            translation=None,
            rotation=None,
            scale=None,
            color_name="white",
        ):
            if vertices.ndimension() == 2:
                vertices = vertices.unsqueeze(0)
            if faces.ndimension() == 2:
                faces = faces.unsqueeze(0)
            if textures is None:
                textures = torch.ones(
                    len(faces),
                    faces.shape[1],
                    renderer.t_size,
                    renderer.t_size,
                    renderer.t_size,
                    3,
                    dtype=torch.float32,
                ).cuda()
                color = torch.FloatTensor(renderer.colors[color_name]).cuda()
                textures = color * textures
            elif textures.ndimension() == 5:
                textures = textures.unsqueeze(0)

            if translation is None:
                translation = renderer.default_translation
            if not isinstance(translation, torch.Tensor):
                translation = torch.FloatTensor(translation).to(vertices.device)
            if translation.ndimension() == 1:
                translation = translation.unsqueeze(0)

            if rotation is not None:
                vertices = torch.matmul(vertices, rotation)
            vertices += translation

            if scale is not None:
                vertices *= scale
            
            self.vertices = vertices
            self.faces = faces
            self.textures = textures
        
        def join(self, model):
            self.faces = torch.cat((self.faces, model.faces + self.vertices.shape[1]), 1)
            self.vertices = torch.cat((self.vertices, model.vertices), 1)
            self.textures = torch.cat((self.textures, model.textures), 1)

    def __init__(self, image_size=256, texture_size=1):
        self.image_size = image_size
        self.default_K = torch.cuda.FloatTensor([[[1, 0, 0.5], [0, 1, 0.5], [0, 0, 1]]])
        self.R = torch.cuda.FloatTensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
        self.t = torch.zeros(1, 3).cuda()
        self.default_translation = torch.cuda.FloatTensor([[0, 0, 2]])
        self.t_size = texture_size
        self.renderer = nr.renderer.Renderer(
            image_size=image_size, K=self.default_K, R=self.R, t=self.t, orig_size=1
        )
        self.set_light_dir([1, 1, 0.4], int_dir=0.3, int_amb=0.7)
        self.set_bgcolor([0, 0, 0])

    def __call__(
        self,
        models,
        image=None,
        K=None,
    ):
        if K is not None:
            self.renderer.K = K

        all_models = None
        for model in models:
            if all_models is None:
                all_models = model
            else:
                all_models.join(model)

        rend, depth, sil = self.renderer.render(all_models.vertices, all_models.faces, all_models.textures)
        rend = rend.detach().cpu().numpy().transpose(0, 2, 3, 1)  # B x H x W x C
        rend = np.clip(rend, 0, 1)[0]

        self.renderer.K = self.default_K  # Restore just in case.
        if image is None:
            return rend
        else:
            sil = sil.detach().cpu().numpy()[0]
            h, w, *_ = image.shape
            L = max(h, w)
            if image.max() > 1:
                image = image.astype(float) / 255.0
            new_image = np.pad(image, ((0, L - h), (0, L - w), (0, 0)))
            new_image = cv2.resize(new_image, (self.image_size, self.image_size))
            new_image[sil > 0] = rend[sil > 0]
            r = self.image_size / L
            new_image = new_image[: int(h * r), : int(w * r)]
            return new_image

    def set_light_dir(self, direction=(1, 0.5, -1), int_dir=0.3, int_amb=0.7):
        self.renderer.light_direction = direction
        self.renderer.light_intensity_directional = int_dir
        self.renderer.light_intensity_ambient = int_amb

    def set_bgcolor(self, color):
        self.renderer.background_color = color


def vis_obj_pose_im(verts, faces, rot, t, scale, im, out_f, idx=0):
    im_size = max(im.shape[:2])
    renderer = PerspectiveRenderer(im_size)
    renderer.set_light_dir([1, 0.5, 1], 0.3, 0.5)
    renderer.set_bgcolor([1, 1, 1])
    im_vis = renderer(
        vertices=verts,
        faces=faces,
        image=im,
        translation=t[idx],
        rotation=rot[idx],
        scale=scale,
        color_name="red"
    )
    im_vis = im_vis[:, :, ::-1] * 255
    cv2.imwrite(out_f, im_vis)
    print("Wrote vis obj pose im to: {}".format(out_f))
    return im_vis