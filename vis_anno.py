import argparse
import cv2
import neural_renderer as nr
import os
import random
import numpy as np 
import shutil
import torch
import json
import mano
from utils import PerspectiveRenderer


with open('poses.json', 'r') as f:
    annos = json.load(f)

anno = annos[0]

obj_path = 'example/images/{}.obj'.format(anno['image_id'])
im_path = 'example/models/{}.jpg'.format(anno['image_id'])

print("Loading image...")
im = cv2.imread(im_path)[:, :, ::-1]
im_size = max(im.shape[:2])
renderer = PerspectiveRenderer(im_size)
renderer.set_light_dir([1, 0.5, 1], 0.3, 0.5)
renderer.set_bgcolor([1, 1, 1])


print("Loading object...")
obj_verts, obj_faces = nr.load_obj(obj_path)
verts, faces = center_vertices(obj_verts, obj_faces)

model = PerspectiveRenderer.Model(
    renderer,
    verts,
    faces,
    translation=torch.tensor(anno['t']).reshape((1, 3)).to('cuda:0'),
    rotation=torch.tensor(anno['R']).reshape((3, 3)).to('cuda:0'),
    scale=anno['s'],
    color_name='red'
)

print("Loading hand...")
rhm_path = 'mano/MANO_RIGHT.pkl'
rh_model = mano.load(
    model_path=rhm_path,
    model_type='mano',
    num_pca_comps=45,
    batch_size=1,
    flat_hand_mean=False
)

mano_pose = torch.tensor(anno['hand_pose']).unsqueeze(0)
transl = torch.tensor(anno['trans']).unsqueeze(0)
pose, global_orient = mano_pose[:, 3:], mano_pose[:, :3]

output = rh_model(
    global_orient=global_orient,
    hand_pose=pose,
    transl=transl,
    return_verts=True,
    return_tips=True
)

hand = PerspectiveRenderer.Model(
    renderer,
    output.vertices.detach().squeeze(0).to('cuda:0'),
    torch.tensor(rh_model.faces.astype('int32')).to('cuda:0'),
    translation=torch.tensor(anno['hand_t']).reshape((1, 3)).to('cuda:0'),
    rotation=torch.tensor(anno['hand_R']).reshape((3, 3)).type(torch.float32).to('cuda:0'),
    scale=anno['hand_s'],
    color_name='blue'
)

print("Rendering...")
im_vis = renderer(
    [model, hand],
    image=im
)
plt.imshow(im_vis)
plt.show()