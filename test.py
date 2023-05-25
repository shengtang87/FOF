import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import time

# from lib.dataset.train_dataset_fof import get_dataloader
from lib.network.HRNet import HRNetV2_W32 as network
from lib.utils.utils import toDevice
from lib.utils.utils import Recon


def test(args):
    torch.backends.cudnn.benchmark = True

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    e = Recon(device)

    net = network().to(device)
    if args.ckpt == "latest":
        ckpt_list = sorted(os.listdir(os.path.join("ckpt", args.name)))
        if len(ckpt_list) > 0:
            ckpt_path = os.path.join("ckpt", args.name, ckpt_list[-1])
            print('Resuming from', ckpt_path)
            state_dict = torch.load(ckpt_path)
            net.load_state_dict(state_dict["net"])
            del state_dict
        else:
            print("No checkpoint found!")
            exit()
    else:
        print('Resuming from', args.ckpt)
        state_dict = torch.load(args.ckpt)
        net.load_state_dict(state_dict["net"])
        del state_dict

    # Test
    input_dir = sorted(os.listdir(args.input))

    net.eval()
    with torch.no_grad():
        for d in input_dir:
            start_time = time.time()
            img = cv2.imread(os.path.join(args.input, d), -1)
            print(d)
            mask = img[:, :, 3:4]
            img = img[:, :, :3]
            img = torch.from_numpy(img.transpose((2, 0, 1)))[None]
            mask = torch.from_numpy(mask.transpose((2, 0, 1)))[None]
            img = img.to(device)
            mask = mask.to(device) > 127
            ceof = net((img / 127.5 - 1) * mask) * mask
            v, f = e.decode(ceof[0])

            # Load original image
            original_img = cv2.imread(os.path.join(args.input, d), -1)
            original_img_b = cv2.imread(os.path.join(args.input, 'young_back.png'), -1)

            # Write colored vertices to OBJ file
            with open(os.path.join(args.output, d.replace(".png", ".obj")), "w") as mf:
                # Write material file name
                mf.write("mtllib colors.mtl\n")
                # Use default material
                mf.write("usemtl material\n")

                # Write vertex and color information
                
                for vertex in v:                    
                    x, y, z = vertex
                    # Convert 3D model vertex coordinates to pixel coordinates of the original image
                    px = int((x + 1) * 0.5 * (512 - 1))
                    py = int((y + 1) * 0.5 * (512 - 1))
                    # Get color value at the corresponding pixel position
                    if (z>=0): color = original_img[-py, px]
                    else : color = original_img_b[-py, px]

                    r, g, b = color[0], color[1], color[2]

                    mf.write("v %f %f %f %f %f %f\n" % (x, y, z, r / 255.0, g / 255.0, b / 255.0))

                # Write face information
                for face in f:
                    v1, v2, v3 = face
                    # Note: Vertex indices in OBJ file start from 1, so add 1
                    mf.write("f %d %d %d\n" % (v1 + 1, v2 + 1, v3 + 1))

            # Write material file for colors
            with open(os.path.join(args.output, "colors.mtl"), "w") as mtl_file:
                mtl_file.write("newmtl material\n")  # Define a material named "material"
                mtl_file.write("Ka 0.2 0.2 0.2\n")  # Ambient reflectivity
                mtl_file.write("Kd 1.0 1.0 1.0\n")  # Diffuse reflectivity
                mtl_file.write("Ks 0.0 0.0 0.0\n")  # Specular reflectivity
                mtl_file.write("Ns 0.0\n")  # Specular exponent
                mtl_file.write("illum 2\n")  # Illumination model

                # for i, vertex in enumerate(colored_v):
                #     r, g, b = vertex[3:]
                #     mtl_file.write("newmtl material%d\n" % i)
                #     mtl_file.write("Ka 0.2 0.2 0.2\n")
                #     mtl_file.write("Kd %f %f %f\n" % (r / 255.0, g / 255.0, b / 255.0))
                #     mtl_file.write("Ks 0.0 0.0 0.0\n")
                #     mtl_file.write("Ns 0.0\n")
                #     mtl_file.write("illum 2\n")

            end_time = time.time()
            print(f"Iteration took {end_time - start_time:.4f} seconds")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="base", help="name of the experiment")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id for cuda")
    parser.add_argument("--ckpt", type=str, default="latest", help="path of the checkpoint")
    parser.add_argument("--input", type=str, default="input", help="path of the input_dir")
    parser.add_argument("--output", type=str, default="output", help="path of the output_dir")
    args = parser.parse_args()
    test(args)