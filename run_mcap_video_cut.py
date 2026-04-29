import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import sys
from mcap_protobuf.decoder import DecoderFactory
from mcap.reader import make_reader
DEBUG = False

from depth_anything_v2.dpt import DepthAnythingV2

def decodeImage(channel, proto_msg):
    print("image: ", proto_msg.width, "x", proto_msg.height, "type: ", proto_msg.type) 
    # JPEG
    if(proto_msg.type == 10):
        # Convert the bytes into a NumPy uint8 array
        nparr = np.frombuffer(proto_msg.data, np.uint8)

        # Decode the array into an image (OpenCV format)
        img = None
        if proto_msg.channels == 3 or proto_msg.channels == 4:
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif proto_msg.channels == 1:
            img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        else:
            raise TypeError("Unsupported channels: ", proto_msg.channels)

        # debug viz
        # cv2.imshow(channel.topic, img)
        # cv2.waitKey(1)   
    elif(proto_msg.type == 0):
        # Convert the bytes into a NumPy uint8 array
        nparr = np.frombuffer(proto_msg.data, np.uint8)
        # cast array to mat
        img = nparr.reshape((proto_msg.height, proto_msg.width, proto_msg.channels))

        # debug viz
        # cv2.imshow(channel.topic, img)
        # cv2.waitKey(1)
    else:
        raise TypeError("Unsupported image type: ", proto_msg.type)
    return img

def depth_to_normals(depth):
    """Compute surface normal map from a depth map.
    Returns a BGR uint8 image (OpenCV convention) where R=X, G=Y, B=Z."""
    dz_dy, dz_dx = np.gradient(depth)
    normals = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
    norm = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / (norm + 1e-8)
    # Map from [-1, 1] to [0, 255]
    normals_rgb = ((normals + 1) / 2 * 255).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    return normals_rgb[:, :, ::-1]

import numpy as np
import cv2

# def normals_to_edges(normals, clip_percentile=100, gamma=1.2):
#     """
#     Computes edge magnitude from normal maps using Angular Deviation (Dot Product).
    
#     Args:
#         normals: HxWx3 uint8 normal map (BGR).
#         clip_percentile: top percentile for normalization.
#         gamma: exponent for edge boosting.
#     """
#     # 1. Converti da BGR uint8 [0, 255] a vettori float32 [-1, 1]
#     # Assumiamo che la codifica sia: 128 -> 0, 255 -> 1, 0 -> -1
#     n = normals.astype(np.float32)
#     n = (n / 127.5) - 1.0
    
#     # Normalizza i vettori per sicurezza (magnitudo = 1)
#     norm_factor = np.sqrt(np.sum(n**2, axis=-1, keepdims=True)) + 1e-8
#     n /= norm_factor

#     # 2. Calcola il prodotto scalare con i vicini (Shift detection)
#     # Spostiamo l'immagine di 1 pixel in X e Y
#     # n[y, x] \cdot n[y, x+1]
#     dot_x = np.sum(n[:, :-1] * n[:, 1:], axis=-1)
#     # n[y, x] \cdot n[y+1, x]
#     dot_y = np.sum(n[:-1, :] * n[1:, :], axis=-1)

#     # 3. Calcola la deviazione angolare
#     # Il prodotto scalare è 1 se i vettori sono identici, < 1 se divergono.
#     # Usiamo 1 - dot per avere valori alti sui bordi.
#     diff_x = 1.0 - np.clip(dot_x, -1.0, 1.0)
#     diff_y = 1.0 - np.clip(dot_y, -1.0, 1.0)

#     # 4. Ricostruisci la matrice edges (padding per tornare a HxW originale)
#     edges_x = np.zeros(normals.shape[:2], dtype=np.float32)
#     edges_y = np.zeros(normals.shape[:2], dtype=np.float32)
    
#     edges_x[:, :-1] = diff_x
#     edges_y[:-1, :] = diff_y
    
#     # Magnitudo combinata
#     edges = np.sqrt(edges_x**2 + edges_y**2)

#     # 5. Post-processing (come da tua richiesta originale)
#     ceil = np.percentile(edges, clip_percentile)
#     edges = np.clip(edges, 0, ceil) / (ceil + 1e-8)
    
#     # Applichiamo gamma
#     edges = np.power(edges, gamma)

#     return (edges * 255).astype(np.uint8)

def normals_to_edges(normals, clip_percentile=100, gamma=2.0):
    """Computes edge magnitude from normal maps.

    Args:
        normals: HxWx3 uint8 normal map (BGR).
        clip_percentile: top percentile used as the normalization ceiling.
                         Lower values (e.g. 90) make weak edges brighter.
        gamma: exponent applied after normalization. Values < 1 boost faint edges.
               0.5 is a good starting point; lower = more pronounced.
    """
    edges = np.zeros(normals.shape[:2], dtype=np.float32)
    # edges = cv2.bilateralFilter(normals, d=5, sigmaColor=75, sigmaSpace=75)
    for c in range(3):
        gx = cv2.Sobel(normals[:, :, c], cv2.CV_32F, 1, 0, ksize=21)
        gy = cv2.Sobel(normals[:, :, c], cv2.CV_32F, 0, 1, ksize=21)
        edges += gx**2 + gy**2
    edges = np.sqrt(edges)
    # Clip at percentile to prevent bright outliers from compressing the range
    ceil = np.percentile(edges, clip_percentile)
    edges = np.clip(edges, 0, ceil) / (ceil + 1e-8)
    # Gamma < 1 brightens weak edges
    edges = np.power(edges, gamma)
    return (edges * 255).astype(np.uint8)

def depth_to_normals_sobel(depth_map):
        rows, cols = depth_map.shape
        print('rows,cols', rows,cols)

        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        x = x.astype(np.float32)
        y = y.astype(np.float32)

        # Calculate the partial derivatives of depth with respect to x and y
        dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=9)
        dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=9)

        # Compute the normal vector for each pixel
        normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
        norm = np.sqrt(np.sum(normal**2, axis=2, keepdims=True))
        normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)

        # Map the normal vectors to the [0, 255] range and convert to uint8
        normal = (normal + 1) * 127.5
        normal = normal.clip(0, 255).astype(np.uint8)
        # return normal
        # Save the normal map to a file
        normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        return normal_bgr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--videopath', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--normal', dest='normal', action='store_true', help='compute and display normal maps instead of depth')
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    if os.path.isfile(args.videopath):
        if args.videopath.endswith('txt'):
            with open(args.videopath, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.videopath]
    else:
        filenames = glob.glob(os.path.join(args.videopath, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('turbo') #gist_ncar

    img = None
    out = None
    count = 0
    with open(args.videopath, "rb") as f:
        reader = make_reader(f, decoder_factories=[DecoderFactory()])
        for schema, channel, message, proto_msg in reader.iter_decoded_messages():
            print(f"msg {channel.topic} {schema.name} [{message.log_time}]")
            if(schema.name == "proto.tk.msg.Image"):
                print(channel.topic)
                if channel.topic=='/conti11/image':
                    count += 1
                    if count % 10 != 0:
                        continue
                    img = decodeImage(channel, proto_msg)
                    w,h = img.shape[1], img.shape[0]
                    crop_h = 346
                    crop_w = 1019 # h//3, w//3
                    crop_x = 82
                    crop_y= 313
                    # # print('crop img')
                    # mask = cv2.imread('/home/elena/repos/Depth-Anything-V2-for-Robotic-Picking/cam_surround_l.png')
                    img = img[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # mask = mask[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
                    
                    # img = img*mask
                    cv2.imshow('masked_image', img)
                    print("image size: ", img.shape[1], img.shape[0])

                    if out is None:
                        if args.pred_only:
                            output_width = w
                        else:
                            output_width = w * 2 + margin_width
                        suffix = '_normal' if args.normal else '_depth'
                        output_path = os.path.join(args.outdir, args.encoder + suffix  + '.mp4')
                        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (output_width, h))

                    depth = depth_anything.infer_image(img, args.input_size)

                    if args.normal:
                        # cv2.imwrite(f'VIDEO/rgb_{count}.png', img)
                        depth_vis = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                        # depth = depth.astype(np.uint8)
                        # cv2.imwrite(f'VIDEO/depth_{count}.png', depth )
                        vis = depth_to_normals_sobel(depth)
                        img_normals = np.concatenate([img, vis], axis=1)
                        # cv2.imwrite(f'VIDEO/{count}.png', img_normals)

                        cv2.imshow('normals', vis)
                        
                        edges = normals_to_edges(vis)
                        edges_full = np.zeros((h, w), dtype=np.uint8)
                        edges_full[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w] = edges
                        cv2.imshow('edges', edges_full)
                        cv2.waitKey(1)
                    else:
                        crop_h, crop_w = 418, 1106 # h//3, w//3
                        crop_x = 56
                        crop_y= 359
                        depth = depth[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
                        # depth[:crop_h, :] = 10
                        # depth[-crop_h//2:, :] = 10
                        # depth[:, :crop_w//4] = 10
                        # depth[:, -crop_w//4:] = 10
                        # depth = depth[(h-crop_h)//2:(h+crop_h)//2, (w-crop_w)//2:(w+crop_w)//2]
                        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                        depth = depth.astype(np.uint8)
                        cv2.imshow('depth', depth)
                        cv2.waitKey(1)
                        
                        if args.grayscale:
                            vis = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                        else:
                            vis = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

                    if DEBUG:
                        cv2.imshow("output", vis)
                        cv2.waitKey()
                    # if args.pred_only:
                    #     out.write(vis)
                    # else:
                    #     split_region = np.ones((h, margin_width, 3), dtype=np.uint8) * 255
                    #     combined_frame = cv2.hconcat([img, split_region, vis])
                    #     out.write(combined_frame)
    if out is not None:
        out.release()