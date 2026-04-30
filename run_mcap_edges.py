"""
Read an MCAP file, compute edge maps from depth+normals for each image message,
and write a new MCAP with the original images + edge maps on /topicX/edge topics.

Usage (with per-topic config file):
    python run_mcap_edges.py input.mcap output.mcap --config crops.yaml

Usage (single topic, manual crop):
    python run_mcap_edges.py input.mcap output.mcap \\
        --topic /conti11/image \\
        --crop-x 82 --crop-y 313 --crop-w 1019 --crop-h 346
"""

import argparse
import copy
import os
import sys

import cv2
import numpy as np
import torch
import yaml
from mcap.reader import make_reader
from mcap.writer import CompressionType, Writer
from mcap_protobuf.decoder import DecoderFactory
from tqdm import tqdm

from depth_anything_v2.dpt import DepthAnythingV2

DEBUG = False


# ── image decoding (mirrors run_mcap_video_cut.py) ────────────────────────────

def decode_image(channel, proto_msg):
    nparr = np.frombuffer(proto_msg.data, np.uint8)
    if proto_msg.type == 10:   # JPEG
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR if proto_msg.channels >= 3 else cv2.IMREAD_GRAYSCALE)
    elif proto_msg.type == 0:  # raw
        img = nparr.reshape((proto_msg.height, proto_msg.width, proto_msg.channels))
    else:
        raise TypeError(f"Unsupported image type: {proto_msg.type}")
    return img


# ── depth → normals → edges pipeline ─────────────────────────────────────────

def depth_to_normals_sobel(depth_map):
    rows, cols = depth_map.shape
    dx = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=9)
    dy = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=9)
    normal = np.dstack((-dx, -dy, np.ones((rows, cols))))
    norm = np.sqrt(np.sum(normal ** 2, axis=2, keepdims=True))
    normal = np.divide(normal, norm, out=np.zeros_like(normal), where=norm != 0)
    normal = (normal + 1) * 127.5
    normal_bgr = cv2.cvtColor(normal.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return normal_bgr


def normals_to_edges(normals, clip_percentile=100, gamma=2.0):
    edges = np.zeros(normals.shape[:2], dtype=np.float32)
    for c in range(3):
        gx = cv2.Sobel(normals[:, :, c], cv2.CV_32F, 1, 0, ksize=21)
        gy = cv2.Sobel(normals[:, :, c], cv2.CV_32F, 0, 1, ksize=21)
        edges += gx ** 2 + gy ** 2
    edges = np.sqrt(edges)
    ceil = np.percentile(edges, clip_percentile)
    edges = np.clip(edges, 0, ceil) / (ceil + 1e-8)
    edges = np.power(edges, gamma)
    return (edges * 255).astype(np.uint8)


def compute_edges_full(model, img_bgr, input_size, crop_x, crop_y, crop_w, crop_h):
    """
    Crop → depth → normals → edges, then paste into a zero canvas at original resolution.
    Returns a uint8 grayscale image of shape (H_orig, W_orig).
    """
    H_orig, W_orig = img_bgr.shape[:2]

    img_crop = img_bgr[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    img_rgb  = cv2.cvtColor(img_crop, cv2.COLOR_BGR2RGB)

    depth   = model.infer_image(img_rgb, input_size)
    normals = depth_to_normals_sobel(depth)
    edges   = normals_to_edges(normals)

    edges_full = np.zeros((H_orig, W_orig), dtype=np.uint8)
    edges_full[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w] = edges
    return edges_full


# ── MCAP I/O ──────────────────────────────────────────────────────────────────

def edge_topic(original_topic: str) -> str:
    """Map /contiX/image -> /contiX/edge"""
    parts = [p for p in original_topic.strip("/").split("/") if p != "image"]
    return "/" + "/".join(parts) + "/edge"


def main():
    parser = argparse.ArgumentParser(description="Replace RGB with edge maps in an MCAP file")
    parser.add_argument("input",  help="Input .mcap file")
    parser.add_argument("output", help="Output .mcap file")

    parser.add_argument("--encoder", default="vitl",
                        choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument("--load-from", default=None,
                        help="Checkpoint path (default: checkpoints/depth_anything_v2_<encoder>.pth)")
    parser.add_argument("--input-size", type=int, default=518)

    parser.add_argument("--config", default=None,
                        help="YAML config file with per-topic crop settings (e.g. crops.yaml). "
                             "Topics are taken from the config; --topic and --crop-* are ignored.")

    parser.add_argument("--topic", nargs="+", default=None,
                        help="One or more image topics (space- or comma-separated). "
                             "Ignored when --config is provided.")
    parser.add_argument("--crop-x", type=int, default=82)
    parser.add_argument("--crop-y", type=int, default=313)
    parser.add_argument("--crop-w", type=int, default=1019)
    parser.add_argument("--crop-h", type=int, default=346)

    parser.add_argument("--colormap", action="store_true",
                        help="Apply MAGMA colormap to edges (3-ch); default: replicate grayscale")
    parser.add_argument("--every-n", type=int, default=1,
                        help="Run inference on 1 frame every N (default: 1 = all frames)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Stop after this many inferred frames (per topic)")
    args = parser.parse_args()

    # ── Load config or fall back to CLI args ───────────────────────────────
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        topic_crops: dict[str, dict] = cfg["topics"]
        print(f"Config loaded from {args.config}  ({len(topic_crops)} topics)")
    else:
        if args.topic is None:
            parser.error("Provide --config <file> or at least one --topic.")
        topics = [t for raw in args.topic for t in raw.split(",") if t]
        topic_crops = {
            t: {"crop_x": args.crop_x, "crop_y": args.crop_y,
                "crop_w": args.crop_w, "crop_h": args.crop_h}
            for t in topics
        }

    for topic, crop in topic_crops.items():
        print(f"  {topic}: x={crop['crop_x']} y={crop['crop_y']} "
              f"w={crop['crop_w']} h={crop['crop_h']}")

    # ── Model ─────────────────────────────────────────────────────────────
    DEVICE = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available()
              else "cpu")

    model_configs = {
        "vits": {"encoder": "vits", "features": 64,  "out_channels": [48,  96,  192,  384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96,  192, 384,  768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536,1536,1536,1536]},
    }

    model = DepthAnythingV2(**model_configs[args.encoder])
    ckpt  = args.load_from or f"checkpoints/depth_anything_v2_{args.encoder}.pth"
    try:
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False))
    except Exception:
        sd    = torch.load(ckpt, map_location="cpu", weights_only=False)
        clean = {k.replace("module.", ""): v for k, v in sd["model"].items()}
        model.load_state_dict(clean)
    model = model.to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}  ({ckpt})")
    print(f"Number of max frames: {args.max_frames}")

    # ── MCAP pass ─────────────────────────────────────────────────────────
    with open(args.input, "rb") as f_in, open(args.output, "wb") as f_out:
        reader = make_reader(f_in, decoder_factories=[DecoderFactory()])

        header  = reader.get_header()
        profile = header.profile if header else ""
        library = header.library if header else "run_mcap_edges"

        writer = Writer(f_out, compression=CompressionType.ZSTD)
        writer.start(profile=profile, library=library)

        for meta in reader.iter_metadata():
            writer.add_metadata(name=meta.name, data=meta.metadata)

        summary    = reader.get_summary()
        total_msgs = (summary.statistics.message_count
                      if summary and summary.statistics else None)

        if summary and summary.channels:
            print("Topics in MCAP:")
            for ch in summary.channels.values():
                marker = "✓" if ch.topic in topic_crops else " "
                print(f"  [{marker}] {ch.topic}")
            print()

        schema_map:      dict[int, int] = {}
        channel_map:     dict[int, int] = {}
        edge_channel_map: dict[int, int] = {}  # original channel id -> edge channel id
        frame_count:     dict[str, int] = {}
        infer_count:     dict[str, int] = {}

        progress = tqdm(reader.iter_decoded_messages(),
                        total=total_msgs, unit="msg", dynamic_ncols=True)

        for schema, channel, message, proto_msg in progress:

            # ── Register schema once ──────────────────────────────────────
            if schema.id not in schema_map:
                schema_map[schema.id] = writer.register_schema(
                    name=schema.name,
                    encoding=schema.encoding,
                    data=schema.data,
                )

            new_schema_id = schema_map[schema.id]

            # ── Register channel once ─────────────────────────────────────
            is_target = (schema.name == "proto.tk.msg.Image"
                         and channel.topic in topic_crops)

            if not is_target:
                continue

            if channel.id not in channel_map:
                channel_map[channel.id] = writer.register_channel(
                    topic=channel.topic,
                    message_encoding=channel.message_encoding,
                    schema_id=new_schema_id,
                    metadata=channel.metadata,
                )
                etopic = edge_topic(channel.topic)
                edge_channel_map[channel.id] = writer.register_channel(
                    topic=etopic,
                    message_encoding=channel.message_encoding,
                    schema_id=new_schema_id,
                    metadata=channel.metadata,
                )
                print(f"  {channel.topic}  ->  {etopic}")

            new_channel_id = channel_map[channel.id]

            # ── Copy original verbatim ────────────────────────────────────
            writer.add_message(
                channel_id=new_channel_id,
                log_time=message.log_time,
                publish_time=message.publish_time,
                data=message.data,
            )

            # ── Inference: apply --every-n and --max-frames ───────────────
            progress.set_postfix_str(channel.topic, refresh=False)
            topic = channel.topic
            frame_count[topic] = frame_count.get(topic, 0) + 1
            infer_count[topic] = infer_count.get(topic, 0)

            skip = (frame_count[topic] - 1) % args.every_n != 0
            if not skip and args.max_frames is not None:
                skip = infer_count[topic] >= args.max_frames

            if skip:
                continue

            infer_count[topic] += 1
            img_bgr    = decode_image(channel, proto_msg)
            crop       = topic_crops[topic]
            edges_gray = compute_edges_full(
                model, img_bgr, args.input_size,
                crop["crop_x"], crop["crop_y"], crop["crop_w"], crop["crop_h"],
            )

            if args.colormap:
                edge_out = cv2.applyColorMap(edges_gray, cv2.COLORMAP_MAGMA)
            else:
                edge_out = np.stack([edges_gray] * 3, axis=-1)

            if DEBUG:
                cv2.imshow("original", img_bgr)
                cv2.imshow("edges",    edge_out)
                cv2.waitKey(1)

            if proto_msg.type == 10:
                _, buf     = cv2.imencode(".jpg", edge_out)
                edge_bytes = bytes(buf)
            else:
                edge_bytes = edge_out.tobytes()

            edge_proto          = copy.deepcopy(proto_msg)
            edge_proto.width    = img_bgr.shape[1]
            edge_proto.height   = img_bgr.shape[0]
            edge_proto.channels = 3
            edge_proto.type     = proto_msg.type
            edge_proto.data     = edge_bytes

            writer.add_message(
                channel_id=edge_channel_map[channel.id],
                log_time=message.log_time,
                publish_time=message.publish_time,
                data=edge_proto.SerializeToString(),
            )

            # Stop early only after the edge has been written
            if args.max_frames is not None:
                if all(infer_count.get(t, 0) >= args.max_frames for t in topic_crops):
                    progress.close()
                    break

        writer.finish()

        if not channel_map:
            print("\n[WARNING] Nessun messaggio scritto. I topic specificati con --topic "
                  "non corrispondono a nessun topic nell'MCAP. "
                  "Controlla l'elenco sopra e rilancia con --topic corretto.")
            return

    print(f"\nDone. Written to: {args.output}")
    verify_mcap(args.output)


def verify_mcap(path: str):
    print(f"\n── Verifying {path} ──")
    with open(path, "rb") as f:
        reader  = make_reader(f)
        header  = reader.get_header()
        summary = reader.get_summary()
        print(f"  profile : {header.profile if header else 'n/a'}")
        print(f"  library : {header.library if header else 'n/a'}")
        if summary and summary.statistics:
            s = summary.statistics
            print(f"  messages: {s.message_count}")
            if s.message_count == 0:
                print("  [WARNING] 0 messages — check your --topic / --config")
                return
            print(f"  channels: {len(summary.channels)}")
        msg_counts: dict[str, int] = {}
        for _, channel, _ in reader.iter_messages():
            msg_counts[channel.topic] = msg_counts.get(channel.topic, 0) + 1
        print("  topics:")
        for topic, count in sorted(msg_counts.items()):
            print(f"    {topic:40s}  {count:>6} msgs")
    print("── OK ──\n")


if __name__ == "__main__":
    main()
