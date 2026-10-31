"""Contact sheet across many networks: one hi-res still per (layer, channel).

Each still uses the loop-anchor recipe (ring.py): joint octaves from 224 up to SIZE
(all frequencies free, multi-scale views), with the clip hinge and chroma cap scaled by
the channel's activation. One layer per batch. Resumable: finished stills are skipped.

Usage: sheet.py --nets googlenet,vgg16 --per 6 --size 1024 --out DIR
Writes DIR/full/NET/LAYER_CH.png, DIR/thumb/NET/LAYER_CH.jpg and DIR/manifest.json.
"""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
# hard cap on MPS memory: overshooting raises an OOM instead of pushing the Mac into swap
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.45")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.35")
import argparse, json, random, time
import torch
import torch.nn.functional as F
from PIL import Image
from striate.param import FourierImage, seed_spectrum_from_image
from striate.util import pick_device, save_image
from striate.model import Backbone
from zoo import MultiScaleCrop
from octave import LOGIT_RAIL, CHROMA_RATIO, hf_energy

# (layer, display name, channel range to sample). VGG layers are the ReLU outputs, because
# its ReLUs are in-place and would overwrite a hooked conv output. DenseNet blocks pass
# their input channels through, so only the channels each block adds are sampled.
LAYERS = {
    "googlenet": [
        ("inception3a", "3a", None),
        ("inception3b", "3b", None),
        ("inception4b", "4b", None),
        ("inception4d", "4d", None),
        ("inception4e", "4e", None),
        ("inception5b", "5b", None),
    ],
    "inception_v3": [
        ("Mixed_5b", "5b", None),
        ("Mixed_5d", "5d", None),
        ("Mixed_6a", "6a", None),
        ("Mixed_6c", "6c", None),
        ("Mixed_6e", "6e", None),
        ("Mixed_7b", "7b", None),
    ],
    "vgg16": [
        ("features.8", "conv2_2", None),
        ("features.15", "conv3_3", None),
        ("features.18", "conv4_1", None),
        ("features.22", "conv4_3", None),
        ("features.25", "conv5_1", None),
        ("features.29", "conv5_3", None),
    ],
    "resnet50_v1": [
        ("layer1", "layer1", None),
        ("layer2", "layer2", None),
        ("layer3.2", "layer3.2", None),
        ("layer3", "layer3", None),
        ("layer4", "layer4", None),
    ],
    "resnet50_v2": [
        ("layer1", "layer1", None),
        ("layer2", "layer2", None),
        ("layer3.2", "layer3.2", None),
        ("layer3", "layer3", None),
        ("layer4", "layer4", None),
    ],
    "densenet121": [
        ("features.denseblock1", "block1", (64, 256)),
        ("features.denseblock2", "block2", (128, 512)),
        ("features.denseblock3", "block3", (256, 1024)),
        ("features.denseblock4", "block4", (512, 1024)),
    ],
    "efficientnet_b0": [
        ("features.2", "stage2", None),
        ("features.3", "stage3", None),
        ("features.4", "stage4", None),
        ("features.5", "stage5", None),
        ("features.6", "stage6", None),
    ],
    "regnet_y_1_6gf": [
        ("trunk_output.block1", "block1", None),
        ("trunk_output.block2", "block2", None),
        ("trunk_output.block3", "block3", None),
        ("trunk_output.block4", "block4", None),
    ],
    "convnext_tiny": [
        ("features.1", "stage1", None),
        ("features.3", "stage2", None),
        ("features.5", "stage3", None),
        ("features.7", "stage4", None),
    ],
}
TITLES = {
    "googlenet": "GoogLeNet",
    "inception_v3": "Inception v3",
    "vgg16": "VGG16",
    "resnet50_v1": "ResNet-50 V1",
    "resnet50_v2": "ResNet-50 V2",
    "densenet121": "DenseNet-121",
    "efficientnet_b0": "EfficientNet-B0",
    "regnet_y_1_6gf": "RegNetY-1.6GF",
    "convnext_tiny": "ConvNeXt-T",
}


def act_of(net, x, layer, chans):
    a = net(x)[layer]
    return a[torch.arange(len(chans)), torch.tensor(chans)].mean((1, 2))


def render(net, layer, chans, sizes, steps, seed, dev):
    """Joint octaves for one batch of channels of one layer, with both color penalties."""
    torch.manual_seed(seed)
    random.seed(seed)
    img = None
    for size in sizes:
        param = FourierImage(size, size, batch=len(chans), device=dev)
        if img is not None:
            up = F.interpolate(img, size=(size, size), mode="bicubic", align_corners=False).clamp(
                0, 1
            )
            seed_spectrum_from_image(param, up)
        opt = torch.optim.Adam([param.spectrum], lr=0.05)
        view = MultiScaleCrop(224 / size, 512)
        for _ in range(steps):
            opt.zero_grad()
            pre = param.preact()
            x = torch.sigmoid(pre)
            a = act_of(net, view(x), layer, chans)
            hinge = (torch.relu(pre.abs() - LOGIT_RAIL) ** 2).mean((1, 2, 3))
            y = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(
                1, keepdim=True
            )
            ratio = hf_energy(torch.cat([x[:, 2:3] - y, x[:, 0:1] - y], 1)) / hf_energy(y).detach()
            chroma = torch.relu(ratio / CHROMA_RATIO - 1)
            (-a + a.detach().abs() * (4 * hinge + 4 * chroma)).sum().backward()
            opt.step()
        img = param.to_image().detach()
        del param, opt
    return img, a.detach().cpu().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", default=",".join(LAYERS))
    ap.add_argument("--per", type=int, default=6)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=256)
    ap.add_argument("--batch", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--limit-layers", type=int, default=0, help="for timing: only the first N layers per net"
    )
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    dev = pick_device()
    sizes = [s for s in (224, 320, 448, 640, 896) if s < a.size] + [a.size]
    mpath = f"{a.out}/manifest.json"
    manifest = json.load(open(mpath)) if os.path.exists(mpath) else {}
    t0 = time.time()
    for name in a.nets.split(","):
        net = Backbone(name, dev)
        entry = manifest.setdefault(name, {"title": TITLES[name], "layers": []})
        for layer, label, rng in LAYERS[name][: a.limit_layers or None]:
            net.clear_hooks()  # drop the previous layer's hooks
            n = net.channels(layer)
            lo, hi = rng or (0, n)
            pick = sorted(random.Random(f"{name}/{layer}/{a.seed}").sample(range(lo, hi), a.per))
            lent = next((l for l in entry["layers"] if l["layer"] == layer), None)
            if lent is None:
                lent = {"layer": layer, "label": label, "channels": n, "items": []}
                entry["layers"].append(lent)
            done = {it["ch"] for it in lent["items"]}
            todo = [c for c in pick if c not in done]
            for i in range(0, len(todo), a.batch):
                chans = todo[i : i + a.batch]
                net.clear_hooks()
                net.hook(layer)
                tb = time.time()
                img, acts = render(net, layer, chans, sizes, a.steps, a.seed, dev)
                stem = layer.replace(".", "-")
                os.makedirs(f"{a.out}/full/{name}", exist_ok=True)
                os.makedirs(f"{a.out}/thumb/{name}", exist_ok=True)
                for b, c in enumerate(chans):
                    full = f"full/{name}/{stem}_{c}.png"
                    thumb = f"thumb/{name}/{stem}_{c}.jpg"
                    save_image(img[b : b + 1], f"{a.out}/{full}")
                    Image.open(f"{a.out}/{full}").resize((256, 256), Image.LANCZOS).save(
                        f"{a.out}/{thumb}", quality=88
                    )
                    lent["items"].append(
                        {"ch": c, "act": round(acts[b], 4), "full": full, "thumb": thumb}
                    )
                lent["items"].sort(key=lambda it: it["ch"])
                json.dump(manifest, open(mpath + ".tmp", "w"), indent=1)
                os.replace(mpath + ".tmp", mpath)
                print(
                    f"[{time.time() - t0:.0f}s] {name} {label} {chans} {time.time() - tb:.0f}s",
                    flush=True,
                )
                if dev == "mps":
                    torch.mps.empty_cache()
        net.clear_hooks()
        del net
        if dev == "mps":
            torch.mps.empty_cache()
    print("== sheet done", flush=True)


if __name__ == "__main__":
    main()
