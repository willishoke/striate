"""Full-size look at one de-smear: raw spectrum-interp midpoint vs critic-projected,
plus both clean endpoints. Reuses the saved critic + corpus (no retrain)."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch

from striate.param import FourierImage
from striate.transforms import TransformRobustness
from striate.util import pick_device, save_image
from critic import Critic, prep

dev = pick_device()
blob = torch.load("outputs/corpus.pt", weights_only=False)
spectra = blob["spectra"].to(dev)
size = blob["size"]
critic = Critic().to(dev)
critic.load_state_dict(torch.load("outputs/critic.pt"))
critic.eval()
for p in critic.parameters():
    p.requires_grad_(False)
transforms = TransformRobustness(degrees=[])
crit_size = 160

A, B = 7, 15  # corpus indices -> channels 42 and 90
sA, sB = spectra[A : A + 1], spectra[B : B + 1]
interp = 0.5 * sA + 0.5 * sB


def render(spec):
    fp = FourierImage(size, size, device=dev)
    with torch.no_grad():
        fp.spectrum.copy_(spec)
    return fp.to_image().detach()


def score(img, draws=8):
    with torch.no_grad():
        return (
            torch.stack(
                [torch.sigmoid(critic(prep(img, crit_size, transforms))) for _ in range(draws)]
            )
            .mean()
            .item()
        )


save_image(render(sA), "outputs/zoom_A.png")
save_image(render(sB), "outputs/zoom_B.png")
raw = render(interp)
save_image(raw, "outputs/zoom_raw_t50.png")

param = FourierImage(size, size, device=dev)
with torch.no_grad():
    param.spectrum.copy_(interp)
opt = torch.optim.Adam([param.spectrum], lr=0.03)
for _ in range(220):
    opt.zero_grad()
    (5.0 * torch.sigmoid(critic(prep(param.to_image(), crit_size, transforms)))).backward()
    opt.step()
des = param.to_image().detach()
save_image(des, "outputs/zoom_des_t50.png")

print(f"raw midpoint s={score(raw):.3f}   de-smeared s={score(des):.3f}")
# how close is the de-smeared result to each endpoint (image-space L2)?
dA = ((des - render(sA)) ** 2).mean().item()
dB = ((des - render(sB)) ** 2).mean().item()
dRaw = ((des - raw) ** 2).mean().item()
print(f"de-smeared MSE to A={dA:.4f}  to B={dB:.4f}  to raw-midpoint={dRaw:.4f}")
