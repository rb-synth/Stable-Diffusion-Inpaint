import argparse
import os
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
import cv2
import numpy as np
import torch
from main_inpainting import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from inpaint_utils import seed_everything
from inpaint_utils import make_batch
from contextlib import suppress


def load_model(device, yaml_profile, ckpt):
    config = OmegaConf.load(yaml_profile)
    model = instantiate_from_config(config.model)
    model.load_state_dict(torch.load(ckpt)["state_dict"])
    print(f"Loading modeling from {ckpt}")
    model = model.to(device)
    return model


def image_uint8(image):
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    assert image.min() >=0 and image.max() <= 1, "Image must be in range [0,1]."
    # rescale to [0,255] and convert to uint8
    image = (image * 255).astype(np.uint8)
    # move channel to last dimension
    return np.moveaxis(image, 0, -1)


def decode_and_inpaint(model, samples_ddim, image, mask):
    """Decode samples from latent space and inpaint masked pixels. Return as uint8 in range [0,255] with shape (H,W,C)."""
    # decode from latent space
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    # rescale to range [0,1] (from [-1,1])
    predicted_image = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
    # inpaint (only replace masked pixels)
    inpainted = (1 - mask) * image + mask * predicted_image
    return image_uint8(inpainted[0])


def save_image(inpainted, outpath):
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    Image.fromarray(inpainted).save(outpath)


def save_video(images, outpath, duration=5):
    """Save images as video. Duration is in seconds."""
    print(f"Saving video to {outpath}")
    height, width, _ = images[0].shape
    fps = len(images) / duration
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    video = cv2.VideoWriter(outpath, fourcc, fps, (width, height))
    for image in images:
        video.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.destroyAllWindows()
    video.release()


def save_results(inpainted, intermediates, image, fname, ema, outdir, steps, save_intermediates):
    outpath = outdir if outdir is not None else os.path.join(os.path.dirname(fname), "output")
    outpath = os.path.join(outpath, os.path.split(fname)[1].split(".")[0], f"ema-{str(ema)}_Steps-{steps}")
    print(f"Saving results in {outpath}")
    os.makedirs(outpath, exist_ok=True)

    save_image(inpainted, outpath=os.path.join(outpath, "final.jpg"))

    if save_intermediates:
        # save intermediate results
        for t, i in tqdm(intermediates.items()):
            save_image(i, outpath=os.path.join(outpath, "intermediates", f"step_{t:03d}.jpg"))
        # prepend 10% frames of input image
        nframes = min(1, len(intermediates) // 10)
        intermediates = [image] * nframes + list(intermediates.values())
        save_video(intermediates, outpath=os.path.join(outpath, "intermediates.avi"))


def infer_image(model, sampler: DDIMSampler, device, image, mask, resize, steps):
    batch = make_batch(image, mask, device=device, resize_to=resize)
    c_masked = model.cond_stage_model.encode(batch["masked_image"])

    cc_mask = torch.nn.functional.interpolate(batch["mask"], size=c_masked.shape[-2:])

    c = torch.cat((c_masked,cc_mask), dim=1)

    shape = (3,) + c_masked.shape[2:]

    samples_ddim, intermediates, time_range = sampler.sample(
        S=steps,
        conditioning=c,
        batch_size=c.shape[0],
        shape=shape,
        verbose=False,
    )

    image = torch.clamp((batch["image"] + 1.0) / 2.0, min=0.0, max=1.0)
    mask = torch.clamp((batch["mask"] + 1.0) / 2.0, min=0.0, max=1.0)

    inpainted = decode_and_inpaint(model, samples_ddim, image, mask)
    intermediates = {t: decode_and_inpaint(model, i, image, mask) for i, t in zip(intermediates["x_inter"], time_range)}

    return inpainted, intermediates, image


def main(device, yaml_profile, ckpt, image, mask, resize, steps, no_intermediates=False, seed=42, ema=False, outdir=None):
    save_intermediates = not no_intermediates
    fname = image
    seed_everything(seed)
    device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(device, yaml_profile, ckpt)
    sampler = DDIMSampler(model)
    scope = model.ema_scope if ema else suppress

    with torch.no_grad():
        with scope("Sampling"):
            inpainted, intermediates, image = infer_image(model, sampler, device, image, mask, resize, steps)
            save_results(inpainted, intermediates, image_uint8(image[0]), fname, ema, outdir, steps, save_intermediates)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Stable Diffusion Inpainting")
    parser.add_argument(
        "--image",
        type=str,
        help="Input image",
        required=True
    )
    parser.add_argument(
        "--mask",
        type=str,
        help="Input mask",
        required=True
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="Directory for writing results. Defaults to \"output\" in same directory as input image.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        help="path of weights to load",
        required=True
    )
    parser.add_argument(
        "--yaml_profile",
        type=str,
        help="yaml file describing the model to initialize",
        required=True
    )
    parser.add_argument(
        "--resize",
        type=int,
        default=512,
        help="Resize images to this size",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for inference (cpu, cuda, cuda:x)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--ema",
        action='store_true',
        help="use ema weights. Not recommended if trained on only a few images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed",
    )
    parser.add_argument(
        "--no_intermediates",
        action='store_true',
        help="do not save intermediate results",
    )
    opt = parser.parse_args()
    main(**vars(opt))
