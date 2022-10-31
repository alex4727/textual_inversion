import argparse
import torch
import os
from omegaconf import OmegaConf
from tqdm import tqdm
import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target_images",
        type=int,
        default=1300,
        help="Number of images to generate per class",
    )
    parser.add_argument(
        "--start_idx",
        type=int,
        default=0,
        help="start index for imagenet (inclusive)"
    )
    parser.add_argument(
        "--end_idx",
        type=int,
        default=999,
        help="end index for imagenet (inclusive)"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/cin256-v2/model.ckpt",
        help="path to checkpoint"
    )
    parser.add_argument(
        "--dataset_out_path",
        type=str,
        default="./dataset_out",
        help="path to output dataset"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32
    )
    return parser.parse_args()

def prepare_imagenet_stats(opt):
    print("Preparing imagenet stats")
    imagenet_stats = dict()
    with open("dataset_info/ImageNet_labels.txt", "r") as f:
        lines = f.read().splitlines()
        cls_ids, cls_idxs, cls_names = zip(*[i.split(" ") for i in lines])
        for cls_id, cls_idx, cls_name in zip(cls_ids, cls_idxs, cls_names):
            sample_dict = dict()
            sample_dict['name'] = cls_name
            sample_dict['id'] = cls_id
            sample_dict['idx'] = cls_idx
            imagenet_stats[cls_idx] = sample_dict
            os.makedirs(os.path.join(f"{opt.dataset_out_path}", "train", f"{cls_id}"), exist_ok=True)
    return imagenet_stats

def save_dataset_chunk(all_gpu_samples, cls_idx, output_path):
    all_gpu_samples = 255. * rearrange(all_gpu_samples.cpu().numpy(), 'b c h w -> b h w c')
    np.save(os.path.join(output_path, f"fake_{cls_idx}.npy"), all_gpu_samples.astype(np.uint8))

def main():
    opt = get_args()
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, opt.ckpt)
    sampler = DDIMSampler(model, verbose=False)
    imagenet_stats = prepare_imagenet_stats(opt)
    classes = [i for i in range(opt.start_idx, opt.end_idx+1)]

    ddim_steps = 20
    ddim_eta = 0.0
    scale = 3.0   # for unconditional guidance
    for class_label in classes:
        print(f"Generating images for class {class_label}")
        n_iter = opt.target_images // opt.batch_size if opt.target_images % opt.batch_size == 0 else opt.target_images // opt.batch_size + 1
        n_drop = n_iter * opt.batch_size - opt.target_images 
        all_samples = None
        with torch.no_grad():
            with model.ema_scope():
                for n in tqdm(range(3)):
                    uc = model.get_learned_conditioning({model.cond_stage_key: torch.tensor(opt.batch_size*[1000]).to(model.device)})
                    xc = torch.tensor(opt.batch_size*[class_label])                    
                    c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})                        
                    samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                    conditioning=c,
                                                    batch_size=opt.batch_size,
                                                    shape=[3, 64, 64],
                                                    verbose=False,
                                                    unconditional_guidance_scale=scale,
                                                    unconditional_conditioning=uc, 
                                                    eta=ddim_eta)
                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                                min=0.0, max=1.0)
                    all_samples = x_samples_ddim.cpu() if all_samples is None else torch.cat([all_samples, x_samples_ddim.cpu()], dim=0)
        all_samples = all_samples[:-n_drop] if n_drop > 0 else all_samples
        save_dataset_chunk(all_samples, class_label, os.path.join(f"{opt.dataset_out_path}", "train", f"{imagenet_stats[str(class_label)]['id']}"))

if __name__ == "__main__":
    main()