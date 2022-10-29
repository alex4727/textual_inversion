import argparse, os, json, time
import numpy as np
from PIL import Image
from datetime import datetime

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
            sample_dict['prompt'] = f"a photo of a {cls_name.replace('_', ' ')}"
            sample_dict['folder'] = os.path.join(f"{opt.watch_dir}", "train", f"{cls_id}")
            imagenet_stats[cls_idx] = sample_dict
    
    status = dict()
    for cls_idx, cls_dict in imagenet_stats.items():
        status[cls_idx] = "not_done"

    return imagenet_stats, status

def prepare_cifar10_stats(opt):
    print("Preparing cifar10 stats")
    cifar10_stats = dict()
    with open("dataset_info/cifar10-labels.txt", "r") as f:
        lines = f.read().splitlines()
        cls_ids, cls_idxs, cls_names = zip(*[i.split(" ") for i in lines])
        for cls_id, cls_idx, cls_name in zip(cls_ids, cls_idxs, cls_names):
            sample_dict = dict()
            sample_dict['name'] = cls_name
            sample_dict['id'] = cls_id
            sample_dict['idx'] = cls_idx
            sample_dict['prompt'] = f"a photo of a {cls_name.replace('_', ' ')}"
            sample_dict['folder'] = os.path.join(f"{opt.watch_dir}", "train", f"{cls_id}")
            cifar10_stats[cls_idx] = sample_dict

    status = dict()
    for cls_idx, cls_dict in cifar10_stats.items():
        status[cls_idx] = "not_done"

    return cifar10_stats, status

def prepare_cifar100_stats(opt):
    print("Preparing cifar100 stats")
    cifar100_stats = dict()
    with open("dataset_info/cifar100-labels.txt", "r") as f:
        lines = f.read().splitlines()
        cls_ids, cls_idxs, cls_names = zip(*[i.split(" ") for i in lines])
        for cls_id, cls_idx, cls_name in zip(cls_ids, cls_idxs, cls_names):
            sample_dict = dict()
            sample_dict['name'] = cls_name
            sample_dict['id'] = cls_id
            sample_dict['idx'] = cls_idx
            sample_dict['prompt'] = f"a photo of a {cls_name.replace('_', ' ')}"
            sample_dict['folder'] = os.path.join(f"{opt.watch_dir}", "train", f"{cls_id}")
            cifar100_stats[cls_idx] = sample_dict

    status = dict()
    for cls_idx, cls_dict in cifar100_stats.items():
        status[cls_idx] = "not_done"

    return cifar100_stats, status

def save_jpeg(npy_name, dataset_stats, cls_idx, target_images):
    count = 0
    samples = np.load(npy_name)
    if samples.shape[0] != target_images:
        print(f"Possible Error: {cls_idx} npz has {samples.shape[0]} samples, but should have {target_images}")
    for sample in samples:
        img = Image.fromarray(sample)
        img.save(os.path.join(dataset_stats[cls_idx]["folder"], f"fake_{cls_idx}_{count:04}.JPEG"))
        count += 1

def update_status_file(status, status_file):
    with open(status_file, "w") as f:
        json.dump(status, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--watch_dir", type=str, required=True)
    parser.add_argument("--status_file", type=str, default="None", required=False)
    parser.add_argument("--target_images", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)"
    opt = parser.parse_args()
    if opt.dataset == "imagenet":
        dataset_stats, status = prepare_imagenet_stats(opt)
    elif opt.dataset == "cifar10":
        dataset_stats, status = prepare_cifar10_stats(opt)
    elif opt.dataset == "cifar100":
        dataset_stats, status = prepare_cifar100_stats(opt)

    if opt.status_file != "None":
        with open(opt.status_file, "r") as f:
            status = json.load(f)
    else:
        current_time = datetime.now().strftime('%Y_%b%d_%H-%M-%S')
        opt.status_file = f"{current_time}-watching.json"
        with open(opt.status_file, "w") as f:
            json.dump(status, f)

    while True:
        for cls_idx in dataset_stats.keys():
            npy_name = os.path.join(dataset_stats[cls_idx]["folder"], f"fake_{cls_idx}.npy")
            fake_images = [i for i in os.listdir(dataset_stats[cls_idx]["folder"]) if "fake" in i]
            if status[cls_idx] == "not_done" and os.path.exists(npy_name):
                # if npy file exists and no jpegs
                print(f"Found {npy_name}... turning into images", flush=True)
                time.sleep(60)
                save_jpeg(npy_name, dataset_stats, cls_idx, opt.target_images)
                status[cls_idx] = "done"
                update_status_file(status, opt.status_file)
                os.remove(npy_name)
            elif status[cls_idx] == "not_done" and len(fake_images) == opt.target_images:
                # already converted to jpegs (possible in continued case)
                print(f"Found images for {cls_idx}... marking as done", flush=True)
                if os.path.exists(npy_name):
                    os.remove(npy_name)
                status[cls_idx] = "done"
                update_status_file(status, opt.status_file)
            else:
                pass
        print("Going to sleep...")
        time.sleep(200)

if __name__ == "__main__":
    main()
