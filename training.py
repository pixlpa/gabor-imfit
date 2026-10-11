import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import os
import re
import copy
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur
from imfit import GaborLayer,ImageFitter

try:
    import ffmpeg          # only needed by --mode vx
except ImportError:
    ffmpeg = None

def process_image(image_name,weight_path,args,device,previous=None):
    im_name = os.path.basename(image_name)
    if args.mode == 'folder' or args.mode == 'video':
        image_path = os.path.join(args.image, image_name)
    else:
        image_path = image_name
    short_name = os.path.splitext(im_name)[0]
    # Determine target size
    target_size = args.size
    init_gabors = args.num_gabors
    init = None
    if previous:
        init = previous

    # Initialize fitter with target size and learning rates
    kwargs = dict(
        global_lr=args.global_lr,
        debug = args.debug,
        mutation_strength=args.mutation_strength,
        gamma = args.gamma,
        sobel = args.sobel,
        gradient = args.gradient,
        l1 = args.l1
    )
    if getattr(args, 'guards', None) is not None:
        kwargs['guards'] = args.guards
    fitter = getattr(args, 'fitter_cls', ImageFitter)(
        image_path,
        weight_path,
        init_gabors,
        target_size,
        device,
        init,
        **kwargs
    )
    #save initial image
    if init and args.debug:
        fitter.save_image(os.path.join(args.output_dir, f"initial_{image_name}.png"))

    # Training loop
    print(f"Training {image_name} on {device}...")
    progress = 0
    fitter.init_optimizer(args.global_lr)
    #run loop for each scale
    for a in range(args.rescales):
        factor = args.rescales - a
        scaler = args.size/(2 ** factor)
        print(f"Optimizing at size: {scaler: .1f}")
        fitter.resize_target(int(scaler))
        # re-initialize scheduler for full learning rate
        for param_group in fitter.optimizer.param_groups:
            param_group['lr'] = args.global_lr
        fitter.scheduler.base_lrs = [args.global_lr]
        fitter.scheduler.last_epoch = -1
        iter_count = int( args.iterations * factor * args.iter_multiple)
        with tqdm(total=iter_count) as pbar:
            for i in range(iter_count):
                loss = fitter.train_step(i, args.iterations)    
                if i % 5 == 0:
                    temp = fitter.scheduler.get_last_lr()[0]
                    pbar.set_postfix(loss=f"{loss:.6f}", lr=f"{temp:.6f}")
                    pbar.update(5)
    fitter.resize_target(args.size)
    for param_group in fitter.optimizer.param_groups:
        param_group['lr'] = args.global_lr
    fitter.scheduler.base_lrs = [args.global_lr]
    fitter.scheduler.last_epoch = -1
    print("Optimizing at full size")
    with tqdm(total=args.iterations) as pbar:
        for i in range(args.iterations):
            loss = fitter.train_step(i, args.iterations, save_best = True)    
            if i % 5 == 0:
                temp = fitter.scheduler.get_last_lr()[0]
                pbar.set_postfix(loss=f"{loss:.6f}", lr=f"{temp:.6f}")
                pbar.update(5)
    
    # Save final result
    if args.output_dir:
        if args.debug:
            if args.mode == 'image':
                fitter.save_final(os.path.join(args.output_dir, f"{short_name}.png"))
            else:
                fitter.save_final(os.path.join(args.output_dir, image_name))
        fitter.save_weights(os.path.join(args.output_dir, f"{short_name}.txt"))
        print(f"Saved {image_name} result with loss: {fitter.best_loss:.6f}")
        return os.path.join(args.output_dir,f"{short_name}.txt")

def by_nums(filename):
    numstring = re.findall(r'\d+', filename)
    if len(numstring)>0:
        return int(numstring[0])
    else:
        return 0

def sorted_by_nums(filenames):
    return sorted(filenames, key=by_nums)

def limit_by_nums(filenames,start,end):
    newlist = []
    for file in filenames:
        numstring = re.findall(r'\d+', file)
        numval = int(numstring[0])
        if numval >= start and numval <= end:
            newlist.append(file)
    return newlist

def process_folder(args,device):
    filelist = sorted_by_nums(os.listdir(args.image))
    for filename in filelist:
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Check file extension
            im_name = os.path.splitext(filename)[0]
            if args.weight:
                weight_name = f"{im_name}.png"
                weight_path = os.path.join(args.weight,weight_name)
                if os.path.exists(weight_path):
                    process_image(filename,weight_path,args,device,args.init)
                else:
                    process_image(filename,None,args,device,args.init)
            else:
                process_image(filename,None,args,device,args.init)

def process_video(args,device):
    filelist = sorted_by_nums(os.listdir(args.image))
    if args.end:
        filelist = limit_by_nums(filelist,args.start,args.end)
    count = 0
    kframes = args.keyframes
    previous = None
    for filename in filelist:
        if filename.endswith((".png", ".jpg", ".jpeg")):  # Check file extension
            im_name = os.path.splitext(filename)[0]
            if args.weight:
                weight_name = f"{im_name}.png"
                weight_path = os.path.join(args.weight,weight_name)
                if count%kframes == 0:
                    previous = None
                if os.path.exists(weight_path):
                    previous = process_image(filename,weight_path,args,device,previous)
                else:
                    previous = process_image(filename,None,args,device,previous)
            else:
                previous = process_image(filename,None,args,device,previous)
            count=count+1
def extract_frames(src, dest, size=None):
    """Extract src to dest/img%04d.png, skipping if frames are already there.

    size caps the longest edge (aspect preserved). Frames only ever get resized
    down to --size before fitting, so exporting larger than that is wasted disk:
    a 6800-frame 720p clip is ~10 GB at full resolution and ~250 MB at 256.
    """
    os.makedirs(dest, exist_ok=True)
    stamp = os.path.join(dest, '.export_size')
    existing = [f for f in os.listdir(dest) if f.endswith('.png')]
    if existing:
        prev = None
        if os.path.exists(stamp):
            prev = open(stamp).read().strip() or None
        want = str(size) if size else None
        if prev != want:
            raise SystemExit(
                f"{dest} already holds {len(existing)} frames exported at "
                f"{prev or 'full size'}, but --export-size is now "
                f"{want or 'unset (full size)'}.\n"
                f"Delete that directory to re-export, or match the previous value.")
        print(f"Reusing {len(existing)} already-extracted frames in {dest}")
        return dest

    print(f"Extracting {src} -> {dest}" + (f" (longest edge {size})" if size else ""))
    stream = ffmpeg.input(src)
    if size:
        # longest edge -> size, other edge auto; -2 keeps it even.
        # lanczos to match ImageFitter's own PIL resize, so a frame exported at
        # --size gives the same target as one exported full-size and downscaled.
        stream = stream.filter('scale',
                               f'if(gt(iw,ih),{size},-2)',
                               f'if(gt(iw,ih),-2,{size})',
                               flags='lanczos')
    stream.output(os.path.join(dest, 'img%04d.png')).run()
    n = len([f for f in os.listdir(dest) if f.endswith('.png')])
    if n == 0:
        raise SystemExit(f"ffmpeg extracted no frames from {src}")
    with open(stamp, 'w') as f:
        f.write(str(size) if size else '')
    print(f"Extracted {n} frames")
    return dest

def process_vx(args,device):
    if ffmpeg is None:
        raise SystemExit("--mode vx needs ffmpeg-python: pip install ffmpeg-python")
    if not os.path.exists(args.image):
        raise SystemExit(f"no such video: {args.image}")
    vargs = copy.copy(args)
    vargs.mode = 'video'
    # --export used to be required, and training was skipped without it with no
    # message at all. Default it next to the output so the documented
    # invocation works.
    export = args.export or os.path.join(args.output_dir, 'frames')
    os.makedirs(export, exist_ok=True)
    esize = args.export_size
    if esize is None and args.size:
        print(f"note: exporting at full resolution. Frames are downscaled to "
              f"--size {args.size} before fitting anyway, so --export-size "
              f"{args.size} would save a lot of disk.")
    elif esize and args.size and esize < args.size:
        print(f"warning: --export-size {esize} is below --size {args.size}; "
              f"frames will be upscaled back up before fitting.")
    vargs.image = extract_frames(args.image, os.path.join(export, 'images'), esize)
    if args.weight:
        vargs.weight = extract_frames(args.weight, os.path.join(export, 'weights'), esize)
    process_video(vargs,device)


def process_vm(args,device):
    fpath = args.image
    outdir = args.output_dir
    folderlist = os.listdir(fpath)
    for folder in folderlist:
        fullpath = os.path.join(fpath,folder)
        if os.path.exists(os.path.join(fullpath,'images')):
            print(f"Training video: {folder}")
            fullpath = os.path.join(fpath,folder)
            print(f"Full path: {fullpath}")
            fname = os.path.split(folder)[1]
            vargs = copy.copy(args)
            vargs.image = os.path.join(fullpath,'images')
            vargs.weight = os.path.join(fullpath,'weights')
            vargs.output_dir = os.path.join(args.output_dir,fname)
            print(f"Output Directory:{vargs.output_dir}")
            os.makedirs(vargs.output_dir, exist_ok=True)
            vargs.mode = 'video'
            process_video(vargs,device)

def main():
    """Run Gabor image fitting on an input image."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fit image with Gabor functions')
    parser.add_argument('image', type=str, help='Path to input image')
    parser.add_argument('--weight', type=str, help='Path to weight image (grayscale)', default=None)
    parser.add_argument('--num-gabors', type=int, default=256,
                       help='Number of Gabor functions to fit')
    parser.add_argument('--iterations', type=int, default=1000,
                       help='Number of training iterations')
    parser.add_argument('--single-iterations', type=int, default=100,
                       help='Number of training iterations')
    parser.add_argument('--init', type=str, default=None,
                       help='load initial parameters from file')
    parser.add_argument('--rescales', type=int, default=2,
                       help='number of scale factors to use')
    parser.add_argument('--iter-multiple', type=float, default=1.5,
                       help='number of scale factors to use')
    # Add size arguments
    parser.add_argument('--size', type=int, default=None,
                       help='Target size (maintains aspect ratio)')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Directory for output files')
    parser.add_argument('--gradient', type=float, default=0.,
                       help='weight of gradient loss function')
    parser.add_argument('--l1', type=float, default=0.,
                       help='weight of L1 loss function')
    parser.add_argument('--sobel', type=float, default=0.,
                       help='weight of sobel loss function')
    parser.add_argument('--global-lr', type=float, default=0.01,
                       help='Learning rate for global phase')
    parser.add_argument('--mutation-strength', type=float, default=0.0,
                       help='Mutation strength')
    parser.add_argument('--gamma', type=float, default=0.997,
                       help='learning rate gamma')
    parser.add_argument('--mode', type=str, default='image')
    parser.add_argument('--debug',type=int, default= None, help="Size of debug images")
    parser.add_argument('--keyframes', type=int, default=100, help="Keyframe every N frames")
    parser.add_argument('--start', type=int, default=0, help="Start frame")
    parser.add_argument('--end', type=int, default=None, help="Optional end frame")
    parser.add_argument('--export', type=str, default=None,
                       help="Where --mode vx writes extracted frames "
                            "(default: <output-dir>/frames)")
    parser.add_argument('--export-size', type=int, default=None,
                       help="Longest edge for frames extracted by --mode vx "
                            "(aspect preserved). Default: source resolution. "
                            "Frames are downscaled to --size before fitting, so "
                            "setting this to --size avoids writing huge PNGs.")
    parser.add_argument('--seed', type=int, default=None,
                       help="Seed the RNG so runs are reproducible (needed to A/B engines)")

    # ── engine selection ──────────────────────────────────────────────────
    parser.add_argument('--engine', choices=['classic', 'fast'], default='classic',
                       help="'classic' = imfit.py dense render; "
                            "'fast' = imfit_fast.py fused Metal kernel (MPS)")
    # Guards below apply to --engine fast only; all default to off.
    parser.add_argument('--band-dilate-px', type=float, default=0.0,
                       help="[fast] Mip band-limit in pixels; suppresses sub-pixel needles")
    parser.add_argument('--freq-floor-px', type=float, default=0.0,
                       help="[fast] Minimum carrier period in pixels")
    parser.add_argument('--amp-penalty', type=float, default=0.0,
                       help="[fast] Penalty weight on amplitude in sub-pixel-thin atoms")
    parser.add_argument('--group-lr', action='store_true',
                       help="[fast] Per-parameter-group learning rates")
    parser.add_argument('--relocate-every', type=int, default=0,
                       help="[fast] Relocate negligible atoms every N steps (0=off)")
    parser.add_argument('--relocate-thresh', type=float, default=1e-3,
                       help="[fast] Contribution proxy below which an atom is parked")
    parser.add_argument('--relocate-until', type=float, default=0.75,
                       help="[fast] Stop relocating after this fraction of each phase")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Resolve the engine. I/O is identical either way; `fast` only adds the
    # `guards` kwarg, which process_image passes through when present.
    args.guards = None
    if args.engine == 'fast':
        from imfit_fast import Guards, ImageFitter as FastImageFitter
        args.fitter_cls = FastImageFitter
        args.guards = Guards.from_args(args)
    else:
        args.fitter_cls = ImageFitter

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    if args.mode == 'folder':
        process_folder(args, device)
    elif args.mode == 'video':
        process_video(args, device)
    elif args.mode == 'vm':
        process_vm(args,device)
    elif args.mode == 'vx':
        process_vx(args,device)
    else:
        process_image(args.image,args.weight,args,device,None)

if __name__ == '__main__':
    main()