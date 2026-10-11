# torch-imfit
RGB Image fitting using a Gabor model - rewritten from scratch for modern pytorch.

These scripts optimize an array of Gabor wavelets to synthesize an input image. The purpose for this is to generate a parametric representation of the image that can be resynthesized elsewhere. 

Each Gabor wavelet has 12 floating point values:
- Position (X,Y)
- Rotation (theta)
- Gaussian Shape (sigma, gamma)
- Wavelength (f)
- Phase (R,G,B)
- Amplitude (R,G,B)

## Synthesis
The Gabor wavelet representation is fairly easy to replicate in GLSL fragment program with this (pseudocode):
~~~
    //calculate frequency from wavelength value
    float f = TWO_PI/exp(wavelength);
    //subtract fragment location from wavelet position (center)
    vec2 delt = texcoord - pos;

    //rotate coordinates
    vec2 rot = rotate2D(delt, theta);
    //calculate wave across x-coordinate
	vec3 wave = cos(vec3(f*rot.x)+(phase*TWO_PI));
    
    //calculate 2D gaussian window
	vec2 p = rot*rot
    vec2 scale = vec2(sigma,gamma);
    vec2 sg = scale*scale;
    float gaussian = exp(-(p.x)/(2.*sg.x) - (p.y)/(2.*sg.y));
	out_color = vec4(clamp(wave * amplitude * gaussian, -1., 1.), 1.);
~~~

There are different strategies for rendering the entire set, including passing all of the parameters into a shader as uniforms and rendering all of the wavelets in a loop, or rendering each wavelet as a billboard quad and blending those together. The important thing to remember is that all of the wavelets should be blended additively, and the resulting image should be re-scaled from a -1 … 1 range to a 0 … 1 range.
The benefit of the resulting representation as wavelets allows for a great deal of creative manipulation of those parameter values, and a compelling abstract look, especially around the gradient edges and beyond the trained area.

## Usage
The main script to use here is ~training.py~ which can be run from the terminal using python.
~~~
    cd torch-imfit
    python3 training.py myimage.png --output-dir results/myresults/ --mode image --weight myimage-wt.png
~~~
### Arguments:
+ (image/folder) - path to the image or folder of images to train (required)
+ --mode - (image/folder/video) Defines how paths are interpreted and how the loop is run. Image trains a single image file. Folder trains a folder of images independently. Video mode trains a folder of image frames, with the added feature that it will initialize training of each frame using the previous frame results, with a new "keyframe" every 10 frames. The video mode is created to decrease the amount of flicker between frames from discontinuities in how the wavelets are trained.
+ --weight - path to the image or folder of images to use as weights. Expected naming convention is "<imagename>-wt.png" (optional)
+ --iterations - baseline number of iterations for each training scale (required)
+ --iter-multiple - iteration multiplier at each training scale, allows for more training at lower resolutions. default is 1.5
+ --output-dir - path to the directory where files will be saved (required)
+ --rescales - number of training scales. Each scale is a division by 2, so with 2 rescales and size 256, it will train at 64, 128, and 256. default is 2
+ --size - maximum pixel size for training. Aspect ratio will automatically be maintained. default is 256
+ --num-gabors - number of wavelets to train. Higher values allow for more fidelity. default is 256
### Training/Loss Arguments
Most of these are unnecessary and were mostly experiments during development, but might be good to play with for different results.
+ -- gradient - values greater than 0 will add a gradient loss calculation. default is 0
+ -- sobel - values greater than 0 will add a sobel filter loss calculation, useful for more edge definition. default is 0
+ -- l1 -values greater than 0 will add L1 criterion loss calculation (sum of per-pixel loss)
+ -- global-lr - Learning rate for training. default is 0.01
+ -- gamma - Learning rate multiplier when loss stops decreasing. default is 0.997
+ -- mutation-scale - add parameter mutation during training for some added randomness. default is 0

## Installation and Setup
Make sure to Python 3 and PIP are installed before attempting.
1. Open a terminal/command prompt and clone this repo:
~~~
cd path-to-folder
git clone https://github.com/pixlpa/gabor-imfit.git
cd gabor-imfit
~~~
2. (Recommended) Create a python virtual environment with venv:
    % python -m venv venv
    % venv/Scripts/activate
3. Install dependencies:
    % pip install torch torchvision numpy tqdm
4. Create `source-images, source-weights, and results` folders in the gabor-imfit/ directory.
5. Place images and weights in these folders, respectively. Image names should follow "img0001.png" format, and weight images should follow "img0001-wt.png" file name convention, especially for 'folder' and 'video' modes.
6. You should now be ready to run the `training.py` script. You can test by running the following command:
    % python training.py source-images/img0001.png --weight source-weights/img001-wt.png --output-dir results/test/ --size 256
7. If all was successful, it should run the training loop and store a preview image and weights txt file in the results/test/ folder. 
8. If you set up a virtual environment, you can run `venv/Scripts/activate` whenever starting a new session

##Examples
~~~
% python3 training.py images/img001.png --weight weights/img001.png --iterations 200 --rescales 2 --output-dir results/ --size 256 --num-gabors 256 --mode image
~~~
In image mode, you can train a single image

~~~
% python3 training.py images/img001.png --weight weights/img001.png --iterations 200 --rescales 2 --output-dir results/ --size 256 --num-gabors 256 --mode image
~~~
In image mode, you can train a single image

~~~
% python3 training.py images/ --weight weights/ --iterations 500 --rescales 3 --output-dir results/ --size 512 --num-gabors 256 --mode folder
~~~
This will run a higher quality training with more iterations, more rescale cycles, and higher training resolution. This will run much more slowly for each image but will provide greater accuracy.

~~~
% python3 training.py D1/images/ --weight D1/weights/ --iterations 200 --rescales 2 --output-dir results/ --size 256 --num-gabors 256 --mode video
~~~
In video mode, you can train a folder of images in a sequence, where each frame is initialized from the results of the previous frame. Every 100 frames there is a "keyframe" that is not initialized and begins the training process fresh. This is done to reduce flicker and tween issues from having each frame be completely unique. You can set the interval of fresh keyframes with the --keyframes argument. 

~~~
% python3 training.py D1/ --iterations 200 --rescales 2 --output-dir results/ --size 256 --num-gabors 256 --mode vm
~~~
In 'vm' mode, you can train a folder full of video folders, each of which must have an "images" and a "weights" folder inside the directory. This mode allows for training a whole series of video image sequences in bulk rather than running individual sessions.

% python3 training.py D1.mp4 --weight D1-Z.mp4 --iterations 200 --rescales 2 --output-dir results/ --size 256 --num-gabors 256 --mode vx --start 100 --end 300 --keyframes 100 --export-size 256
~~~
In 'vx' mode, you can input movie clips directly and it will automatically export frame sequences with a proper folder structure, and begin training. This example also uses start and end frame numbers to define a limited range of training. It also sets the keyframe interval - tradeoff between less "glitches" with longer keyframe intervals and more accuracy with shorter intervals.

Frames land in `--export` (default `<output-dir>/frames`) and are reused on a
re-run, so a second pass with the same settings skips extraction. Changing
`--export-size` against an existing export is refused rather than silently
mixing resolutions — delete the directory to re-export.

**Set `--export-size`.** Extraction writes one PNG per frame at the source
resolution by default, and frames are downscaled to `--size` before fitting
anyway, so anything larger is wasted disk: a 6873-frame 720p clip is **6.8 GB
at full resolution and 0.5 GB at `--export-size 256`**. Setting it to `--size`
is almost always right. (The scaler uses lanczos to match the fitter's own PIL
resize; the two routes still differ by ~41 dB PSNR, far below the fit's own
error, but don't mix exports at different sizes within one sequence.)

Note that `--start` / `--end` filter the extracted frames, they do not limit
extraction — the whole clip is still written out first. To work on a short
stretch, cut the source with ffmpeg instead.


## Fast engine (`--engine fast`)

`imfit_fast.py` is an alternate fitting engine that swaps the dense Gabor
broadcast for a fused Metal kernel (`gabor_metal/`, vendored from
jitter-vision). It plugs into `training.py` through the same interface and
writes the same `.txt` weight format, so it is interchangeable with the default
engine mid-pipeline, including the video `--init` warm-start chain.

The classic `GaborLayer.forward` builds `amplitude * gaussian * sinusoid` at
`[N, 3, H, W]` before summing over N. At `--num-gabors 256 --size 512` that is
805 MB for one intermediate, ~2 GB per forward with the others, and again for
the backward. The fast engine computes the same sum with O(1) intermediate
memory, so it raises the resolution and atom-count ceiling as well as the speed.

~~~
% python3 training.py image.png --engine fast --num-gabors 256 --size 512 --iterations 1000
~~~

Measured at `--num-gabors 256 --size 512 --iterations 200 --rescales 0`:

| | classic | fast |
|---|---|---|
| wall clock | 68.4 s | **2.97 s** |
| final loss | 0.024570 | **0.023630** |
| forward render alone | 114 ms | **1.20 ms** |
| backward render alone | — | **4.84 ms** |

The fast engine also lands slightly *better*, not just sooner — the tighter
gamma floor keeps atoms out of the kernel's frozen-gradient band (see below).

That is 23x. The first version of this engine only reached 5.3x, because
swapping the forward exposed the backward: `gabor_metal` dispatched one *thread*
per (batch, gabor), so at B=1, N=256 just 256 threads each swept all 262144
pixels and the step was 44.5 ms of backward against 1.3 ms of forward. The
kernel now dispatches one *threadgroup* per gabor with a simd_sum reduction
(ported from `jit.gaborfit`), taking the backward to 4.84 ms.

That kernel picks its dispatch by `B*N`: the parallel path wins by up to 16x
while `B*N` alone underfills the GPU and loses above `B*N ~= 2800`, where the
reduction becomes pure overhead. Single-image fitting like this one is the best
case for it.

Requires Apple Silicon (MPS). On any other device it falls back to an
equivalent dense render with no speedup, so the flag is always safe to pass.

### Guards

Optional fit guardrails, ported from jitter-vision `tools/data/fit_guards.py`.
All default to off, so `--engine fast` on its own changes speed and nothing
else. They apply to the fast engine only.

| flag | effect |
|---|---|
| `--band-dilate-px` | Mip band-limit, `sigma_eff = sqrt(sigma² + d²)`; suppresses sub-pixel needle atoms |
| `--freq-floor-px` | minimum carrier period in pixels |
| `--amp-penalty` | penalty on amplitude carried by sub-pixel-thin atoms |
| `--group-lr` | per-parameter-group learning rates (position > geometry > frequency) |
| `--relocate-every` | every N steps, move negligible atoms to high-residual locations and reset their Adam moments |
| `--relocate-until` | stop relocating after this fraction of each phase (default 0.75) |

`--relocate-until` exists because relocation that runs to the last step leaves a
cohort that never converges: in the jitter-vision cache fits, dead atoms went
from 3.1% at 200 steps to 11.2% at 1600 with no cutoff.

A `--seed` argument was added at the same time; without it neither engine is
reproducible, which makes A/B comparison impossible.

### Difference from the default engine

One intentional behavioural difference: the fast engine clamps `gamma` to
`[1e-3, 1]` where the classic engine uses `[1e-4, 1]`. The Metal kernel floors
the gamma denominator at `1e-3` and zeroes its gradient below, so atoms in
`[1e-4, 1e-3)` would be permanently frozen as degenerate needles. With the floor
matched, the two renders agree to float round-off (max 1e-5 on a [-1,1] image).

Note that this affects more atoms than it sounds like. `GaborLayer.__init__`
calls `Tensor.normal_(GABOR_MIN[k], GABOR_MAX[k])`, and `normal_` takes
*(mean, std)* rather than a range — so `gamma` is drawn from `N(0.01, 0.4)` and
roughly half of every initial population lands on its clamp. On a 48-atom test,
28 atoms started below `1e-3`.
