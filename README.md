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
(image/folder) - path to the image or folder of images to train (required)
--mode - (image/folder/video) Defines how paths are interpreted and how the loop is run. Image trains a single image file. Folder trains a folder of images independently. Video mode trains a folder of image frames, with the added feature that it will initialize training of each frame using the previous frame results, with a new "keyframe" every 10 frames. The video mode is created to decrease the amount of flicker between frames from discontinuities in how the wavelets are trained.
--weight - path to the image or folder of images to use as weights. Expected naming convention is "<imagename>-wt.png" (optional)
--iterations - baseline number of iterations for each training scale (required)
--iter-multiple - iteration multiplier at each training scale, allows for more training at lower resolutions. default is 1.5
--output-dir - path to the directory where files will be saved (required)
--rescales - number of training scales. Each scale is a division by 2, so with 2 rescales and size 256, it will train at 64, 128, and 256. default is 2
--size - maximum pixel size for training. Aspect ratio will automatically be maintained. default is 256
--num-gabors - number of wavelets to train. Higher values allow for more fidelity. default is 256
### Training/Loss Arguments
Most of these are unnecessary and were mostly experiments during development, but might be good to play with for different results.
-- gradient - values greater than 0 will add a gradient loss calculation. default is 0
-- sobel - values greater than 0 will add a sobel filter loss calculation, useful for more edge definition. default is 0
-- l1 -values greater than 0 will add L1 criterion loss calculation (sum of per-pixel loss)
-- global-lr - Learning rate for training. default is 0.01
-- gamma - Learning rate multiplier when loss stops decreasing. default is 0.997
-- mutation-scale - add parameter mutation during training for some added randomness. default is 0
