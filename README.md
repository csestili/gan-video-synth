# gan-video-synth
An abstract video generator based on the [BigGAN](https://tfhub.dev/deepmind/biggan-deep-512/1) image-generation model.

![](doc/example.gif)
# What?
A [video synthesizer](https://en.wikipedia.org/wiki/Video_synthesizer) is a machine (hardware or software or other) that produces video signals that can be changed by an operator in real time through a set of inputs.

A [Generative Adversarial Network (GAN)](https://en.wikipedia.org/wiki/Generative_adversarial_network) is a machine learning model that produces images.
Like a video synthesizer, the images can be changed by changing numerical inputs.
The [BigGAN](https://tfhub.dev/deepmind/biggan-deep-512/1) model is a specific GAN that was trained to produce images that look like photographs.
If you're comfortable programming in Python, you could just use this model directly without any of the code in this repo.

In this repo, there is a simple program that runs this model repeatedly to generate all the frames of an animation, frame-by-frame, thereby generating a video.
The inputs to the program are supposed to be useful if you're creating videos that pulse in time with music.

I call this a **video generator** rather than a video synthesizer because the process to generate the data for each frame and then render as a video is too slow to be run in real-time.
I really want to make this a real-time tool, and if I succeed then I'll call it a true video synthesizer.

The set of programs in this repo may grow over time, as I think about new ways to provide interesting inputs to GANs.

Much of the code in this repo is from the [BigGAN TF Hub Demo Colab](https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb).

# Why?
To make it easy for artists to generate videos with a significant machine learning model.

# How?
The software in this repo runs a neural network on the GPU.
If you'd like to generate video with GANs without doing the following local setup, try using the BigGAN Colab linked above.
The advantages of running locally are that you won't have to download the generated GIFs, and if you have a nice GPU, you might be able to generate videos faster than in the Colab.

## Local setup
1. Ensure that Python 3.7 is installed on your machine.
2. Install `pipenv`:
    ```
   pip install --user pipenv
   ```
   This will allow you to create a virtual environment for an easy and clean install of the libraries used in this repo.
3. Install [CUDA 10.0](https://developer.nvidia.com/cuda-10.0-download-archive). Note this is not the newest version of CUDA. It is the version that is compatible with the `tensorflow-gpu` version used in this repo.
4. Download [cuDNN v7.6.1](https://developer.nvidia.com/rdp/cudnn-archive) and [set it up with your CUDA 10.0 installation](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html). If you've never done this before, have no fear, it's just copying a few files and is easier than it looks :)
5. Install the Python libraries used in this repo:
    ```
    pipenv install
   ```
   
## Running the model
1. Start python in your virtual environment:
    ```
   pipenv run python -i
    ```
2. Load the BigGAN model:
   ```python
   from example import *
   import numpy as np

   gvs = GanVideoSynth()
   ```
 3. Define performance parameters:
    ```python
    bpm = 135
    
    num_beats = 16
    
    classes = [398, 402]
    
    y_scale = 2
    
    random_label = False
    
    quantize_label = False
    
    axis_sets = [
        range(0, 8),
        range(8, 16),
        range(16, 24),
        range(24, 32),
        range(32, 40),
        range(40, 48),
        range(48, 56),
        range(56, 64),
        range(64, 72),
        range(72, 80)
    ]
    
    magnitudes = [
        0.8,
        0.8,
        0.9,
        0.9,
        1,
        1,
        1,
        1,
        1,
        1
    ]
    
    funcs = [
        ramp,
        np.sin,
        ramp,
        np.sin,
        ramp,
        np.sin,
        np.cos,
        np.sin,
        np.cos,
        np.sin
    ]
    ```
4. Run the model with the given parameters:
    ```python
    generate_in_tempo(gvs, bpm=bpm, magnitudes=magnitudes, funcs=funcs, axis_sets=axis_sets,
                      random_label=random_label, classes=classes, quantize_label=quantize_label,
                      y_scale=y_scale)
    ```