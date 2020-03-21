"""example.py: Get Colab working locally.
From https://colab.research.google.com/drive/1ifHb_9Pj5zcCRuCZ_H6P3DjjBbxvXnMH#scrollTo=stWb21nlcyCm
"""

import io
import os
import numpy as np
from scipy.stats import truncnorm
import tensorflow as tf
import tensorflow_hub as hub
import moviepy.editor as mpy
from datetime import datetime

# Cache model so it only has to be downloaded once
os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'


class GanVideoSynth(object):

  @property
  def _module(self):
    if self.__module is None:
      # TODO allow model selection as flag
      module_path = 'https://tfhub.dev/deepmind/biggan-deep-256/1'  # 256x256 BigGAN-deep

      tf.compat.v1.reset_default_graph()
      print('Loading BigGAN module from:', module_path)
      self.__module = hub.Module(module_path)
    return self.__module

  @property
  def _inputs(self):
    if self.__inputs is None:
      self.__inputs = {k: tf.compat.v1.placeholder(v.dtype, v.get_shape().as_list(), k)
                for k, v in self._module.get_input_info_dict().items()}
    return self.__inputs

  @property
  def _output(self):
    if self.__output is None:
      self.__output = self._module(self._inputs)
    return self.__output

  @property
  def _session(self):
    if self.__session is None:
      # Create a TF session and initialize variables
      initializer = tf.compat.v1.global_variables_initializer()
      self.__session = tf.compat.v1.Session()
      self.__session.run(initializer)
    return self.__session

  def __init__(self, render_fps=30):
    self.__module = None
    self.__inputs = None
    self.__output = None
    self.__inputs = None
    self.__session = None

    self.input_z = self._inputs['z']
    self.input_y = self._inputs['y']
    self.input_trunc = self._inputs['truncation']

    self.dim_z = self.input_z.shape.as_list()[1]
    self.vocab_size = self.input_y.shape.as_list()[1]

    self._render_fps = render_fps

  def _write_gif(self, ims, out_dir='renders', ext='.gif'):
    """
    :param self:
    :param ext: (str) '.gif' or '.mp4'
    """
    ext = '.gif'  # or '.mp4'
    fname = os.path.join(out_dir, datetime.now().strftime("%Y%M%d%H%M%S") + ext)
    write_gif(ims, duration=duration, fname=fname, fps=self._render_fps)

  def sample(self, zs, ys, truncation=1., batch_size=8,
             vocab_size=None):
    # zs: [num_interps, gan_video_synth.dim_z]
    # ys: [num_interps, gan_video_synth.vocab_size]
    # truncation: float
    if vocab_size is None:
        vocab_size = self.vocab_size
    zs = np.asarray(zs)
    ys = np.asarray(ys)
    num = zs.shape[0]
    if len(ys.shape) == 0:
      ys = np.asarray([ys] * num)
    if ys.shape[0] != num:
      raise ValueError('Got # z samples ({}) != # y samples ({})'
                       .format(zs.shape[0], ys.shape[0]))
    ys = one_hot_if_needed(ys, self.vocab_size)
    ims = []
    for batch_start in range(0, num, batch_size):
      s = slice(batch_start, min(num, batch_start + batch_size))
      feed_dict = {self.input_z: zs[s], self.input_y: ys[s], self.input_trunc: truncation}
      ims.append(self._session.run(self._output, feed_dict=feed_dict))


    ims = np.concatenate(ims, axis=0)
    assert ims.shape[0] == num
    ims = np.clip(((ims + 1) / 2.0) * 256, 0, 255)
    ims = np.uint8(ims)
    return ims


def truncated_z_sample(batch_size, dim, truncation=1., seed=None):
  state = None if seed is None else np.random.RandomState(seed)
  values = truncnorm.rvs(-2, 2, size=(batch_size, dim), random_state=state)
  return truncation * values

def one_hot(index, vocab_size):
  index = np.asarray(index)
  if len(index.shape) == 0:
    index = np.asarray([index])
  assert len(index.shape) == 1
  num = index.shape[0]
  output = np.zeros((num, vocab_size), dtype=np.float32)
  output[np.arange(num), index] = 1
  return output

def one_hot_if_needed(label, vocab_size):
  label = np.asarray(label)
  if len(label.shape) <= 1:
    label = one_hot(label, vocab_size)
  assert len(label.shape) == 2
  return label


def interpolate(A, B, num_interps):
  if A.shape != B.shape:
    raise ValueError('A and B must have the same shape to interpolate.')
  alphas = np.linspace(0, 1, num_interps)
  return np.array([(1-a)*A + a*B for a in alphas])

def imgrid(imarray, cols=5, pad=1):
  if imarray.dtype != np.uint8:
    raise ValueError('imgrid input imarray must be uint8')
  pad = int(pad)
  assert pad >= 0
  cols = int(cols)
  assert cols >= 1
  N, H, W, C = imarray.shape
  rows = N // cols + int(N % cols != 0)
  batch_pad = rows * cols - N
  assert batch_pad >= 0
  post_pad = [batch_pad, pad, pad, 0]
  pad_arg = [[0, p] for p in post_pad]
  imarray = np.pad(imarray, pad_arg, 'constant', constant_values=255)
  H += pad
  W += pad
  grid = (imarray
          .reshape(rows, cols, H, W, C)
          .transpose(0, 2, 1, 3, 4)
          .reshape(rows*H, cols*W, C))
  if pad:
    grid = grid[:-pad, :-pad]
  return grid

def imshow(a, format='png', jpeg_fallback=True):
  a = np.asarray(a, dtype=np.uint8)
  data = io.BytesIO()
  PIL.Image.fromarray(a).save(data, format)
  im_data = data.getvalue()
  try:
    disp = IPython.display.display(IPython.display.Image(im_data))
  except IOError:
    if jpeg_fallback and format != 'jpeg':
      print(('Warning: image was too large to display in format "{}"; '
             'trying jpeg instead.').format(format))
      return imshow(a, format='jpeg')
    else:
      raise
  return disp

def write_gif(ims, duration=4, fps=30, fname='ani.gif'):

  def make_frame(t):
    # Given time in seconds, produce an array

    frame_num = int(t / duration * ims.shape[0])
    return ims[frame_num]

  clip = mpy.VideoClip(make_frame, duration=duration)
  if fname.endswith('.gif'):
    clip.write_gif(fname, fps=fps, verbose=False)
  elif fname.endswith('.mp4'):
    clip.write_videofile(fname, fps=fps, verbose=False, codec='mpeg4')


    


# My custom generation

#@title Cycles { display-mode: "form", run: "auto" }

truncation = 1 #@param {type:"slider", min:0.02, max:1, step:0.02}
duration = 1 #@param {type:"slider", min:1, max:10, step:0.5}
num_samples = 1 #@param {type:"slider", min:1, max:100, step:1}

def generate(gan_video_synth, fps=30):

  num_interps = duration * fps

  # Indexed label vec
  y_axes = [309]
  y_magnitude = 0.9
  y = np.zeros((1, gan_video_synth.vocab_size))
  for axis in y_axes:
    y[0, axis] = 1
  y = y / np.linalg.norm(y) * y_magnitude

  # Expand ys out to full shape
  ys = np.repeat(y, num_interps, axis=0)

  # Random z vec
  # Here the seed itself is a function of the current microsecond
  noise_seed_z = int(datetime.now().strftime('%f'))
  z0 = truncated_z_sample(1, gan_video_synth.dim_z, truncation, noise_seed_z)

  # Interpolation settings
  # Axes to change in [0, 128]
  sin_axes = range(0, 32)
  cos_axes = range(110, 120)
  sin_double_axes = range(70, 80)
  cos_double_axes = range(80, 90)
  sin_quad_axes = range(100, 110)
  cos_quad_axes = range(32, 64)
  # Magnitude of change
  change_mag = 1
  change_mag_double = 0.8
  change_mag_quad = 0.7

  zs = np.repeat(z0, num_interps, axis=0)
  ts_1, ts_2, ts_4 = [
    np.linspace(0, itm * 2 * np.pi, num=num_interps)
    for itm in [1, 2, 4]
  ]
  for axis in sin_axes:
    zs[:, axis] += np.sin(ts_1) * change_mag
  for axis in cos_axes:
    zs[:, axis] += np.cos(ts_1) * change_mag
  for axis in sin_double_axes:
    zs[:, axis] += np.sin(ts_2) * change_mag_double
  for axis in cos_double_axes:
    zs[:, axis] += np.cos(ts_2) * change_mag_double
  for axis in sin_quad_axes:
    zs[:, axis] += np.sin(ts_4) * change_mag_quad
  for axis in cos_quad_axes:
    zs[:, axis] += np.cos(ts_4) * change_mag_quad

  # Generate images
  ims = gan_video_synth.sample(zs, ys, truncation=truncation)
  gan_video_synth._write_gif(ims, out_dir='renders')


gan_video_synth = GanVideoSynth()

# TODO generate multiple samples as a batch, not as a loop
for _ in range(num_samples):
  generate(gan_video_synth)


