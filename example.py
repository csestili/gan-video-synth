"""example.py: Generate video with BigGAN model.
Much of the code in this example is from the BigGAN TF Hub Demo Colab: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb
"""

import argparse
import os
import numpy as np
from scipy.stats import truncnorm
import tensorflow as tf
import tensorflow_hub as hub
import moviepy.editor as mpy
from datetime import datetime

# Cache model so it only has to be downloaded once
os.environ["TFHUB_CACHE_DIR"] = '/tmp/tfhub'


TAU = 2 * np.pi


MODULE_PATH_MAPPING = {
  # These models are included since they all have 128-dimensional z-space.
  '128-deep': 'https://tfhub.dev/deepmind/biggan-deep-128/1',
  '256-deep': 'https://tfhub.dev/deepmind/biggan-deep-256/1',
  '512-deep': 'https://tfhub.dev/deepmind/biggan-deep-512/1',
  '512': 'https://tfhub.dev/deepmind/biggan-512/2',
}


class GanVideoSynth(object):

  @property
  def _module(self):
    if self.__module is None:
      module_path = MODULE_PATH_MAPPING[self.model_string]

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

  def __init__(self, model_string='512-deep'):
    self.model_string = model_string
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

  def write_gif(self, ims, duration, out_dir='renders', fps=20, ext='.gif'):
    """
    :param self:
    :param ext: (str) '.gif' or '.mp4'
    """
    fname = os.path.join(out_dir, datetime.now().strftime("%Y%m%d%H%M%S") + ext)
    write_gif(ims, duration=duration, fname=fname, fps=fps)

  def sample(self, zs, ys, truncation=1., batch_size=16,
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

def write_gif(ims, duration=4, fps=30, fname='ani.gif'):
  num_frames = ims.shape[0]

  def make_frame(t):
    # Given time in seconds, produce an array

    frame_num = int(t / duration * num_frames)
    return ims[frame_num]

  clip = mpy.VideoClip(make_frame, duration=duration)
  if fname.endswith('.gif'):
    clip.write_gif(fname, fps=fps, verbose=False)
  elif fname.endswith('.mp4'):
    clip.write_videofile(fname, fps=fps, verbose=False, codec='mpeg4')


# Image processing functions

def fit_to_ratio(ims, aspect_ratio=16/9):
  """Pad a square image out to a rectangular aspect ratio, by mirroring the sides.
  Args:
    ims (np array) [num_frames, height, width, 3]
    aspect_ratio (float) ratio of width to height

  Outputs:
    (np array) [num_frames, height, int(width * aspect_ratio), 3]
  """
  if aspect_ratio < 1:
    raise NotImplementedError("Expected aspect ratio to have greater width than height")
  if len(ims.shape) != 4:
    raise NotImplementedError("Expected input with dimensions (num_frames, height, width, 3); got {}".format(ims.shape))

  orig_width = ims.shape[2]

  # Round out_width to nearest pixel.
  out_width = int(orig_width * aspect_ratio)
  left_pad, right_pad = int(np.floor((out_width - orig_width) / 2)), int(np.ceil((out_width - orig_width) / 2))

  out_shape = (ims.shape[0], ims.shape[1], out_width, ims.shape[3])
  res = np.zeros(out_shape, dtype=np.uint8)

  for i in range(ims.shape[0]):
    # Original image in center
    res[i, :, left_pad:left_pad + orig_width, :] = ims[i]
    # Mirror left
    res[i, :, 0:left_pad, :] = ims[i, :, left_pad:0:-1, :]
    # Mirror right
    res[i, :, left_pad + orig_width:out_width, :] = ims[i, :, orig_width:orig_width - right_pad - 1:-1, :]

  return res


# Generation functions

def generate(gan_video_synth, fps=30):
  truncation = 1
  duration = 1

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
    np.linspace(0, itm * TAU, num=num_interps)
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
  gan_video_synth.write_gif(ims, duration, out_dir='renders')


def ramp(x, phase=0):
  """Saw wave."""
  return (x + phase) % TAU


def generate_in_tempo(gan_video_synth, bpm=120, num_beats=16, classes=[309], y_scale=1, truncation=1,
                      random_label=False, ext=None, fps=30, axis_sets=None, magnitudes=None, periods=None,
                      funcs=None, quantize_label=False):
  
  duration = 1 / bpm * num_beats * 60
  num_frames = int(duration * fps)

  if random_label:
    # Random label vec
    y = truncated_z_sample(1, gan_video_synth.vocab_size, truncation, int(datetime.now().strftime('%f')))
  else:
    # Indexed label vec
    y = np.zeros((1, gan_video_synth.vocab_size))
    for axis in classes:
      y[0, axis] = 1

  y = y / np.linalg.norm(y) * y_scale
  
  # Quantize y values to 0 and 1
  if quantize_label:
    y = (abs(y) > 0.5).astype(int)

  # Expand ys out to full shape
  ys = np.repeat(y, num_frames, axis=0)

  # Random z vec
  # Here the seed itself is a function of the current microsecond
  noise_seed_z = int(datetime.now().strftime('%f'))
  z0 = truncated_z_sample(1, gan_video_synth.dim_z, truncation, noise_seed_z)

  # Dimension sets to vary rhythmically; in [0, 128)
  if axis_sets is None:
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

  if magnitudes is None:
    magnitudes = [
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1,
      1
    ]

  if periods is None:
    periods = [
      1,
      1,
      2,
      2,
      4,
      4,
      8,
      8,
      16,
      16
    ]

  if funcs is None:
    funcs = [
      ramp,
      lambda x: ramp(x, phase=np.pi),
      ramp,
      np.sin,
      np.cos,
      np.sin,
      np.cos,
      np.sin,
      np.cos,
      np.sin
    ]

  zs = np.repeat(z0, num_frames, axis=0)

  for ax_set, mag, period, func in zip(axis_sets, magnitudes, periods, funcs):
    if ax_set is None or mag is None or period is None:
      continue

    for ax in ax_set:
      zs[:, ax] += func(np.linspace(0, 1.0 / period * num_beats * TAU, num=num_frames + 1)[:num_frames]) * mag

  # Generate images
  ims = gan_video_synth.sample(zs, ys, truncation=truncation, batch_size=1)
  # Transform
  ims = fit_to_ratio(ims)
  # Save numpy array
  np.save(os.path.join('npys', 'out.npy'), ims)
  # Save its hash, for cached reads
  with open(os.path.join('npys', 'last_hash.txt'), 'w') as f:
    f.write(str(hash(ims.tostring())) + '\n')

  # Save rendered animation
  if ext is not None:
    gan_video_synth.write_gif(ims, duration, out_dir='renders', ext=ext)


def _get_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--num-samples', default=1, type=int)
  parser.add_argument('--bpm', default=120, type=int)
  parser.add_argument('--num-beats', default=4, type=int)
  parser.add_argument('--classes', nargs='+', type=int, default=[309])
  parser.add_argument('--y-scale', default=0.9, type=float)
  parser.add_argument('--truncation', default=1, type=float)
  parser.add_argument('--random-label', action='store_true')
  parser.add_argument('--ext', default=None, help='Extension to save rendered animation. If not passed, then don\'t render and just save numpy array. Options are .gif, .mp4')
  parser.add_argument('--quantize-label', action='store_true')
  parser.add_argument('--model', choices=MODULE_PATH_MAPPING.keys(), default='512-deep')
  return parser


if __name__ == '__main__':
  args = _get_parser().parse_args()
  gan_video_synth = GanVideoSynth(model_string=args.model)

  # TODO generate multiple samples as a batch, not as a loop
  for _ in range(args.num_samples):
      generate_in_tempo(gan_video_synth, args.bpm, args.num_beats, args.classes, args.y_scale, args.truncation,
                        args.random_label, ext=args.ext, quantize_label=args.quantize_label)


