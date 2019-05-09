# coding=utf-8
# Copyright 2018 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common classes for automatic speech recognition (ASR) datasets.

The audio import uses sox to generate normalized waveforms, please install
it as appropriate (e.g. using apt-get or yum).
"""

import numpy as np

from tensor2tensor.data_generators import audio_encoder
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.layers import common_audio
from tensor2tensor.layers import common_layers
from tensor2tensor.layers import modalities
from tensor2tensor.utils import metrics

import tensorflow as tf

def time_warp(mel_fbanks, W=80):
    '''
    mel_fbanks: melspectrogram tensor.
        (1, timesteps or n or τ, number of mel freq bins or v, 1)
    W: int. time warp parameter.
    '''
    fbank_size = tf.shape(mel_fbanks)
    n, v = fbank_size[1], fbank_size[2]

    # Source
    pt = tf.random_uniform([], W, n-W, tf.int32) # radnom point along the time axis
    src_ctr_pt_freq = tf.range(v // 2)  # control points on freq-axis
    src_ctr_pt_time = tf.ones_like(src_ctr_pt_freq) * pt  # control points on time-axis
    src_ctr_pts = tf.stack((src_ctr_pt_time, src_ctr_pt_freq), -1)
    src_ctr_pts = tf.to_float(src_ctr_pts)

    # Destination
    w = tf.random_uniform([], -W, W, tf.int32)  # distance
    dest_ctr_pt_freq = src_ctr_pt_freq
    dest_ctr_pt_time = src_ctr_pt_time + w
    dest_ctr_pts = tf.stack((dest_ctr_pt_time, dest_ctr_pt_freq), -1)
    dest_ctr_pts = tf.to_float(dest_ctr_pts)

    # warp
    source_control_point_locations = tf.expand_dims(src_ctr_pts, 0)  # (1, v//2, 2)
    dest_control_point_locations = tf.expand_dims(dest_ctr_pts, 0)  # (1, v//2, 2)

    warped_image, _ = tf.contrib.image.sparse_image_warp(mel_fbanks, source_control_point_locations, dest_control_point_locations)
    return warped_image

def freq_mask(mel_fbanks, F=27):
    '''
    mel_fbanks: melspectrogram tensor.
        (1, timesteps or n, number of mel freq bins or v, 1)
    F: int. freqeuncy mask parameter.
    '''
    fbank_size = tf.shape(mel_fbanks)
    n, v = fbank_size[1], fbank_size[2]

    f = tf.random_uniform([], 0, F, tf.int32)
    f0 = tf.random_uniform([], 0, v-f, tf.int32)
    mask = tf.concat((tf.ones(shape=(1, n, v-f0-f, 1)),
                      tf.zeros(shape=(1, n, f, 1)),
                      tf.ones(shape=(1, n, f0, 1)),
                      ), 2)
    masked = mel_fbanks * mask
    return tf.to_float(masked)

def time_mask(mel_fbanks, T=100):
    '''
    mel_fbanks: melspectrogram tensor.
        (1, timesteps or n, number of mel freq bins or v, 1)
    T: int. time mask parameter.
    '''
    fbank_size = tf.shape(mel_fbanks)
    n, v = fbank_size[1], fbank_size[2]

    t = tf.random_uniform([], 0, T, tf.int32)
    t0 = tf.random_uniform([], 0, n-T, tf.int32)
    mask = tf.concat((tf.ones(shape=(1, n-t0-t, v, 1)),
                      tf.zeros(shape=(1, t, v, 1)),
                      tf.ones(shape=(1, t0, v, 1)),
                      ), 1)
    masked = mel_fbanks * mask
    return tf.to_float(masked)

class ByteTextEncoderWithEos(text_encoder.ByteTextEncoder):
  """Encodes each byte to an id and appends the EOS token."""

  def encode(self, s):
    return super(ByteTextEncoderWithEos, self).encode(s) + [text_encoder.EOS_ID]


class SpeechRecognitionProblem(problem.Problem):
  """Base class for speech recognition problems."""

  def hparams(self, defaults, model_hparams):
    p = model_hparams
    # Filterbank extraction in bottom instead of preprocess_example is faster.
    p.add_hparam("audio_preproc_in_bottom", False)
    # The trainer seems to reserve memory for all members of the input dict
    p.add_hparam("audio_keep_example_waveforms", False)
    p.add_hparam("audio_sample_rate", 16000)
    p.add_hparam("audio_preemphasis", 0.97)
    p.add_hparam("audio_dither", 1.0 / np.iinfo(np.int16).max)
    p.add_hparam("audio_frame_length", 25.0)
    p.add_hparam("audio_frame_step", 10.0)
    p.add_hparam("audio_lower_edge_hertz", 20.0)
    p.add_hparam("audio_upper_edge_hertz", 8000.0)
    p.add_hparam("audio_num_mel_bins", 80)
    p.add_hparam("audio_add_delta_deltas", True)
    p.add_hparam("num_zeropad_frames", 250)

    p = defaults
    p.modality = {"inputs": modalities.SpeechRecognitionModality,
                  "targets": modalities.SymbolModality}
    p.vocab_size = {"inputs": None,
                    "targets": 256}

  @property
  def is_character_level(self):
    return True

  @property
  def input_space_id(self):
    return problem.SpaceID.AUDIO_SPECTRAL

  @property
  def target_space_id(self):
    return problem.SpaceID.EN_CHR

  def feature_encoders(self, _):
    return {
        "inputs": None,  # Put None to make sure that the logic in
                         # decoding.py doesn't try to convert the floats
                         # into text...
        "waveforms": audio_encoder.AudioEncoder(),
        "targets": ByteTextEncoderWithEos(),
    }

  def example_reading_spec(self):
    data_fields = {
        "waveforms": tf.VarLenFeature(tf.float32),
        "targets": tf.VarLenFeature(tf.int64),
    }

    data_items_to_decoders = None

    return data_fields, data_items_to_decoders

  def preprocess_example(self, example, mode, hparams):
    p = hparams
    if p.audio_preproc_in_bottom:
      example["inputs"] = tf.expand_dims(
          tf.expand_dims(example["waveforms"], -1), -1)
    else:
      waveforms = tf.expand_dims(example["waveforms"], 0)
      mel_fbanks = common_audio.compute_mel_filterbank_features(
          waveforms,
          sample_rate=p.audio_sample_rate,
          dither=p.audio_dither,
          preemphasis=p.audio_preemphasis,
          frame_length=p.audio_frame_length,
          frame_step=p.audio_frame_step,
          lower_edge_hertz=p.audio_lower_edge_hertz,
          upper_edge_hertz=p.audio_upper_edge_hertz,
          num_mel_bins=p.audio_num_mel_bins,
          apply_mask=False)
      if p.audio_add_delta_deltas:
        mel_fbanks = common_audio.add_delta_deltas(mel_fbanks)
      fbank_size = common_layers.shape_list(mel_fbanks)
      assert fbank_size[0] == 1

      # This replaces CMVN estimation on data
      var_epsilon = 1e-09
      mean = tf.reduce_mean(mel_fbanks, keepdims=True, axis=1)
      variance = tf.reduce_mean(tf.square(mel_fbanks - mean),
                                keepdims=True, axis=1)
      mel_fbanks = (mel_fbanks - mean) * tf.rsqrt(variance + var_epsilon)

      ######### specaugment added by kyubyong #########
      if mode == tf.estimator.ModeKeys.TRAIN:
          # mel_fbanks = time_warp(mel_fbanks)
          mel_fbanks = freq_mask(mel_fbanks)
          mel_fbanks = time_mask(mel_fbanks)

      ######### /specaugment added by kyubyong #########

      # Later models like to flatten the two spatial dims. Instead, we add a
      # unit spatial dim and flatten the frequencies and channels.
      example["inputs"] = tf.concat([
          tf.reshape(mel_fbanks, [fbank_size[1], fbank_size[2], fbank_size[3]]),
          tf.zeros((p.num_zeropad_frames, fbank_size[2], fbank_size[3]))], 0)

    if not p.audio_keep_example_waveforms:
      del example["waveforms"]
    return super(SpeechRecognitionProblem, self
                ).preprocess_example(example, mode, hparams)

  def eval_metrics(self):
    defaults = super(SpeechRecognitionProblem, self).eval_metrics()
    return defaults + [
        metrics.Metrics.EDIT_DISTANCE,
        metrics.Metrics.WORD_ERROR_RATE
    ]
