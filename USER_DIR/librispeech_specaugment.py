
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

"""Librispeech dataset."""

import os
import tarfile
from tensor2tensor.data_generators import generator_utils
from tensor2tensor.data_generators import problem
# from tensor2tensor.data_generators import speech_recognition
from tensor2tensor.utils import registry

import tensorflow as tf

from USER_DIR import speech_recognition

_LIBRISPEECH_TRAIN_DATASETS = [
    [
        "http://www.openslr.org/resources/12/train-clean-100.tar.gz",  # pylint: disable=line-too-long
        "train-clean-100"
    ],
    [
        "http://www.openslr.org/resources/12/train-clean-360.tar.gz",
        "train-clean-360"
    ],
    [
        "http://www.openslr.org/resources/12/train-other-500.tar.gz",
        "train-other-500"
    ],
]
_LIBRISPEECH_DEV_DATASETS = [
    [
        "http://www.openslr.org/resources/12/dev-clean.tar.gz",
        "dev-clean"
    ],
    [
        "http://www.openslr.org/resources/12/dev-other.tar.gz",
        "dev-other"
    ],
]
_LIBRISPEECH_TEST_DATASETS = [
    [
        "http://www.openslr.org/resources/12/test-clean.tar.gz",
        "test-clean"
    ],
    [
        "http://www.openslr.org/resources/12/test-other.tar.gz",
        "test-other"
    ],
]


def _collect_data(directory, input_ext, transcription_ext):
  """Traverses directory collecting input and target files."""
  # Directory from string to tuple pair of strings
  # key: the filepath to a datafile including the datafile's basename. Example,
  #   if the datafile was "/path/to/datafile.wav" then the key would be
  #   "/path/to/datafile"
  # value: a pair of strings (media_filepath, label)
  data_files = dict()
  for root, _, filenames in os.walk(directory):
    transcripts = [filename for filename in filenames
                   if transcription_ext in filename]
    for transcript in transcripts:
      transcript_path = os.path.join(root, transcript)
      with open(transcript_path, "r") as transcript_file:
        for transcript_line in transcript_file:
          line_contents = transcript_line.strip().split(" ", 1)
          media_base, label = line_contents
          key = os.path.join(root, media_base)
          assert key not in data_files
          media_name = "%s.%s"%(media_base, input_ext)
          media_path = os.path.join(root, media_name)
          data_files[key] = (media_base, media_path, label)
  return data_files


@registry.register_problem()
class LibrispeechSpecaugment(speech_recognition.SpeechRecognitionProblem):
  """Problem spec for Librispeech using clean and noisy data."""

  # Select only the clean data
  TRAIN_DATASETS = _LIBRISPEECH_TRAIN_DATASETS[:1]
  DEV_DATASETS = _LIBRISPEECH_DEV_DATASETS[:1]
  TEST_DATASETS = _LIBRISPEECH_TEST_DATASETS[:1]

  @property
  def num_shards(self):
    return 100

  @property
  def use_subword_tokenizer(self):
    return False

  @property
  def num_dev_shards(self):
    return 1

  @property
  def num_test_shards(self):
    return 1

  @property
  def use_train_shards_for_dev(self):
    """If true, we only generate training data and hold out shards for dev."""
    return False

  def generator(self, data_dir, tmp_dir, datasets,
                eos_list=None, start_from=0, how_many=0):
    del eos_list
    i = 0
    for url, subdir in datasets:
      filename = os.path.basename(url)
      compressed_file = generator_utils.maybe_download(tmp_dir, filename, url)

      read_type = "r:gz" if filename.endswith("tgz") else "r"
      with tarfile.open(compressed_file, read_type) as corpus_tar:
        # Create a subset of files that don't already exist.
        #   tarfile.extractall errors when encountering an existing file
        #   and tarfile.extract is extremely slow
        members = []
        for f in corpus_tar:
          if not os.path.isfile(os.path.join(tmp_dir, f.name)):
            members.append(f)
        corpus_tar.extractall(tmp_dir, members=members)

      raw_data_dir = os.path.join(tmp_dir, "LibriSpeech", subdir)
      data_files = _collect_data(raw_data_dir, "flac", "txt")
      data_pairs = data_files.values()

      encoders = self.feature_encoders(data_dir)
      audio_encoder = encoders["waveforms"]
      text_encoder = encoders["targets"]

      for utt_id, media_file, text_data in sorted(data_pairs)[start_from:]:
        if how_many > 0 and i == how_many:
          return
        i += 1
        wav_data = audio_encoder.encode(media_file)
        spk_id, unused_book_id, _ = utt_id.split("-")
        yield {
            "waveforms": wav_data,
            "waveform_lens": [len(wav_data)],
            "targets": text_encoder.encode(text_data),
            "raw_transcript": [text_data],
            "utt_id": [utt_id],
            "spk_id": [spk_id],
        }

  def generate_data(self, data_dir, tmp_dir, task_id=-1):
    train_paths = self.training_filepaths(
        data_dir, self.num_shards, shuffled=False)
    dev_paths = self.dev_filepaths(
        data_dir, self.num_dev_shards, shuffled=False)
    test_paths = self.test_filepaths(
        data_dir, self.num_test_shards, shuffled=True)

    generator_utils.generate_files(
        self.generator(data_dir, tmp_dir, self.TEST_DATASETS), test_paths)

    if self.use_train_shards_for_dev:
      all_paths = train_paths + dev_paths
      generator_utils.generate_files(
          self.generator(data_dir, tmp_dir, self.TRAIN_DATASETS), all_paths)
      generator_utils.shuffle_dataset(all_paths)
    else:
      generator_utils.generate_dataset_and_shuffle(
          self.generator(data_dir, tmp_dir, self.TRAIN_DATASETS), train_paths,
          self.generator(data_dir, tmp_dir, self.DEV_DATASETS), dev_paths)

