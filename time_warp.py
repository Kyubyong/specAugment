import numpy as np
import tensorflow as tf
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import matplotlib.pyplot as plt
# Settings
audio_f = librosa.util.example_audio_file()  # file path
sr = 22050  # sample rate
file_format = "ogg"  # or wav, ...
num_channels = 1  # mono
n_fft = 2048  # number of fft length. 2**n
win_length = 1000  # window length <= n_fft
hop_length = 250  # hopping step
n_mels = 80  # number of mels
n_mfccs = 40  # number of mfccs
preemp = .97  # preemphasis rate
n_iter = 50  # Griffin-Lim's law


def data_load(audio_f, sr=22050, file_format="wav", num_channels=1):
    audio_binary = tf.read_file(audio_f)
    y = tf.contrib.ffmpeg.decode_audio(audio_binary, file_format, sr, num_channels)
    return tf.squeeze(y, 1), sr

​y, sr = data_load(audio_f, sr, file_format, num_channels)

y = y[:100000]


def get_spectrograms(y, sr=22050, n_fft=2048, win_length=2048, hop_length=512, n_mels=None, power=1):
    linear = tf.contrib.signal.stft(y, frame_length=win_length, frame_step=hop_length,
                                    fft_length=n_fft)  # linear spectrogram
    mag = tf.abs(linear)  # magnitude

    if n_mels is not None:
        mel_basis = tf.convert_to_tensor(librosa.filters.mel(sr, n_fft, n_mels), tf.float32)
        mel = tf.matmul(mag ** power, mel_basis, transpose_b=True)  # (t, n_mels)
    else:
        mel = None

    return linear, mag, mel


linear, mag, mel = get_spectrograms(y, sr, n_fft, win_length, hop_length, n_mels)
print(linear.eval().shape, mag.eval().shape, mel.eval().shape)
_mel = mel.eval()
_mel
plt.imshow(_mel.T)
librosa.display.specshow(librosa.power_to_db(_mel.T, ref=np.max), fmax=8000)
# plt.colorbar(format='%+2.0f dB')
# plt.tight_layout()
image = tf.expand_dims(tf.expand_dims(tf.transpose(mel), 0), -1)
image
_, freq, t, _ = image.get_shape().as_list()
y = np.arange(freq + 1)
y
W = 80
x = np.arange(W, t - W + 1)
x
distance = np.random.randint(-W, W, x.shape, np.int32)
distance


def get_pairs(y, x):
    yx = []
    for yy in y:
        for xx in x:
            yx.append([yy, xx])
            #     yx_ = np.array(yx_)
    return np.array(yx, np.int32)


src_yx = get_pairs(y, x)
src_yx
tgt_yx = get_pairs(y, x + distance)
tgt_yx
source_control_point_locations = tf.convert_to_tensor(np.expand_dims(src_yx, 0), tf.float32)
dest_control_point_locations = tf.convert_to_tensor(np.expand_dims(tgt_yx, 0), tf.float32)
dest_control_point_locations
image
warped_image, _ = tf.contrib.image.sparse_image_warp(
    image,
    source_control_point_locations,
    dest_control_point_locations,
    interpolation_order=2,
    regularization_weight=0.0,
    num_boundary_points=0,
    name='sparse_image_warp'
)
​
warped_image
sess.run(tf.global_variables_initializer())
sess.run(warped_image)
​
