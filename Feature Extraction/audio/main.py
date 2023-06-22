DESC = """
Script to extract handcrafted and deep audio features.
"""

import os
import numpy as np
import librosa
import Speech_silence_vad
import torch
import fairseq
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description=DESC)
parser.add_argument('--root', required=True, type=str, help='Folder where the individual audio clips are placed.')
parser.add_argument('--deep_ft_out', required=True, type=str, help='Folder where the wav2vec features will be saved (in npy format).')
parser.add_argument('--hc_ft_out', required=True, type=str, help='Folder where the handcrafted features will be saved (in npy format).')
parser.add_argument('--checkpoint', required=True, type=str, help='Checkpoint path of a fairseq wav2vec model (for more info, see github.com/facebookresearch/fairseq/blob/main/examples/wav2vec/README.md#wav2vec).')
parser.add_argument('--deep_csv', default=None, help='File path to save the deep features in csv format. If not given, no csv is produced.')
parser.add_argument('--hc_csv', default=None, help='File path to save the handcrafted features in csv format. If not given, no csv is produced.')
args = parser.parse_args()

# Wav2Vec audio features
def DeepAudioFeatures(audio_path, W2Vmodel):
    speech, sr = librosa.load(audio_path, sr=None)
    speechVAD = Speech_silence_vad.silence_handler(speech, sr, fl=int(20 / 1000 * sr), fs=int(5 / 1000 * sr),
                                                   max_thres_below=40, min_thres=-55, shortest_len_in_ms=50,
                                                   flag_output=1)
    model = W2Vmodel[0]
    model.eval()
    z = model.feature_extractor(torch.from_numpy(speechVAD).unsqueeze(0))
    z = model.feature_aggregator(z)
    w2v = z.cpu().detach().numpy()
    w2v_mean = np.mean(w2v, axis=2)
    w2v_std = np.std(w2v, axis=2)
    return w2v_mean, w2v_std, w2v

# Handcrafted audio features
def HandcraftedAudioFeatures(audio_path):
    speech, sr = librosa.load(audio_path, sr=None)
    speechVAD = Speech_silence_vad.silence_handler(speech, sr, fl=int(20 / 1000 * sr), fs=int(5 / 1000 * sr),
                                                   max_thres_below=40, min_thres=-55, shortest_len_in_ms=50,
                                                   flag_output=1)
    mfcc = librosa.feature.mfcc(y=speechVAD, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta_mean = np.mean(mfcc_delta, axis=1)
    mfcc_delta_std = np.std(mfcc_delta, axis=1)
    mfcc_delta_delta = librosa.feature.delta(mfcc_delta)
    mfcc_delta_delta_mean = np.mean(mfcc_delta_delta, axis=1)
    mfcc_delta_delta_std = np.std(mfcc_delta_delta, axis=1)
    cent = librosa.feature.spectral_centroid(y=speechVAD, sr=sr)
    cent_mean = np.mean(cent, axis=1)
    cent_std = np.std(cent, axis=1)
    spec_bw = librosa.feature.spectral_bandwidth(y=speechVAD, sr=sr)
    spec_bw_mean = np.mean(spec_bw, axis=1)
    spec_bw_std = np.std(spec_bw, axis=1)
    S = np.abs(librosa.stft(speechVAD))
    contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
    contrast_mean = np.mean(contrast, axis=1)
    contrast_std = np.std(contrast, axis=1)
    flatness = librosa.feature.spectral_flatness(y=speechVAD)
    flatness_mean = np.mean(flatness, axis=1)
    flatness_std = np.std(flatness, axis=1)
    rolloff = librosa.feature.spectral_rolloff(y=speechVAD, sr=sr, roll_percent=0.85)
    rolloff_mean = np.mean(rolloff, axis=1)
    rolloff_std = np.std(rolloff, axis=1)
    y = librosa.effects.harmonic(speechVAD)
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    tonnetz_mean = np.mean(tonnetz, axis=1)
    tonnetz_std = np.std(tonnetz, axis=1)
    zc = librosa.feature.zero_crossing_rate(speechVAD)
    zc_mean = np.mean(zc, axis=1)
    zc_std = np.std(zc, axis=1)
    tempogram = librosa.feature.tempogram(y=speechVAD, sr=sr)
    tgr = librosa.feature.tempogram_ratio(tg=tempogram, sr=sr)
    tgr_mean = np.mean(tgr, axis=1)
    tgr_std = np.std(tgr, axis=1)

    hc = np.concatenate([mfcc_mean,
                         mfcc_std,
                         mfcc_delta_mean,
                         mfcc_delta_std,
                         mfcc_delta_delta_mean,
                         mfcc_delta_delta_std,
                         cent_mean, cent_std,
                         spec_bw_mean, spec_bw_std,
                         flatness_mean, flatness_std,
                         contrast_mean, contrast_std,
                         rolloff_mean,
                         rolloff_std,
                         tonnetz_mean,
                         tonnetz_std,
                         zc_mean,
                         zc_std,
                         tgr_mean,
                         tgr_std
                         ], axis=0)

    return hc

# Nevermind that, just read the files off the folder instead
root_folder = args.root
file_names = os.listdir(root_folder)
file_paths = [os.path.join(root_folder, f) for f in os.listdir(root_folder)]

# Process each audio file and save it in the output folder
cp_path = args.checkpoint
model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([cp_path])

output_folder = args.deep_ft_out
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

data = []
for audio_file in file_paths:
    z = DeepAudioFeatures(audio_file, model)
    z0 = z[0]
    z1 = z[1]
    z2 = z[2]
    output_path = os.path.join(output_folder, os.path.basename(audio_file))
    np.save(output_path, z2)
    z = np.concatenate([z0,z1], axis=0)
    z_list = z.tolist()
    data.append(z_list)

# done, package into dataframe and save as csv
if args.deep_csv is not None:
    outpath = args.deep_csv
    df = pd.DataFrame(data=data, index=file_names)
    print(f'Saving to {outpath}')
    df.to_csv(outpath, header=False)

output_folder = args.hc_ft_out
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

data = []
for audio_file in file_paths:
    z = HandcraftedAudioFeatures(audio_file)
    output_path = os.path.join(output_folder, os.path.basename(audio_file))
    np.save(output_path, z)
    z_list = z.tolist()
    data.append(z_list)

if args.hc_csv is not None:
    # done, package into dataframe and save as csv
    outpath = args.hc_csv
    df = pd.DataFrame(data=data, index=file_names)
    print(f'Saving to {outpath}')
    df.to_csv(outpath, header=False)