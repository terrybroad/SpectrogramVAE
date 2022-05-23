import os
import math
import util
import yaml
import heapq
import torch
import argparse
import torchaudio
import librosa
import librosa.display
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from numpy import linspace
from glob import glob
from torchvision import utils
from torch import autograd, optim
from tqdm import tqdm
from functools import partial
from torchaudio.transforms import Spectrogram
from torch.utils.data import DataLoader
from model import Encoder, Decoder
from util import *

# torch.set_default_tensor_type('torch.cuda.FloatTensor')

#Generate one-shot samples from latent space with random or manual seed
def one_shot_gen(decoder, args, specfunc,  num_samples=1, use_seed=False, seed=1001, z_scale=-2.2, save=True, name="one_shot", path="/home/terence/repos/SpectrogramVAE/sample"):
    num_samples_to_generate =   num_samples
    _use_seed = use_seed
    _seed = seed
    scale_z_vectors = z_scale
    save_audio = save
    audio_name = name
    audio_save_directory = path

    y = np.random.randint(0, 2**32-1)  # generated random int to pass and convert into vector
    i=0
    while i < num_samples_to_generate:
      if not _use_seed:
        z = generate_random_z_vect(y, num_samples_to_generate,scale=scale_z_vectors)
      else:
        z = generate_random_z_vect(_seed, num_samples_to_generate,scale=scale_z_vectors)
      z_sample = np.array(vae.sample_from_latent_space(z))
      towave_from_z(z_sample[i], args, specfunc, name=f'{audio_name}_{i}',path=audio_save_directory,show=False, save=save_audio)
      i+=1

    if not _use_seed:
      print("Generated from seed:", y)
    else:
      print("Generated from seed:", _seed)

#Generate arbitrary long audio from latent space with random or custom seed using uniform, Perlin or fractal noise
def noise_gen(decoder, args, specfunc, num_samples=1, _noise_type="fractal", _use_seed=False, _seed=1001, z_scale=2.5, save=False, name="noise_generation", path="/home/terence/repos/SpectrogramVAE/sample"):
    num_seeds_to_generate = num_samples
    noise_type = _noise_type #params are ["uniform", "perlin", "fractal"]
    use_seed = _use_seed
    seed = _seed
    scale_z_vectors =  z_scale
    save_audio = save
    audio_name = name
    audio_save_directory = path


    y = np.random.randint(0, 2**32-1)                         # generated random int to pass and convert into vector
    if not use_seed:
      if noise_type == "uniform":
        z = generate_random_z_vect(y, num_seeds_to_generate,scale_z_vectors,args.vector_dim)            # vectors to input into latent space
      if noise_type == "perlin":
        z = generate_z_vect_from_perlin_noise(y, num_seeds_to_generate,scale_z_vectors,args.vector_dim)            # vectors to input into latent space
      if noise_type == "fractal":
        z = generate_z_vect_from_fractal_noise(y, num_seeds_to_generate,scale_z_vectors,args.vector_dim)            # vectors to input into latent space
    if use_seed:
      if noise_type == "uniform":
        z = generate_random_z_vect(seed, num_seeds_to_generate,scale_z_vectors,args.vector_dim)            # vectors to input into latent space
      if noise_type == "perlin":
        z = generate_z_vect_from_perlin_noise(seed, num_seeds_to_generate,scale_z_vectors,args.vector_dim)            # vectors to input into latent space
      if noise_type == "fractal":
        z = generate_z_vect_from_fractal_noise(seed, num_seeds_to_generate,scale_z_vectors,args.vector_dim)            # vectors to input into latent space
    z = torch.tensor(z, dtype=torch.float32).to('cuda')
    gen = decoder(z)
    utils.save_image(gen, f'interp/random_noise_output.png',
      nrow=8,
      normalize=True,
      range=(-1, 1))
    z_sample = gen.transpose(1,3).cpu().detach().numpy()
    # z_sample = np.array(vae.sample_from_latent_space(z))
    assembled_spec = testass(z_sample)
    towave_from_z(assembled_spec,args,specfunc,audio_name,audio_save_directory,show=False,save=save_audio)

    if not use_seed:
      print("Generated from seed:", y)
    else:
      print("Generated from seed:", seed)

#Interpolate between two seeds for n-amount of steps
def interp_gen(decoder, args, specfunc, num_samples=1, _use_seed=False, _seed=1001, interp_steps=5, z_scale=-2.2, interp_scale=1.2, save=False, name="one_shot", path="/content/"):
    use_seed = _use_seed #@param {type:"boolean"}
    seed =  _seed #@param {type:"slider", min:0, max:4294967295, step:1}
    num_interpolation_steps = interp_steps#@param {type:"integer"}
    scale_z_vectors =  z_scale #@param {type:"slider", min:-5.0, max:5.0, step:0.1}
    scale_interpolation_ratio =  interp_scale #@param {type:"slider", min:-5.0, max:5.0, step:0.1}
    save_audio = save #@param {type:"boolean"}
    audio_name = name #@param {type:"string"}
    audio_save_directory = path #@param {type:"string"}

    # generate points in latent space as input for the generator
    def generate_latent_points(latent_dim, n_samples, n_classes=10):
      # generate points in the latent space
      x_input = randn(latent_dim * n_samples)
      # reshape into a batch of inputs for the network
      z_input = x_input.reshape(n_samples, latent_dim)
      return z_input

    # uniform interpolation between two points in latent space
    def interpolate_points(p1, p2,scale, n_steps=10):
      # interpolate ratios between the points
      ratios = linspace(-scale, scale, num=n_steps)
      # linear interpolate vectors
      vectors = list()
      for ratio in ratios:
        v = (1.0 - ratio) * p1 + ratio * p2
        vectors.append(v)
      return asarray(vectors)

    y = np.random.randint(0, 2**32-1)
    if not use_seed:
      pts = generate_random_z_vect(y,num_samples,scale_z_vectors, args.vector_dim)
    else:
      pts = generate_random_z_vect(seed,num_samples,scale_z_vectors,args.vector_dim)

    # interpolate points in latent space
    interpolated = interpolate_points(pts[0], pts[1], scale_interpolation_ratio, num_interpolation_steps)
    #print(np.shape(interpolated))
    interpolated = torch.tensor(interpolated, dtype=torch.float32).to('cuda')
    interp = decoder(interpolated)
    utils.save_image(interp, f'interp/random_output.png',
      nrow=8,
      normalize=True,
      range=(-1, 1))
    interp = interp.transpose(1,3).cpu().detach().numpy()
    assembled_spec = testass(interp)
    towave_from_z(assembled_spec,args,specfunc,audio_name,audio_save_directory,show=False, save=save_audio)
    #print(np.shape(assembled_spec))

    if not use_seed:
      print("Generated from seed:", y)
    else:
      print("Generated from seed:", seed)

def reconstruct_gen(encoder, decoder, args, specfunc, data, path="generated"):
  num_batches = np.ceil(len(data) / args.batch)
  batch_idxs = np.array_split(range(len(data)), num_batches)
  wavs = []
  for i, batch_idx in enumerate(batch_idxs):

    batch = data[batch_idx,:]
    x = torch.tensor(batch).to('cuda').transpose(1,3)
    z, kld, mu = encoder(x)
    _x = decoder(mu)    
    recon_np = _x.transpose(1,3).cpu().detach().numpy()
    spec = testass(recon_np)
    if i == 0:
      assembled_spec = spec
    else:
      assembled_spec = np.concatenate((assembled_spec,spec), axis = 1)
      print(spec.shape)
      print(assembled_spec.shape)
    wav = towave_from_z(np.array(spec),args,specfunc,args.output_name,path,show=False, save=False)
    # print("SHAPE: " + str(wav.shape))
    wavs.append(wav)
  final_wav = np.concatenate(wavs)
  other_wav = towave_from_z(np.array(assembled_spec),args,specfunc,args.output_name,path,show=False, save=False)
  pathfin = f'{path}/{args.output_name}'
  sf.write(f'{pathfin}.wav', final_wav, args.sr)
  sf.write(f'{pathfin}_assembled.wav', final_wav, args.sr)

if __name__ == "__main__":
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--vector_dim', type=int, default=128)
    parser.add_argument('--maxiter', type=int, default=100001)
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--min_db', type=int, default=-100)
    parser.add_argument('--ref_db', type=int, default=20)
    parser.add_argument('--spec_split', type=int, default=1)
    parser.add_argument('--shape', type=int, default=128)
    parser.add_argument('--beta', type=int, default=0.001)
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--run_name', type=str, default="test")
    parser.add_argument('--save_dir', type=str, default="ckpt")
    parser.add_argument('--output_name', type=str, default="generated_sample")
    parser.add_argument('--noise', type=bool, default=False)
    parser.add_argument('--track', type=str, default="")
    parser.add_argument('--tracks_p_batch', type=int, default=100)
    parser.add_argument('--save_spec', type=bool, default=False)
    parser.add_argument('--num_samples', type=int, default=20)
    args = parser.parse_args()

    encoder = Encoder(args.vector_dim)
    decoder = Decoder(args.vector_dim)

    e_optim = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0, 0.99))
    d_optim = optim.Adam(decoder.parameters(), lr=args.lr, betas=(0, 0.99))
    criterion = nn.MSELoss()

    checkpoint = torch.load(args.ckpt)

    encoder.load_state_dict(checkpoint['encoder'])
    encoder.to('cuda')
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to('cuda')

    specobj = Spectrogram(n_fft=4*args.hop, win_length=4*args.hop, hop_length=args.hop, pad=0, power=2, normalized=False)
    specfunc = specobj.forward

    #AUDIO TO CONVERT
    awv = audio_array_from_track(args.track)         #get waveform array from folder containing wav files
    print(awv)
    track_spec = tospec(awv, args, specfunc)   
    print(track_spec)                 
    track_data = splitcut(track_spec, args)    
    print(track_data)

    # one_shot_gen(decoder, args, specfunc, num_samples=10, name="amazondotcom_test")
    # noise_gen(decoder, args, specfunc, num_samples=64,_use_seed=False,_noise_type="perlin", z_scale=2.5, name="uniform_test2s", save=True)
    # interp_gen(decoder, args, specfunc, num_samples=10, _use_seed=False, _seed=1001, interp_steps=64, z_scale=-1.5, interp_scale=5.0, save=True, name="interp_audio", path="sample")
    reconstruct_gen(encoder, decoder, args, specfunc, track_data, path="generated")