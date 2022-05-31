import os
import torch
import argparse
import torch.nn as nn
import numpy as np

from torch import optim
from torchaudio.transforms import Spectrogram
from model import Encoder, Decoder
from util import *

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
    interpolated = torch.tensor(interpolated, dtype=torch.float32).to(args.device)
    interp = decoder(interpolated)
    interp = interp.transpose(1,3).cpu().detach().numpy()
    assembled_spec = testass(interp)
    final_wav = towave_from_z(np.array(assembled_spec),args,specfunc,args.out_name,args.out_dir,show=False, save=False)
    if not os.path.exists(args.out_dir):
      os.makedirs(args.out_dir)
    pathfin = f'{args.out_dir}/{args.out_name}'
    sf.write(f'{pathfin}.wav', final_wav, args.sr)

    if not use_seed:
      print("Generated from seed:", y)
    else:
      print("Generated from seed:", seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--vector_dim', type=int, default=512)
    parser.add_argument('--maxiter', type=int, default=100001)
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--min_db', type=int, default=-100)
    parser.add_argument('--ref_db', type=int, default=20)
    parser.add_argument('--spec_split', type=int, default=1)
    parser.add_argument('--shape', type=int, default=128)
    parser.add_argument('--beta', type=int, default=0.001)
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--out_name', type=str, default="interp_sample")
    parser.add_argument('--out_dir', type=str, default="generated")
    parser.add_argument('--save_spec', type=bool, default=False)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--interp_steps', type=int, default=64)
    parser.add_argument('--interp_scale', type=int, default=5)
    parser.add_argument('--z_scale', type=float, default=-1.5)
    args = parser.parse_args()

    encoder = Encoder(args.vector_dim)
    decoder = Decoder(args.vector_dim)
    checkpoint = torch.load(args.ckpt)

    encoder.load_state_dict(checkpoint['encoder'])
    encoder.to(args.device)
    decoder.load_state_dict(checkpoint['decoder'])
    decoder.to(args.device)

    specobj = Spectrogram(n_fft=4*args.hop, win_length=4*args.hop, hop_length=args.hop, pad=0, power=2, normalized=False)
    specfunc = specobj.forward

    interp_gen(decoder, args, specfunc, num_samples=args.num_samples, _use_seed=False, _seed=1001, interp_steps=args.interp_steps, z_scale=args.z_scale, interp_scale=args.interp_scale, save=True, name="interp_audio", path="sample")
