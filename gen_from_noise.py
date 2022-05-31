import os
import torch
import argparse
import numpy as np

from torchaudio.transforms import Spectrogram
from model import Encoder, Decoder
from util import *

#Generate arbitrary long audio from latent space with random or custom seed using uniform, Perlin or fractal noise
def noise_gen(decoder, args, specfunc, num_samples=1, _use_seed=False, _seed=1001, z_scale=2.5):
    num_seeds_to_generate = num_samples
    use_seed = _use_seed
    seed = _seed
    scale_z_vectors =  z_scale

    y = np.random.randint(0, 2**32-1)                         # generated random int to pass and convert into vector
    if not use_seed:
      z = generate_random_z_vect(y, num_seeds_to_generate,scale_z_vectors,args.vector_dim)            # vectors to input into latent space
    if use_seed:
      z = generate_random_z_vect(seed, num_seeds_to_generate,scale_z_vectors,args.vector_dim)            # vectors to input into latent space
    z = torch.tensor(z, dtype=torch.float32).to(args.device)
    gen = decoder(z)
    z_sample = gen.transpose(1,3).cpu().detach().numpy()
    # z_sample = np.array(vae.sample_from_latent_space(z))
    assembled_spec = testass(z_sample)
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
    parser.add_argument('--out_name', type=str, default="noise_sample")
    parser.add_argument('--out_dir', type=str, default="generated")
    parser.add_argument('--save_spec', type=bool, default=False)
    parser.add_argument('--num_samples', type=int, default=20)
    parser.add_argument('--z_scale', type=float, default=7.5)
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

    noise_gen(decoder, args, specfunc, num_samples=args.num_samples,_use_seed=False, z_scale=args.z_scale)
   