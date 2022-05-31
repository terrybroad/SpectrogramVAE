import os
import torch
import argparse
import torch.nn as nn
import numpy as np
import soundfile as sf

from torch import optim
from torchaudio.transforms import Spectrogram
from model import Encoder, Decoder
from util import *

def reconstruct_gen(encoder, decoder, args, specfunc, data):
  num_batches = np.ceil(len(data) / args.batch)
  batch_idxs = np.array_split(range(len(data)), num_batches)
  wavs = []
  for i, batch_idx in enumerate(batch_idxs):

    batch = data[batch_idx,:]
    x = torch.tensor(batch).to(args.device).transpose(1,3)
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

  final_wav = towave_from_z(np.array(assembled_spec),args,specfunc,args.out_name,args.out_dir,show=False, save=False)
  if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
  pathfin = f'{args.out_dir}/{args.out_name}'
  sf.write(f'{pathfin}.wav', final_wav, args.sr)

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
    parser.add_argument('--out_name', type=str, default="reconstructed_sample")
    parser.add_argument('--out_dir', type=str, default="generated")
    parser.add_argument('--track', type=str, default="")
    parser.add_argument('--save_spec', type=bool, default=False)
    parser.add_argument('--num_samples', type=int, default=20)
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

    #AUDIO TO CONVERT
    awv = audio_array_from_track(args.track)         #get waveform array from folder containing wav files
    track_spec = tospec(awv, args, specfunc)                 
    track_data = splitcut(track_spec, args)    

    reconstruct_gen(encoder, decoder, args, specfunc, track_data)