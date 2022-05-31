import os
import math
import torch
import argparse
import torch.nn as nn
import numpy as np

from torchvision import utils
from torch import optim
from tqdm import tqdm
from torchaudio.transforms import Spectrogram
from torch.utils.data import DataLoader
from model import Encoder, Decoder
from util import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--vector_dim', type=int, default=512)
    parser.add_argument('--iter_p_batch', type=int, default=1000)
    parser.add_argument('--tracks_p_batch', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--hop', type=int, default=512)
    parser.add_argument('--sr', type=int, default=44100)
    parser.add_argument('--min_db', type=int, default=-100)
    parser.add_argument('--ref_db', type=int, default=20)
    parser.add_argument('--spec_split', type=int, default=1)
    parser.add_argument('--shape', type=int, default=128)
    parser.add_argument('--beta', type=float, default=0.001)
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--data', type=str, default="/home/terence/Music/bach_wavs")
    parser.add_argument('--run_name', type=str, default="test")
    parser.add_argument('--save_dir', type=str, default="ckpt")
    args = parser.parse_args()

    encoder = Encoder(args.vector_dim)
    decoder = Decoder(args.vector_dim)

    e_optim = optim.Adam(encoder.parameters(), lr=args.lr, betas=(0, 0.99))
    d_optim = optim.Adam(decoder.parameters(), lr=args.lr, betas=(0, 0.99))
    criterion = nn.MSELoss()

    if args.ckpt != "":
      state_dict = torch.load(args.ckpt, map_location=args.device)
      # state_dict['encoder'].pop('fc.weight',None)
      # state_dict['encoder'].pop('fc.bias',None)
      new_state_dict_e = encoder.state_dict()
      new_state_dict_e.update(state_dict['encoder'])
      encoder.load_state_dict(new_state_dict_e)

      new_state_dict_d = decoder.state_dict()
      # state_dict['decoder'].pop('decoder_input.weight',None)
      # state_dict['decoder'].pop('decoder_input.bias',None)
      new_state_dict_d.update(state_dict['decoder'])
      decoder.load_state_dict(new_state_dict_d)

    encoder.to(args.device)
    decoder.to(args.device)


    specobj = Spectrogram(n_fft=4*args.hop, win_length=4*args.hop, hop_length=args.hop, pad=0, power=2, normalized=False)
    specfunc = specobj.forward


    dataset = AudioData(args.data)

    dataloader = DataLoader(dataset, batch_size=args.tracks_p_batch, collate_fn=collate_list, shuffle=True, num_workers=0)

    it_count = 0
    
    with tqdm(total=args.num_epochs) as pbar:
      
      for i in range(args.num_epochs):
        
        for index, sample in enumerate(dataloader):
          
          awv = audio_array_from_batch(sample)
          aspec = tospec(awv, args, specfunc)                        #get spectrogram array
          adata = splitcut(aspec, args)      

          for j in range(args.iter_p_batch):
              e_optim.zero_grad()
              d_optim.zero_grad()

              batch = adata[np.random.randint(adata.shape[0], size=args.batch), :]
              
              x = torch.tensor(batch).to(args.device).transpose(1,3)
              z, kld, mu = encoder(x)
              _x = decoder(z)

              recon_loss = criterion(x, _x)

              #TAKE THE LOG TO TRY AND OVERCOME POSTERIOR COLLAPSE
              #ADD ONE TO AVOID NEGATIVE NUMBERS
              kld = torch.log(kld + torch.tensor(1).detach())

              if math.isinf(kld.item()):
                kld = torch.Tensor([[0]]).cuda().detach()

              loss = recon_loss + kld * args.beta
              print("epoch: "+ str(i) + ", iter: "+str(it_count)+ ", total_loss: "+str(loss.item())+", recon_loss: " + str(recon_loss.item()) + ", kld: "+str(kld.item()))
              loss.backward()
              e_optim.step()
              d_optim.step()

              if not os.path.exists('sample'):
                os.makedirs('sample')

              if it_count % 2500 == 0:
                utils.save_image(x, f'sample/{str(it_count).zfill(6)}_input.png',
                  nrow=8,
                  normalize=True,
                  range=(-1, 1))
                # save_spec_as_image(x[:,0], f'sample/{str(i).zfill(6)}_skspec.png')
                utils.save_image(_x, f'sample/{str(it_count).zfill(6)}_output.png',
                  nrow=8,
                  normalize=True,
                  range=(-1, 1))

              if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
              
              if it_count % 10000 == 0:
                torch.save(
                  {
                    "encoder": encoder.state_dict(),
                    "decoder": decoder.state_dict()
                  }, 
                  args.save_dir+'/checkpoint_'+str(it_count)+'.pt')    

              it_count += 1

