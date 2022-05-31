import torch
import torchaudio
import librosa
import librosa.display
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from glob import glob
from torch.utils.data import Dataset
from tqdm import tqdm

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

#Waveform array from path of folder containing wav files
def audio_array_from_path(path):
  ls = glob(f'{path}/**/*.wav', recursive=True)
  ls = ls + glob(f'{path}/**/*.ogg', recursive=True)
  ls = ls + glob(f'{path}/**/*.mp3', recursive=True)
  ls = ls + glob(f'{path}/**/*.aac', recursive=True)
  ls = ls + glob(f'{path}/**/*.flac', recursive=True)
  adata = []
  for i in range(len(ls)):
    x, sr = torchaudio.load(ls[i])
    x = x.numpy()
    adata.append(x[0])
  return np.array(adata)

def audio_array_from_track(path):
  adata = []
  x, sr = torchaudio.load(path)
  x = x.numpy()
  adata.append(x[0])
  return np.array(adata)

def audio_array_from_tensor(audio_tensor):
    adata = []
    x = audio_tensor.numpy()
    adata.append(x[0])
    return np.array(adata)


def audio_array_from_batch(batch):
    adata = []
    for i in range(len(batch)):
      x = batch[i].numpy()
      adata.append(x[0])
    return np.array(adata)

class AudioData(Dataset):
    def __init__(self, root='train'):
        self.root = root
        ls = glob(f'{root}/**/*.wav', recursive=True)
        ls = ls + glob(f'{root}/**/*.ogg', recursive=True)
        ls = ls + glob(f'{root}/**/*.mp3', recursive=True)
        ls = ls + glob(f'{root}/**/*.aac', recursive=True)
        ls = ls + glob(f'{root}/**/*.flac', recursive=True)
        self.files = ls

    def __len__(self):
        # Here, we need to return the number of samples in this dataset.
        return len(self.files)

    def __getitem__(self, index):
        print("loading file: " + str(self.files[index]))
        x, sr = torchaudio.load(self.files[index])
        return x

def collate_list(batch):
    return batch

#Waveform to Spectrogram conversion

''' Decorsière, Rémi, Peter L. Søndergaard, Ewen N. MacDonald, and Torsten Dau.
"Inversion of auditory spectrograms, traditional spectrograms, and other envelope representations."
IEEE/ACM Transactions on Audio, Speech, and Language Processing 23, no. 1 (2014): 46-56.'''

#ORIGINAL CODE FROM https://github.com/yoyololicon/spectrogram-inversion

def spectral_convergence(input, target):
    return 20 * ((input - target).norm().log10() - target.norm().log10())

def GRAD(spec, args, transform_fn, samples=None, init_x0=None, maxiter=1000, tol=1e-6, verbose=1, evaiter=10, lr=0.002):

    spec = torch.Tensor(spec)
    samples = (spec.shape[-1]*args.hop)-args.hop

    if init_x0 is None:
        init_x0 = spec.new_empty((1,samples)).normal_(std=1e-6)
    x = nn.Parameter(init_x0)
    T = spec

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam([x], lr=lr)

    bar_dict = {}
    metric_func = spectral_convergence
    bar_dict['spectral_convergence'] = 0
    metric = 'spectral_convergence'

    init_loss = None
    with tqdm(total=maxiter, disable=not verbose) as pbar:
        for i in range(maxiter):
            optimizer.zero_grad()
            V = transform_fn(x)
            loss = criterion(V, T)
            loss.backward()
            optimizer.step()
            lr = lr*0.9999
            for param_group in optimizer.param_groups:
              param_group['lr'] = lr

            if i % evaiter == evaiter - 1:
                with torch.no_grad():
                    V = transform_fn(x)
                    bar_dict[metric] = metric_func(V, spec).item()
                    l2_loss = criterion(V, spec).item()
                    pbar.set_postfix(**bar_dict, loss=l2_loss)
                    pbar.update(evaiter)

    return x.detach().view(-1).cpu()

def normalize(S, args):
  return np.clip((((S - args.min_db) / -args.min_db)*2.)-1., -1, 1)

def denormalize(S, args):
  return (((np.clip(S, -1, 1)+1.)/2.) * -args.min_db) + args.min_db

def prep(wv, args, specfunc):
  S = np.array(torch.squeeze(specfunc(torch.Tensor(wv).view(1,-1))).detach().cpu())
  S = librosa.power_to_db(S)-args.ref_db
  return normalize(S, args)

def deprep(S, args, specfunc):
  S = denormalize(S,args)+args.ref_db
  S = librosa.db_to_power(S)
  # wv = GRAD(np.expand_dims(S,0), args, specfunc, maxiter=2500, evaiter=10, tol=1e-8)
  wv = librosa.griffinlim(S)
  return np.array(np.squeeze(wv))

#---------Helper functions------------#

#Generate spectrograms from waveform array
def tospec(data, args, specfunc):
  specs=np.empty(data.shape[0], dtype=object)
  for i in range(data.shape[0]):
    x = data[i]
    S=prep(x, args, specfunc)
    S = np.array(S, dtype=np.float32)
    specs[i]=np.expand_dims(S, -1)
  return specs

#Generate multiple spectrograms with a determined length from single wav file
def tospeclong(path, length=4*44100):
  x, sr = librosa.load(path,sr=44100)
  x,_ = librosa.effects.trim(x)
  loudls = librosa.effects.split(x, top_db=50)
  xls = np.array([])
  for interv in loudls:
    xls = np.concatenate((xls,x[interv[0]:interv[1]]))
  x = xls
  num = x.shape[0]//length
  specs=np.empty(num, dtype=object)
  for i in range(num-1):
    a = x[i*length:(i+1)*length]
    S = prep(a)
    S = np.array(S, dtype=np.float32)
    try:
      sh = S.shape
      specs[i]=S
    except AttributeError:
      print('spectrogram failed')
  return specs

#Concatenate spectrograms in array along the time axis
def testass(a):
  but=False
  con = np.array([])
  nim = a.shape[0]
  for i in range(nim):
    im = a[i]
    im = np.squeeze(im)
    if not but:
      con=im
      but=True
    else:
      con = np.concatenate((con,im), axis=1)
  return np.squeeze(con)

#Split spectrograms in chunks with equal size
def splitcut(data,args):
  ls = []
  mini = 0
  minifinal = args.spec_split*args.shape   #max spectrogram length
  print(data.shape[0])
  for i in range(data.shape[0]-1):
    if data[i].shape[1]<=data[i+1].shape[1]:
      mini = data[i].shape[1]
    else:
      mini = data[i+1].shape[1]
    if mini>=3*args.shape and mini<minifinal:
      minifinal = mini
  for i in range(data.shape[0]):
    x = data[i]
    if x.shape[1]>=3*args.shape:
      for n in range(x.shape[1]//minifinal):
        ls.append(x[:,n*minifinal:n*minifinal+minifinal,:])
      ls.append(x[:,-minifinal:,:])
  return np.array(ls)


#-----TESTING FUNCTIONS ----------- #

def select_spec(spec, labels, num_spec=10):
    sample_spec_index = np.random.choice(range(len(spec)), num_spec)
    sample_spec = spec[sample_spec_index]
    sample_labels = labels[sample_spec_index]
    return sample_spec, sample_labels


def plot_reconstructed_spec(spec, reconstructed_spec):
    fig = plt.figure(figsize=(15, 3))
    num_spec = len(spec)
    for i, (image, reconstructed_image) in enumerate(zip(spec, reconstructed_spec)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_spec, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_spec, i + num_spec + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


def plot_spec_encoded_in_latent_space(latent_representations, sample_labels):
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()

#---------------NOISE GENERATOR FUNCTIONS ------------#

def generate_random_z_vect(seed=1001,size_z=1,scale=1,vector_dim=128):
    np.random.seed(seed)
    x = np.random.uniform(low=(scale * -1.0), high=scale, size=(size_z,vector_dim))
    return x


#-------SPECTROGRAM AND SOUND SYNTHESIS UTILITY FUNCTIONS -------- #

#Assembling generated Spectrogram chunks into final Spectrogram
def specass(a,spec, shape):
  but=False
  con = np.array([])
  nim = a.shape[0]
  for i in range(nim-1):
    im = a[i]
    im = np.squeeze(im)
    if not but:
      con=im
      but=True
    else:
      con = np.concatenate((con,im), axis=1)
  diff = spec.shape[1]-(nim*shape)
  a = np.squeeze(a)
  con = np.concatenate((con,a[-1,:,-diff:]), axis=1)
  return np.squeeze(con)

#Splitting input spectrogram into different chunks to feed to the generator
def chopspec(spec, shape):
  dsa=[]
  for i in range(spec.shape[1]//shape):
    im = spec[:,i*shape:i*shape+shape]
    im = np.reshape(im, (im.shape[0],im.shape[1],1))
    dsa.append(im)
  imlast = spec[:,-shape:]
  imlast = np.reshape(imlast, (imlast.shape[0],imlast.shape[1],1))
  dsa.append(imlast)
  return np.array(dsa, dtype=np.float32)

#Converting from source Spectrogram to target Spectrogram
def towave_reconstruct(spec, spec1, args, specfunc, name, path='../content/', show=False, save=False):
  specarr = chopspec(spec, args.shape)
  specarr1 = chopspec(spec1, args.shape)
  print(specarr.shape)
  a = specarr
  print('Generating...')
  ab = specarr1
  print('Assembling and Converting...')
  a = specass(a,spec,args.shape)
  ab = specass(ab,spec1,args.shape)
  awv = deprep(a,args,specfunc)
  abwv = deprep(ab,args,specfunc)
  if save:
    print('Saving...')
    pathfin = f'{path}/{name}'
    sf.write(f'{pathfin}.wav', awv, args.sr)
    print('Saved WAV!')
  if show:
    fig, axs = plt.subplots(ncols=2)
    axs[0].imshow(np.flip(a, -2), cmap=None)
    axs[0].axis('off')
    axs[0].set_title('Reconstructed')
    axs[1].imshow(np.flip(ab, -2), cmap=None)
    axs[1].axis('off')
    axs[1].set_title('Input')
    plt.show()
  return abwv

#Converting from Z vector generated spectrogram to waveform
def towave_from_z(spec, args, specfunc, name, path='../content/', show=False, save=False):
  specarr = chopspec(spec, args.shape)
  print(specarr.shape)
  a = specarr
  print(a.shape)
  print('Generating...')
  print('Assembling and Converting...')
  a = specass(a,spec,args.shape)
  print(a.shape)
  awv = deprep(a,args,specfunc)
  if save:
    print('Saving...')
    pathfin = f'{path}/{name}'
    sf.write(f'{pathfin}.wav', awv, args.sr)
    print('Saved WAV!')
  if show:
    fig, axs = plt.subplots(ncols=1)
    axs.imshow(np.flip(a, -2), cmap=None)
    axs.axis('off')
    axs.set_title('Decoder Synthesis')
    plt.show()
  return awv
