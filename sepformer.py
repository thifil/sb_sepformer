#!/usr/bin/env python3
# coding: utf-8




try:
    loc = get_ipython().__class__.__name__
    if(loc == 'ZMQInteractiveShell'):
        environment = "jupyter"
except NameError:
    # Running as commandline tool
    environment = "shell"

try:
    import speechbrain
except ModuleNotFoundError:
    if environment == "jupyter":
        get_ipython().system('pip install speechbrain')
        get_ipython().system('pip install wandb')
        import IPython
        IPython.Application.instance().kernel.do_shutdown(True) #automatically restarts kernel
    import speechbrain


# In[100]:


try:
    import pytorch_lightning as pl
except ModuleNotFoundError:
    if environment == "jupyter":
        get_ipython().system('pip install pytorch-lightning')
    import pytorch_lightning as pl


# In[101]:


try:
    import torchmetrics
except ModuleNotFoundError:
    if environment == "jupyter":
        get_ipython().system('pip install torchmetrics')
    import torchmetrics


# In[102]:


print("Executing on environment: "+environment)
    


# In[104]:


#!pip install wandb


# # Loading & reading the config file

# In[105]:


yamlfile = "config_sepformer.yaml"
verbose_level = 1
if environment == "jupyter":
    verbose_level = 1
# set to 0 if no infos shall be printed
# set to 1 to only print shape sizes and text statements
# set to 2 to "print" audio players
DATALOADER_WORKERS = 12 # depends on the number of CPUs the used machine can muster
if environment == "jupyter":
    wandb_project = "sepformer_dev"
else:
    wandb_project = "sepformer_training"


# In[106]:


#!pip install wandb


# In[107]:


import speechbrain as sb
import hyperpyyaml as yaml
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import sys
from pytorch_lightning.loggers import WandbLogger
#wandb_logger = WandbLogger(project=wandb_project)

import copy
from torch.cuda.amp import autocast


# # Data Preparation

# ## Preparing the librimix csv files
# ... to fit speechbrain specifications (ID column must be named "ID")

# Copied from https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/dataio/dataio.html#read_audio
# Changed to return the sampling rate fs
global_fs = 8000
def read_audio(waveforms_obj):
    """General audio loading, based on a custom notation.

    Expected use case is in conjunction with Datasets
    specified by JSON.

    The custom notation:

    The annotation can be just a path to a file:
    "/path/to/wav1.wav"

    Or can specify more options in a dict:
    {"file": "/path/to/wav2.wav",
    "start": 8000,
    "stop": 16000
    }

    Arguments
    ----------
    waveforms_obj : str, dict
        Audio reading annotation, see above for format.

    Returns
    -------
    torch.Tensor
        Audio tensor with shape: (samples, ).

    Example
    -------
    >>> dummywav = torch.rand(16000)
    >>> import os
    >>> tmpfile = str(getfixture('tmpdir') / "wave.wav")
    >>> write_audio(tmpfile, dummywav, 16000)
    >>> asr_example = { "wav": tmpfile, "spk_id": "foo", "words": "foo bar"}
    >>> loaded = read_audio(asr_example["wav"])
    >>> loaded.allclose(dummywav.squeeze(0),atol=1e-4) # replace with eq with sox_io backend
    True
    """
    if isinstance(waveforms_obj, str):
        audio, fs = torchaudio.load(waveforms_obj)
        verbose("read_audio:fs", fs)
        return audio.transpose(0, 1).squeeze(1)

    path = waveforms_obj["file"]
    start = waveforms_obj.get("start", 0)
    # Default stop to start -> if not specified, num_frames becomes 0,
    # which is the torchaudio default
    stop = waveforms_obj.get("stop", start)
    num_frames = stop - start
    audio, fs = torchaudio.load(path, num_frames=num_frames, frame_offset=start)
    verbose("read_audio:fs", fs)
    global_fs=fs
    #global_fs.append(fs)
    #verbose("global_fs:", str(global_fs))
    audio = audio.transpose(0, 1)
    return audio.squeeze(1)

def write_signal(signal, path):
    sig_cpu = signal.detach().clone().to("cpu").type(torch.float32)
    torchaudio.save(path, sig_cpu, global_fs)
    verbose("write_audio:written_to", path)

def adjust_path(path):
    if(environment == "jupyter"):
        # In this case the path needs alternation
        #print(path)
        path = path.replace(r"/home/valentinf", r"/notebooks/attention_speech")
        #print(path)
    return path



import pandas as pd
class SpeechDataset(Dataset):
    def __init__(self, filepath):
        #verbose("SpeechDataset:__init__:filepath", filepath)
        
        self.filelist = pd.read_csv(filepath)
        verbose("SpeechDataset:__init__:filelist.shape", self.filelist.shape)
        
    
    def _get_audio(self, path):
        #verbose("SpeechDataset:_get_audio:start", ".")
        path = adjust_path(path)
        signal = read_audio(path)
        verbose("SpeechDataset:_get_audio:signal.shape",signal.shape)
        if signal.shape[-1]%2!=0:
            shp = signal.shape[-1]
            signal = signal[:-1]
            verbose("SpeechDataset:_get_audio:signal","signal shape was adjusted from %s to %s"%(shp, signal.shape[-1]), forcePrint=True)
        return signal
    
    def __getitem__(self, index):
        row = self.filelist.iloc[index]
        verbose("SpeechDataset:__getitem__:row_id", str(row[0]))
        #verbose("SpeechDataset:__getitem__:mix_path",row["mixture_path"])
        mix = self._get_audio(row["mixture_path"])
        #verbose("SpeechDataset:__getitem__:mix.shape", str(mix.shape))
        s1 = self._get_audio(row["source_1_path"])
        #verbose("SpeechDataset:__getitem__:s1.shape", str(s1.shape))
        s2 = self._get_audio(row["source_2_path"])
        #verbose("SpeechDataset:__getitem__:s2.shape", str(s2.shape))
        
        return {"mix":mix, "s1":s1, "s2":s2}
                         
        
    def __len__(self):
        return self.filelist.shape[0]
        
class SpeechDataLightningModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, batch_size, parameters):#data_dir: str ="./"):
        super().__init__()
        verbose("SpeechDataLightningModule:__init__:start", ".")
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.parameters = parameters
        
    def setup(self, stage:str):
        # Assign train/val datasets for use in dataloaders
        verbose("SpeechDataLightningModule:setup:start", ".")
        if stage == "fit":
            verbose("SpeechDataLightningModule:setup:fit", ".")
            self.train_dataset = SpeechDataset(self.train_dir)
            self.val_dataset = SpeechDataset(self.val_dir)
            
        # Assign test dataset for use in dataloaders
        if stage == "test":
            raise NotImplementedError("SpeechDataLightningModule - test is not yet implemented")
        
        if stage == "predict":
            raise NotImplementedError("SpeechDataLightningModule - predict is not yet implemented")
    
    def train_dataloader(self):
        verbose("SpeechDataLightningModule:train_dataloader:start", ".")
        num_workers = DATALOADER_WORKERS
        if environment=="jupyter":
            num_workers=2
        dloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        verbose("SpeechDataLightningModule:train_dataloader:end", ".")
        return dloader
    
    def val_dataloader(self):
        vloader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        return vloader
    
    def test_dataloader(self):
        raise NotImplementedError("SpeechDataLightningModule - test is not yet implemented")

    def predict_dataloader(self):
        raise NotImplementedError("SpeechDataLightningModule - predict is not yet implemented")


# # Model

# ## Implementing Model Classes

# ### Implementing the Encoder:

# In[113]:


def verbose(function, text, forcePrint=False):
    if verbose_level > 0 or forcePrint == True:
        print("%s - %s "%(function, text))#\n

def reset_verbose_level():
    verbose_level = 0


# In[114]:


class SepEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=64, kernel_size=2, stride=-1):
        super().__init__()
        verbose("SepEncoder:__init__:start",".")
        if(stride == -1):
            stride = kernel_size // 2
        self.model = nn.Conv1d(in_channels, out_channels, kernel_size, stride, bias=False, groups=1)
        self.relu = nn.ReLU()
        self.channels_in = in_channels

    def forward(self, data):
        verbose("SepEncoder:Forward:Entered", str(data.shape))
        if(self.channels_in == 1):
            data = torch.unsqueeze(data, dim=1)
        verbose("SepEncoder:Forward:After_Unsqueeze", str(data.shape))
        #if(self.channels_in == 1):
        #    raise ValueError("channel_in is 1 and this requires special handling")
        latent_data = self.relu(self.model(data))
        verbose("SepEncoder:Forward:Return", str(latent_data.shape))
        return latent_data


from typing import Optional

class PositionalFeedForward(nn.Module):
#Implementation according to https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/nnet/attention.html#PositionalFeedForward
# This implements the positional-wise feed forward according to "Attention is All You Need"
    def __init__( self, d_ffn, input_shape=None, input_size=None, dropout=0.0, activationFunction=nn.ReLU):
        super().__init__()
        verbose("PositionalFeedForward:__init__:start",".")
        if input_shape is None and input_size is None:
            raise ValueError("not enough parameters provided")
            
        if input_size is None:
            input_size = input_shape[-1]
        self.feed_forward = nn.Sequential(
            nn.Linear(input_size, d_ffn),
            activationFunction(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, input_size)
        )
        
    def forward(self, x):
        verbose("PositionalFeedForward:forward:start:x",str(x.shape))
        x = x.permute(1,0,2)
        x = self.feed_forward(x)
        x = x.permute(1,0,2)
        verbose("PositionalFeedForward:forward:returned:x",str(x.shape))
        return x

#Implementation according to https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/lobes/models/transformer/Transformer.html#TransformerEncoderLayer
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_ffn, nhead, d_model, kdim=None, vdim=None, dropout=0.0, activation=nn.ReLU, normalize_before=False, attention_type="regularMHA", causal=False):
        super().__init__()
        verbose("TransformerEncoderLayer:__init__:start",".")
        if attention_type == "regularMHA":
            self.self_att = nn.MultiheadAttention(
                embed_dim=d_model,
                num_heads=nhead,
                dropout=dropout, 
                bias=True,
                add_bias_kv=False,
                add_zero_attn=False,
                kdim=kdim,
                vdim=vdim
            )
        else:
            raise ValueError("attention_type %s unknown or not umplemented"%attention_type)
        self.pos_ffn = PositionalFeedForward( #implemented according to sb.nnet.attention.PositionalFeedForward
            d_ffn=d_ffn,
            input_size = d_model,
            dropout = dropout,
            activationFunction = activation
        )
        self.norm1 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.norm2 = sb.nnet.normalization.LayerNorm(d_model, eps=1e-6)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)

        self.normalize_before = normalize_before
    
    def forward(self, src, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, pos_embs: Optional[torch.Tensor] = None): 
        verbose("TransformerEncoderLayer:forward:start:src",str(src.shape))
        if self.normalize_before:
            src1 = self.norm1(src)
        else:
            src1 = src

        output, self_attn = self.self_att(
            src1,
            src1,
            src1,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            pos_embs=pos_embs,
        )

        
        src = src + self.dropout1(output)
        if not self.normalize_before:
            src = self.norm1(src)

        if self.normalize_before:
            src1 = self.norm2(src)
        else:
            src1 = src
        output = self.pos_ffn(src1)

        
        output = src + self.dropout2(output)
        if not self.normalize_before:
            output = self.norm2(output)
        verbose("TransformerEncoderLayer:forward:returned:output",str(output.shape))
        verbose("TransformerEncoderLayer:forward:returned:self_attn",str(self_attn.shape))
        return output, self_attn
            
                
# Implemented according to https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/lobes/models/transformer/Transformer.html#TransformerEncoder
class TransformerEncoder(nn.Module): #inspired from speechbrain -> TransformerEncoder
    def __init__(self, num_layers, nhead, d_ffn,input_shape, d_model, kdim, vdim, dropout, activation, normalize_before, causal, attention_type, layerdrop_propability=0.0, isTraining=True):
        super().__init__()
        verbose("TransformerEncoder:__init__:start",".")
        self.layers = torch.nn.ModuleList([
              TransformerEncoderLayer(
                  nhead=nhead,
                  d_ffn = d_ffn,
                  d_model = d_model,
                  kdim = kdim,
                  vdim = vdim, 
                  dropout = dropout,
                  activation = activation,
                  normalize_before = normalize_before,
                  causal=causal,
                  attention_type=attention_type
              ) for i in range(0,num_layers)
        ])
        self.Training=isTraining
        input_size=d_model
        self.norm = torch.nn.LayerNorm(
            input_size,
            eps = 1e-6,
            elementwise_affine = True
        ) 
        #self.layerdrop_propability = layerdrop_propability
        #self.random_generator = np.random.default_rng()
    
    def forward(self, src, src_mask: Optional[torch.Tensor] = None, src_key_padding_mask: Optional[torch.Tensor] = None, pos_embbeding: Optional[torch.Tensor] = None):
        # Right now I did not implement layer drop probability as I do not know if it will be used
        verbose("TransformerEncoder:forward:start:src",str(src.shape))
        output = src
        attentions = []
        for i, layer in enumerate(self.layers):
            if (self.Training == False): #here again layer drop probability was excluded from implementation
                output, attention = layer(output, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask, pos_embs=pos_embs)
                attentions.append(attention)
        output = self.norm(output) #only the last output is the output of the block
        verbose("TransformerEncoder:forward:returned:output",str(output.shape))
        verbose("TransformerEncoder:forward:returned:attentions",str(output.shape))
        return output, attentions
                



class DualPathBlock(nn.Module):
    def __init__(self, intra_model, inter_model, out_channels, norm="ln", skip_intra=True, linear_after_inter_intra=True):
        super(DualPathBlock, self).__init__()
        verbose("DualPathBlock:__init__:start",".")
        self.model_intra = intra_model
        self.model_inter=inter_model
        self.skip_intra = skip_intra
        self.linear_after_inter_intra=linear_after_inter_intra
        
        self.norm = norm
        if norm == "ln":
            self.intra_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
            self.inter_norm = nn.GroupNorm(1, out_channels, eps=1e-8)
        else:
            raise ValueError("unknown norm")

        # Code for Linear layer after inter intra left out, as not needed in this implementation

    def forward(self, x):
        # Expected input dimensions: [batchsize, number of filters, time points in each chunk, number of chunks]
        # Output dimension = [batchsize, number of filters, time points in each chunk, number of chunks]
        verbose("DualPathBlock:forward:start:x", str(x.shape))
        verbose("DualPathBlock:forward:start:expected-shape", "B,N,K,S")
        # B, N, K, S
        # intra code
        batch_size, n_filters, n_time, n_chunks = x.shape
        intra = x.permute(0,3,2,1).contiguous().view(batch_size * n_chunks, n_time, n_filters)
        verbose("DualPathBlock:forward:before-model_intra:intra", str(intra.shape))
        intra = self.model_intra(intra)
        verbose("DualPathBlock:forward:after-model_intra:intra", str(intra.shape))
        
        # Code for Linear layer after inter intra left out, as not needed in this implementation
        intra = intra.view(batch_size, n_chunks, n_time, n_filters)
        intra = intra.permute(0,3,2,1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)
        
        if self.skip_intra:
            intra = intra + x
        verbose("DualPathBlock:forward:end-intra-computation:intra", str(intra.shape))
        
        #inter code
        inter = intra.permute(0,2,3,1).contiguous().view(batch_size * n_time, n_chunks, n_filters)
        verbose("DualPathBlock:forward:before-model_inter:inter", str(inter.shape))
        inter = self.model_inter(inter)
        verbose("DualPathBlock:forward:after-model_inter:inter", str(inter.shape))
        
        # Code for Linear layer after inter intra left out, as not needed in this implementation
        
        inter = inter.view(batch_size, n_time, n_chunks, n_filters)
        inter = inter.permute(0,3,1,2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        out = inter + intra
        verbose("DualPathBlock:forward:returned:out",str(out.shape))
        return out


# In[118]:


if environment == "jupyter":
    #Testing for positional embeddings
    import numpy as np
    d=10
    for i in torch.arange(int(d/2)):
        print(2*i)
        print(2*i+1)
        #-> erreicht das jedes zweite feld mit sin bzw. cos respectively besetzte wird.

if environment == "jupyter":
    # Original Code for positional Encodings
    import math
    input_size = 6
    mlen = 20
    pe = torch.zeros(mlen, input_size, requires_grad=False)
    positions = torch.arange(0, mlen).unsqueeze(1).float()
    denominator = torch.exp(torch.arange(0, input_size, 2).float()
                * -(math.log(mlen) / input_size))
    print(denominator)
    print(torch.sin(positions * denominator).shape)
    print(torch.sin(positions * denominator))


    print(torch.arange(0, input_size, 2).float()) ## =^ k =^ 2iif environment == "jupyter":
    # machine learning mastery version of positional embeddings
    window_size = input_size
    mlen = mlen # maximum length of the input sequence
    p = torch.zeros(mlen, window_size, requires_grad=False)
    for k in range(mlen):
        for i in torch.arange(int(window_size/2)):
            denominator = torch.pow(mlen, 2*i/window_size)
            p[k, 2*i]= torch.sin(k/denominator)
            p[k, 2*i+1]=torch.cos(k/denominator)
    print(p)
    print(p.shape)
# ![image.png](attachment:c9765924-f99b-4519-b096-d06d639adcfd.png)
# d ... size of output embedding space (=window size)
# k ... index of specific token within window_size
# i ...

# In[119]:


class PositionalEmbedding(nn.Module):
    # -> paper 2.3.1 "First of all, sinusoidal positional encoding e is added to the input z", according to original "Attention is All you need" paper
    # According to https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/
    def __init__(self, window_size, mlen=2500):
        super().__init__()
        verbose("PositionalEmbedding:__init__:start",".")
        self.mlen = mlen # maximum length of the input sequence
        p = torch.zeros(self.mlen, window_size, requires_grad=False)
        for k in range(self.mlen):
            for i in torch.arange(int(window_size/2)):
                denominator = torch.pow(mlen, 2*i/window_size)
                p[k, 2*i]= torch.sin(k/denominator)
                p[k, 2*i+1]=torch.cos(k/denominator)
        p = p.unsqueeze(0)
        verbose("PositionalEmbedding:__init__:end:p",str(p.shape))
        self.register_buffer("p", p)
            
        
        """If you have parameters in your model, which should be saved and restored in the state_dict, but not trained by the optimizer, you should register them as buffers.
Buffers won’t be returned in model.parameters(), so that the optimizer won’t have a change to update them."""
    def forward(self, x):
        #print("Debug here to understand the dimension of x")
        #Type of return: <class 'torch.Tensor'>
        #Shape of return: torch.Size([2500, 250])
        verbose("PositionalEmbedding:forward:start:x.shape",str(x.shape))
        verbose("PositionalEmbedding:forward:start:p.shape", str(self.p.shape))
        verbose("PositionalEmbedding:forward:output.shape", str(self.p[:,:x.size(1)].shape))
        return self.p[:,:x.size(1)].clone().detach()


# In[120]:


# Source https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/lobes/models/dual_path.html#SBTransformerBlock
class TransformerBlock (nn.Module): #This block is used for the inter and intra connections
    def __init__(self, num_layers, d_model, nhead, d_ffn=2048, input_shape=None, kdim=None, vdim=None, use_positional_encoding=False, dropout=0.1, activation="relu", normalize_before=False, attention_type="regularMHA"):
        super().__init__()
        verbose("TransformerBlock:__init__:start",".")
        if(activation == "relu"):
            activation = nn.ReLU
        else:
            raise ValueError("Unknown activation function")
        self.use_positional_encoding = use_positional_encoding
        
        self.trans_enc = TransformerEncoder(num_layers=num_layers, 
                                            nhead=nhead,
                                            d_ffn=d_ffn,
                                            input_shape=input_shape,
                                            d_model = d_model,
                                            kdim=kdim,
                                            vdim=vdim,
                                            dropout=dropout,
                                            activation=activation,
                                            normalize_before=normalize_before,
                                            attention_type=attention_type,
                                            causal = False,
                                            isTraining=True)
        
        if use_positional_encoding:
            self.pos_enc = PositionalEmbedding(window_size=d_model) #part of speechbrain.lobes.models.transformer.Transformer import PositionalEncoding
        
    def forward(self, x):
        verbose("TransformerBlock:forward:start:x",str(x.shape))
        verbose("TransformerBlock:forward:start:expected_input_shape", "[Batchsize, timepoints, number filters]")
        if self.use_positional_encoding:
            pos_enc = self.pos_enc(x)
            verbose("TransformerBlock:forward:pos_enc",str(pos_enc.shape))
            return self.trans_enc(x + pos_enc)[0]
        return self.trans_enc(x)[0]
        
# Documentation: https://speechbrain.readthedocs.io/en/latest/API/speechbrain.lobes.models.dual_path.html#speechbrain.lobes.models.dual_path.Dual_Path_Model
# Code: https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/lobes/models/dual_path.html#Dual_Path_Model
class TransformerModule(nn.Module): #This model pulls the transformer together and is between encoder and decoder (=MaskNet)
    def __init__(self, 
                 in_channels,
                 out_channels,
                 intra_model,
                 inter_model,
                 n_layers=1,
                 norm="ln",
                 K=200,
                 num_spks=2,
                 skip_around_intra=True,
                 linear_layer_after_inter_intra=True,
                 max_length=20000
                ):
        super().__init__()
        self.K=K
        self.n_layers = n_layers
        self.num_spks = num_spks
        if norm == "ln":
            verbose("TransformerModule:__init__:in_channels",in_channels)
            self.norm = nn.GroupNorm(1, in_channels, eps=1e-8)
        else:
            raise ValueError("norm not implemented")
        self.conv1d=nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.dual_module = nn.ModuleList([])
        for i in range(n_layers):
            self.dual_module.append(
                copy.deepcopy(
                    DualPathBlock(intra_model, inter_model, out_channels, norm, skip_intra=skip_around_intra, linear_after_inter_intra=linear_layer_after_inter_intra)
                )
            )
        self.conv2d = nn.Conv2d(
            out_channels, out_channels*num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU() # = programmable relu
        self.activation = nn.ReLU()
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )
        
    def forward(self, x):
        # Copied from source!
        # [B, N, L] [nr batches, number of filters, number of timepoints]
        verbose("TransformerModule:forward:start:x", str(x.shape))
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d(x)
        verbose("TransformerModule:forward:conv1d-1", str(x.shape))
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
                x.size(1) ** 0.5
            )
        verbose("TransformerModule:forward:pos_enc-1", str(x.shape))
        # [B, N, K, S], [nr batches, number of filters, length of each chunk, number of chunks]
        
        # # # 'Chopping up h along the time axis' # # #

        # # # # Adding Padding # # # #
        B_, N_, L_ = x.shape
        P_ = self.K // 2
        #x, gap = self._padding(x, self.K)
        B, N, L = input.shape
        P = K // 2
        gap = self.K - (P_ + L_ % self.K) % self.K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B_, N_, gap)).type(x.type())
            x = torch.cat([x, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B_, N_, P_)).type(x.type())
        x = torch.cat([_pad, x, _pad], dim=2)


        # [B, N, K, S]
        x_1 = x[:, :, :-P_].contiguous().view(B_, N_, -1, self.K)
        x_2 = x[:, :, P_:].contiguous().view(B_, N_, -1, self.K)
        x = (
            torch.cat([x_1, x_2], dim=3).view(B_, N_, -1, self.K).transpose(2, 3)
        )
        x = x.contiguous()

        verbose("TransformerModule:forward:_Segmentation ", str(x.shape))
        verbose("Shape Explanation: (batch, filters, timepoints per chunk, number of time points)", "")

        # [B, N, K, S]
        verbose("TransformerModule:forward:before-dual_module:n_layers ", str(self.n_layers))
        for i in range(self.n_layers):
            x = self.dual_module[i](x)
        # Right now x is Output of Sepformer Block
        x = self.prelu(x) 

        # [B, N*spks, K, S], [nr batches, filters * number of speakers, chunk size, number of chunks]
        x = self.conv2d(x) #Linear Layer
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S) # Changing shape to move nr speakers to nr batches

        # [B*spks, N, L]
        #x = self._over_add(x, gap)
        # 
        # # # performing overlapp and add sheme to turn chunked data into time again. = h'''' # # #
        B_c, N_c, K_c, S_c = x.shape
        P_c = K_c // 2
        # [B, N, S, K]
        x = x.transpose(2, 3).contiguous().view(B_c, N_c, -1, K_c * 2)

        x_1 = input[:, :, :, :K_c].contiguous().view(B_c, N_c, -1)[:, :, P_c:]
        x_2 = input[:, :, :, K_c:].contiguous().view(B_c, N_c, -1)[:, :, :-P_c]
        x = x_1 + x_2
        # [B, N, L]
        if gap > 0:
            x = x[:, :, :-gap]


        x = self.output(x) * self.output_gate(x) # The final 2 feed-forward layers

        # [B*spks, N, L]
        x = self.end_conv1x1(x) # this layer was not mentioned in the paper but in their implementation. 

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)  # ReLU activation at the end

        # [spks, B, N, L]
        x = x.transpose(0, 1)
        verbose("TransformerModule:forward:end:x", str(x.shape))

        return x

# ## Decoder

# In[121]:


class SepDecoder (nn.Module):
    def __init__(self, in_channels, out_channels=1, kernel_size=2, stride=-1, bias=False):
        super().__init__()
        #torch.nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
        if(stride == -1):
            stride = kernel_size // 2
        self.ct1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride)
        
    def forward(self, x):
        verbose("SepDecoder:forward:start:x", str(x.shape))
        verbose("SepDecoder:forward:start:ConvTranspose1d", str(self.ct1d))
        for i in range(0, x.shape[0]):
            print(self.ct1d(x[i]).shape)
        #self.ct1d(x[i]).squeeze() for i in range(0, x.shape[0])
        #self.ct1d(x[i]) for i in range(0, x.shape[0])
        s1 = self.ct1d(x[0])
        s2 = self.ct1d(x[1])
        y = torch.concat((s1, s2), dim=1)
        y = torch.permute(y, (0,2,1))
        #y = torch.stack(
        #    [
        #        self.ct1d(x[i]).squeeze() for i in range(0, x.shape[0])
        #    ]
        #)
        verbose("SepDecoder:forward:end:y", str(y.shape))
        
        return y


# ## Extension of Speechbrain Brain class 
# (=training prcedure)

# In[122]:


#Implementation according to 

class Sepformer(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.SepEncoder = SepEncoder(in_channels=1, 
                                     out_channels=hparams.get("out_channels"), 
                                     kernel_size=hparams.get("encoder_kernel_size"))
        self.hparams = hparams
        
        # Creating the inter & intra Connections
        config_intra = self.hparams.get("intraAttention")
        intra_model = TransformerBlock(
            num_layers = config_intra.get("num_layers"), 
            d_model = config_intra.get("d_model"), 
            nhead = config_intra.get("nhead"), 
            d_ffn=config_intra.get("d_ffn"), 
            use_positional_encoding=config_intra.get("use_positional_encoding"), 
            dropout=config_intra.get("dropout"), 
            normalize_before=config_intra.get("norm_before"), 
        )
        
        inter_model = TransformerBlock(
            num_layers = config_intra.get("num_layers"), 
            d_model = config_intra.get("d_model"), 
            nhead = config_intra.get("nhead"), 
            d_ffn=config_intra.get("d_ffn"), 
            use_positional_encoding=config_intra.get("use_positional_encoding"), 
            dropout=config_intra.get("dropout"), 
            normalize_before=config_intra.get("norm_before"), 
        )
        
        
        hparams_mask = hparams.get("Mask")
        self.MaskNet = TransformerModule(in_channels=hparams_mask.get("in_channels"),
                                         out_channels=hparams_mask.get("out_channels"),
                                         intra_model=intra_model, 
                                         inter_model=inter_model, 
                                         n_layers=hparams_mask.get("num_layers"),
                                         norm=hparams_mask.get("norm"),
                                         K=hparams_mask.get("K"),
                                         num_spks=hparams_mask.get("num_spks"),
                                         skip_around_intra=hparams_mask.get("skip_around_intra"),
                                         linear_layer_after_inter_intra=hparams_mask.get("linear_layer_after_inter_intra"),
                                         max_length=hparams_mask.get("max_length"))
        self.Decoder = SepDecoder(
            in_channels=hparams.get("out_channels"),
            out_channels=1,
            kernel_size=hparams.get("encoder_kernel_size"),
        )
        
        
    
    # def compute_forward(self, mix, targets, stage, noise=None):
    def forward(self, x):
        verbose("Sepformer:forward:start", ".")
        
        ## Computing the model here:
        
        ### Encoder
        dinput=x
        verbose("Sepformer:compute_forward:before_encoder:dinput", str(dinput.shape))
        verbose("Sepformer:compute_forward:before_encoder:dinput-tpye", str(type(dinput)))
        latent_data_encoded = self.SepEncoder(x)
        verbose("Sepformer:compute_forward:after_encoder:latent_data_encoded.shape", str(latent_data_encoded.shape))
        verbose("Sepformer:compute_forward:after_encoder:type(latent_data_encoded)", str(type(latent_data_encoded)))
        ### Masking Network
        verbose("Sepformer:compute_forward:before_masknet:latent_data_encoded", str(latent_data_encoded.shape))
        latent_mask = self.MaskNet(latent_data_encoded)
        verbose("Sepformer:compute_forward:after_masknet:latet_mask.shape", str(latent_mask.shape))
        verbose("Sepformer:compute_forward:after_masknet:type(latet_mask)", str(type(latent_mask)))
        latent_data_encoded = latent_data_encoded.repeat(latent_mask.shape[0],1,1,1)
        verbose("Sepformer:compute_forward:after_masknet:latent_data_expanded", str(latent_data_encoded.shape))
        latent_out = latent_data_encoded * latent_mask
        
        ### Decoder:
        seperated_source = self.Decoder(latent_out)
        verbose("Sepformer:compute_forward:out:seperated_source.shape",str(seperated_source.shape))
        ## Berechnen der Loss Function
        #compute_objectives(seperated_source, dtarget)

        return seperated_source
        
    
    
    # def fit_batch(self, batch):
    #     # This trains one batch
    #     print("We are inside fit_batch")
    #     print("type: %s"%type(batch))
    #     print("len: %s"%len(batch))
    #     print(batch)
        
    #def evaluate_batch(self, batch, stage):
    #    pass
    #def on_stage_end(self, stage, stage_loss, epoch):
    #    pass
    #def add_speed_perturb(self, targets, targ_lens):
    #    pass
    #def cut_signals(self, mixture, targets):
    #    pass
    #def reset_layer_recursively(self, layer):
    #    pass
    #def save_results(self, test_data):
    #    pass
    #def save_audio(self, snt_id, mixture, targets, predictions):
    #    pass
    




class SepLightning(pl.LightningModule):
    def __init__(self, model, hparams):
        super().__init__()
        verbose("SepLightning:__init__:started",".")
        self.save_hyperparameters()
        self.model = model
        self.haparms = hparams
        verbose("SepLightning:__init__:ended",".")
    
    def training_step(self, batch, batch_idx):
        verbose("SepLightning:training_step","started")
        #print("Batch Type:")
        #print(type(batch))
        #print("Batch Len:")
        #print(len(batch))
        #print("Batch Index:")
        #print(batch_idx)
        #print("Batch:")
        #print(batch)
        
        x_data = batch["mix"]
        verbose("SepLightning:training_step:x_data.shape",x_data.shape)
        verbose("SepLightning:training_step:x_data", x_data)
        verbose("batch[s1].shape", str(batch["s1"].shape))
        y_gt = [batch["s1"], batch["s2"]]
        y_gt = torch.stack(y_gt, dim=2) # using a dim of 2 has the effect that the output shape is [batch, time, speakers]
        verbose("SepLightning:training_step:y_gt.shape",y_gt.shape)
        verbose("SepLightning:training_step:y_gt",y_gt)
        

        y_predicted = self.model(x_data)
        verbose("SepLightning:training_step:y_predicted.shape",y_predicted.shape)
        
        if(verbose_level > 1):
            import IPython
            IPython.display.display(IPython.display.Audio(x_data, rate=8000))
            IPython.display.display(IPython.display.Audio(batch["s1"], rate=8000))
            #IPython.display.display(IPython.display.Audio(y_predicted[0,:], rate=8000))
        
        verbose("SepLightning:y_predicted.shape", y_predicted.shape)
        verbose("SepLightning:y_gt.shape", y_gt.shape)

        loss, loss_torchmetrics = septformer_lossfunction(y_predicted, y_gt)
        self.log('loss_paper', loss)
        self.log('loss_torchmetrics', loss_torchmetrics)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #loss_orig =# Todo: kopieren der loss function aus der original implementierung zum Vergleich.
        verbose("SepLightning:training_step:loss",loss)
        
        verbose("SepLightning:training_step:ended",".")
        return loss
    
    def validation_step(self, batch, batch_idx):
        verbose("SepLightning:validation_step","started")
        
        x_data = batch["mix"]
        verbose("SepLightning:validation_step:x_data.shape",x_data.shape)
        y_gt = [batch["s1"], batch["s2"]]
        y_gt = torch.stack(y_gt, dim=2) # using a dim of 2 has the effect that the output shape is [batch, time, speakers]
        verbose("SepLightning:validation_step:y_gt.shape",y_gt.shape)
        
        y_predicted = self.model(x_data)
        verbose("SepLightning:validation_step:y_predicted.shape",y_predicted.shape)
        
        if(verbose_level > 1):
            import IPython
            IPython.display.display(IPython.display.Audio(x_data, rate=8000))
            IPython.display.display(IPython.display.Audio(batch["s1"], rate=8000))
            #IPython.display.display(IPython.display.Audio(y_predicted[0,:], rate=8000))
        
        loss, loss_torchmetrics = septformer_lossfunction(y_predicted, y_gt)
        self.log('val_loss_paper', loss)
        self.log('val_loss_torchmetrics', loss_torchmetrics)
        self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        #loss_orig =# Todo: kopieren der loss function aus der original implementierung zum Vergleich.
        verbose("SepLightning:validation_step:loss",loss)
        
        #verbose("SepLightning:validation_step:y_predicted.shape",y_predicted.shape)
        #verbose("SepLightning:validation_step:y_predicted[0].shape",y_predicted[0].unsqueeze(0).shape)
        
        if(self.current_epoch % 5 == 0):
            #print(y_predicted.shape)
            #print(y_predicted[:,:,0].shape)
            #print(y_predicted[:,:,0].unsqueeze(0).shape)
            write_signal(x_data, "%s/val_s%s_e%s_input.wav"%(hparams["output_dir"], batch_idx, self.current_epoch))
            write_signal(y_predicted[:,:,0], "%s/val_s%s_e%s_prediction_1.wav"%(hparams["output_dir"], batch_idx, self.current_epoch))
            write_signal(y_predicted[:,:,0], "%s/val_s%s_e%s_prediction_2.wav"%(hparams["output_dir"], batch_idx, self.current_epoch))
            write_signal(batch["s1"], "%s/val_s%s_e%s_gt_1.wav"%(hparams["output_dir"], batch_idx, self.current_epoch))
            write_signal(batch["s2"], "%s/val_s%s_e%s_gt_2.wav"%(hparams["output_dir"], batch_idx, self.current_epoch))
        #if(self.current_epoch == 3):
        #    # reducing the amount data printed to console
        #    verbose_level = 0
        verbose("SepLightning:validation_step:ended",".")
        if(environment == "jupyter" and self.current_epoch > 2):
            raise NotImplementedError("exiting development ide here")
        return loss
    
    
    def configure_optimizers(self):
        verbose("SepLightning:configure_optimizers","started")
        optimizer = torch.optim.Adam(self.parameters(), lr=15e-5)
        # Preparing Learning rate Sheduler:
        lr_scheduler = {
            # REQUIRED: The scheduler instance
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=0.5,patience=3), # patience von 3 gewählt da ich nicht eine "warte periode" einstellen kann
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": "lr_sched",
        }
        verbose("SepLightning:optimizer",optimizer)
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        #return optimizer, lr_scheduler


# # Main Function and Code start

# In[125]:


# This method is a helper function to adapt to be able to run code in the development environment.
def dev_adjust_wav_path(wav):
    if(environment == "jupyter"):
        return wav.replace("/home/valentinf/", "/notebooks/attention_speech/")
    # /home/valentinf/LibriMixData/Libri2Mix/wav8k/max/dev/mix_both/3536-8226-0026_1673-143397-0009.wav
    return wavValueError


# In[126]:


## Main Routine:
from collections import OrderedDict
import torch
import torchaudio

def main(hparams, run_opts):
    #dataset = SpeechDataset(hparams["train_data"])
    #first_data = dataset[0]
    verbose("func:main:instatntiate_SpeechDataLightningModule",".")
    speechDataLoader = SpeechDataLightningModule(hparams["train_data"], hparams["val_data"], hparams["test_data"], hparams["batch_size"], hparams)
    
    ###Following code is to test the data iterator:
    #speechDataLoader.setup("fit")
    #dataiter = iter(speechDataLoader.train_dataloader())
    #verbose("main:post_dataiter:dataiter",dataiter)
    #data = next(dataiter)
    #print(data)
    
    #speechDataLoader.train_dataloader
    #speechbrain.create_experiment_directory(
    #    experiment_directory= hparams["output_dir"],
    #    hyperparams_to_save=yamlfile
    #)
    #train_data, val_data, test_data = data_preprocessing(hparams)
    #print(train_data)
    
    # Testing 
    
    #print(hparams["optimizer"])
    #print(type(hparams["optimizer"]))
    #print(hparams)
    #print(type(hparams))
    #print(run_opts)
    #print(type(run_opts))
    #print(hparams["checkpointer"])
    #print(type(hparams["checkpointer"]))
    
    #modules_ordered = (hparams["modules"])
    
    
    #print(modules_ordered)
    
    
    
    
    # Create Model Class
    verbose("func:main:instatntiate_Sepformer",".")
    seper = Sepformer(
        hparams = hparams, 
    )
    #moduledict = nn.ModuleList(seper)
    #print(moduledict)

    verbose("func:main:instatntiate_SepLightning",".")
    sepLightning = SepLightning(seper, hparams)
    if(hparams["load_from_checkpoint"] == True):
        sepLightning = sepLightning.load_from_checkpoint(hparams["load_checkpoint_path"])
    
    # Reset weights ??

    wandb_logger = WandbLogger(project=wandb_project)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=hparams["checkpoint_dir"], filename='epoch={epoch}-validation_loss={val_loss:.2f}', auto_insert_metric_name=False)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
      
    verbose("func:main:instantiate_trainer",".")
    if environment=="jupyter":
        # Gradient clipping of 5 using L2 norm, according to paper
        trainer = pl.Trainer(max_epochs = 1, callbacks=[checkpoint_callback, lr_monitor], default_root_dir=hparams["checkpoint_dir"]+"/root_dir", gradient_clip_val=5, logger=wandb_logger)
    else:
        # Gradient clipping of 5 using L2 norm, according to paper
        trainer = pl.Trainer(max_epochs = 200, callbacks=[checkpoint_callback, lr_monitor], accelerator='gpu', devices=1, default_root_dir=hparams["checkpoint_dir"], gradient_clip_val=5, logger=wandb_logger, precision=16)
        #precision is for mixed precision, more details here: https://pytorch-lightning.readthedocs.io/en/latest/guides/speed.html
    verbose("func:main:fit",".")
    trainer.fit(model = sepLightning, train_dataloaders=speechDataLoader)
    checkpoint_callback.best_model_path
    verbose("func:main:end",".")
    # Evaluate & Save results
    


# ## Loss Function
from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio

def septformer_lossfunction(prediction, gt): #(outputs, batch, Stage.TRAIN):#
    verbose("func:compute_objective:start:prediction.shape", str(prediction.shape))
    verbose("func:compute_objective:start:gt.shape", str(gt.shape))
    
    #prediction = prediction.unsqueeze(0)
    #gt = gt.unsqueeze(0)

    verbose("func:compute_objective:start:prediction.shape", str(prediction.shape))
    verbose("func:compute_objective:start:gt.shape", str(gt.shape))

    snr_ratio = scale_invariant_signal_noise_ratio(prediction, gt)
    verbose("func:compute_objective:snr_ratio", str(snr_ratio))
    loss = torch.mean(snr_ratio)
    verbose("func:compute_objective:start:loss_orig", str(loss), forcePrint=True)
      
    loss = loss * -1
    
    #verbose("func:compute_objective:start:gt.permuted(0,2,1).shape",torch.permute(gt, (0,2,1)).shape)
    #verbose("func:compute_objective:start:predictions.permuted(0,2,1).shape",torch.permute(prediction, (0,2,1)).shape)
    #loss_paper = sb.nnet.losses.get_si_snr_with_pitwrapper(torch.permute(gt, (0,2,1)),torch.permute(prediction, (0,2,1)))
    #verbose("func:compute_objective:loss_paper-before_threshold", str(loss_paper), forcePrint=True)
    loss_paper = sb.nnet.losses.get_si_snr_with_pitwrapper(gt,prediction)
    with autocast():
        threshold = -30
        loss_to_keep = loss_paper[loss_paper > threshold]
        if loss_to_keep.nelement()>0:
            loss_paper = loss_to_keep.mean()
        else:
            raise ValueError("This state should not occur")
            
    """ fix for computational problems (currently not implemented!)
    if (
                loss < self.hparams.loss_upper_lim and loss.nelement() > 0
            ):  # the fix for computational problems
                self.scaler.scale(loss).backward()
                if self.hparams.clip_grad_norm >= 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.modules.parameters(), self.hparams.clip_grad_norm,
                    )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.nonfinite_count += 1
                logger.info(
                    "infinite loss or empty loss! it happened {} times so far - skipping this batch".format(
                        self.nonfinite_count
                    )
                )
                loss.data = torch.tensor(0).to(self.device)
        """
    
    loss_torchmetrics = loss
    verbose("func:compute_objective:loss_paper", str(loss_paper), forcePrint=True)
    verbose("func:compute_objective:loss_torchmetrics", str(loss_torchmetrics), forcePrint=True)
    loss = loss_paper
    
    
    verbose("func:compute_objective:loss", str(loss), forcePrint=True)
    return loss, loss_torchmetrics #loss_paper
    


# ## Code Execution

# In[ ]:


## START THE MAIN HERE
if(environment == "shell"):
    if __name__ == "__main__":
        #yamlfile, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
        overrides = ""
        run_opts = None
        with open(yamlfile) as fin:
            hparams = yaml.load_hyperpyyaml(fin, overrides)
        main(hparams, run_opts)
elif(environment == "jupyter"):
    overrides = ""
    run_opts= None
    with open(yamlfile) as fin:
        hparams = yaml.load_hyperpyyaml(fin, overrides)
    # In case I am executing on 'paperspace, I only have the validation data availible
    hparams["train_data"] = hparams["val_data"]
    hparams["test_data"] = hparams["val_data"]
    hparams["homepath"] = "/notebooks/attention_speech"
    main(hparams, run_opts)
else:
    raise ValueError ("environment unknown")


# ## Data inspection (jupyter only)

# In[ ]:


# Insights into the data
if environment == "jupyter":
    import IPython

    def sample_playback(key, track_type="mixture_path", samplerate=8000):
        wav = train_data.data[key][track_type]
        audio = (sb.dataio.dataio.read_audio(dev_adjust_wav_path(wav)))
        IPython.display.display(IPython.display.Audio(audio, rate=samplerate))
        return audio

    i = 1
    for key in train_data.data:
        print("%s ###############################"%i)
        print(key)
        print("Mixture:")
        sample_playback(key,track_type="mixture_path")
        print("S1:")
        sample_playback(key,track_type="source_1_path")
        print("S2:")
        sample_playback(key,track_type="source_2_path")
        if(i > 5):
            break
        i=i+1





if(environment == "shell"):
    if __name__ == "__main__":
        #yamlfile, run_opts, overrides = sb.parse_arguments(sys.argv[1:])
        overrides = ""
        run_opts = None
        with open(yamlfile) as fin:
            hparams = yaml.load_hyperpyyaml(fin, overrides)
        #main(hparams, run_opts)
elif(environment == "jupyter"):
    overrides = ""
    run_opts= None
    with open(yamlfile) as fin:
        hparams = yaml.load_hyperpyyaml(fin, overrides)
    # In case I am executing on 'paperspace, I only have the validation data availible
    hparams["train_data"] = hparams["val_data"]
    hparams["test_data"] = hparams["val_data"]
    hparams["homepath"] = "/notebooks/attention_speech"
    #main(hparams, run_opts)
else:
    raise ValueError ("environment unknown")