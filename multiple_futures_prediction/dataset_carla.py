#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2019-2020 Apple Inc. All Rights Reserved.
#

from typing import List, Set, Dict, Tuple, Optional, Union, Any
from torch.utils.data import Dataset, DataLoader
import scipy.io as scp
import numpy as np
import torch
import pickle
import json
import os
import natsort
import glob
import cv2

from multiple_futures_prediction.my_utils import rotate_np


# Dataset for pytorch training
class CarlaDataset(Dataset):
  def __init__(self,
          data_dir:str, # where the 'feed_Episode_*_frame_*.json' files are located
          t_h:int=15, t_f:int=30,
          d_s:int = 1,
          enc_size:int=64, use_gru:bool=False, self_norm:bool=False, 
          data_aug:bool=False,
          use_context:bool=False,
          use_yaws:bool=True, # NOTE: include yaws data in the minibatch
          yaws_to_rad:bool=True, # NOTE: convert yaws to radians in preprocessing?
          nbr_search_depth:int=3,
          ds_seed:int=1234,
          shift_hist:bool=False, shift_fut:bool=True, # NOTE: shift hist/future xy pos to 0-ref?
          rotate_hist:bool=False, rotate_fut:bool=False, # NOTE: rotate hist/future xy pos?
          shuffle:bool=True)-> None:

    assert os.path.exists(data_dir), data_dir
    feeds = glob.glob(os.path.join(data_dir, 'feed_Episode_*_frame_*.json'))
    self.feeds = natsort.natsorted(feeds)

    self.t_h = t_h  # length of track history
    self.t_f = t_f  # length of predicted trajectory
    self.d_s = d_s  # down sampling rate of all sequences
    self.enc_size = enc_size # size of encoder LSTM
    self.grid_size = (13,3) # size of context grid
    self.enc_fac = 2 if use_gru else 1
    self.self_norm = self_norm
    self.data_aug = data_aug
    self.noise = np.array([[5e-2, 5e-2]])
    self.dt = 0.1*self.d_s
    self.ft_to_m = 0.3048
    self.use_context = use_context
    self.use_yaws = use_yaws
    self.yaws_to_rad = yaws_to_rad 
    self.nbr_search_depth = nbr_search_depth

    self.shift_hist = shift_hist
    self.shift_fut = shift_fut
    self.rotate_hist = rotate_hist
    self.rotate_fut = rotate_fut

    #build index of [dataset (0 based), veh_id_0b, frame(time)] into a dictionary
    self.ind_random = np.arange(len(self.feeds))
    self.seed = ds_seed
    np.random.seed(self.seed)
    if shuffle:
      np.random.shuffle(self.ind_random)

  def __len__(self) -> int:
    #return len(self.D)
    return len(self.feeds)

  def process_xy_data(self, ego_hist, ego_fut, ego_yaw, actor_hist, actor_fut, actor_yaw):
    if self.shift_hist or self.shift_fut:
      actor_ref_pos = actor_hist[-1,:] # final (i=0) x,y (i=2)
      ego_ref_pos   = ego_hist[-1,:] # final (i=0) x,y (i=2)
      actor_ref_pos = np.expand_dims(actor_ref_pos, 0)
      ego_ref_pos = np.expand_dims(ego_ref_pos, 0)

    if self.shift_hist:
      raise NotImplementedError("history shouldn't be shifted until forward_mfp!")
      actor_hist -= actor_ref_pos
      ego_hist   -= ego_ref_pos

    if self.shift_fut:
      actor_fut -= actor_ref_pos
      ego_fut   -= ego_ref_pos
    else:
      raise NotImplementedError("fut needs to be shifted to start at 0!")

    if self.rotate_hist:
      # use the yaw to rotate the history around the ref_pos
      raise NotImplementedError("history shouldn't be rotated until foward_mfp!")

    if self.rotate_fut:
      # use the yaw to rotate the future around the ref_pos
      ego_fut = rotate_np([0,0], ego_fut, ego_yaw, degrees=True)
      actor_fut = rotate_np([0,0], actor_fut, actor_yaw, degrees=True)

    hist = [ego_hist, actor_hist] 
    fut  = [ego_fut,  actor_fut] 
    return hist, fut

  def __getitem__(self, idx_: int, shift_hist=False, shift_fut=True) -> Tuple[ List, List, Dict, Union[None, np.ndarray] ] :        
    """"given an index in the dataset, return:
    hist, fut, neighbors, im_crop
    """
    # For the left turn and overtaking dataset,
    # we can assume that there are only two cars (of interest) in the scene 
    idx = self.ind_random[idx_]  # select a timestep/datapoint from the shuffled set

    # Load the json file lazily
    # https://discuss.pytorch.org/t/how-to-incorporate-a-dataset-in-json-format-to-dataloader/67010/2
    with open(self.feeds[idx], 'r') as f:
      feed_dict = json.load(f)
    # dict_keys(['S_past_world_frame', 'yaws', 'agent_presence',
    #  'overhead_features', 'light_strings', 'S_future_world_frame',
    #  'A_future_world_frame', 'A_past_world_frame', 'A_yaws'])
    actor_hist = np.array(feed_dict['A_past_world_frame'])
    actor_fut = np.array(feed_dict['A_future_world_frame'])
    ego_hist = np.array(feed_dict['S_past_world_frame'])
    ego_fut = np.array(feed_dict['S_future_world_frame'])

    actor_yaw = feed_dict['A_yaws']
    ego_yaw = feed_dict['yaws']

    # NOTE: we're only shifting hist here, not fut
    hist, fut = self.process_xy_data(ego_hist, ego_fut, ego_yaw,
                                     actor_hist, actor_fut, actor_yaw)
    ego_lidar = np.array(feed_dict['overhead_features']) # (60,360)
    ego_lidar = ego_lidar[np.newaxis,:,:] # add a channel dim, (1,60,360)
    im_crop = ego_lidar

    # neighbors[ind] maps from
    #     0-index'd vehicle_id
    # to
    #     list of (0-index'd neighbor's vehicle_id, neighbor's vehicle_id, neighbor's grid pos)
    neighbors = {
      0: [(1, 1, 1)],
      1: [(0, 0, 0)],
    }

    if self.yaws_to_rad:
      yaws = np.array([np.deg2rad(ego_yaw), np.deg2rad(actor_yaw)])
    else:
      yaws = np.array([ego_yaw, actor_yaw])
    return hist, fut, neighbors, im_crop, yaws

  def collate_fn(self, samples: List[Any]) -> Tuple[Any,Any,Any,Any,Any,Union[Any,None],Union[Any,None],Any] :
    """Prepare a batch suitable for MFP training.

    Input:
      list of tuples (each the output of getitem[i])
    Output: 
      (hist_batch, nbrs_batch, nbr_inds_batch, fut_batch, mask_batch, context_batch, nbrs_infos)
    """
    nbr_batch_size = 0
    num_samples = 0
    for _,_,nbrs,im_crop,_ in samples:
      nbr_batch_size +=  sum([len(nbr) for nbr in nbrs.values()])      
      num_samples += len(nbrs)

    maxlen = self.t_h  # We want to fix this to 15, if we do +1 we get 16
    if nbr_batch_size <= 0:      
      # none of the getitems[idx] had a non-empty neighbor list
      nbrs_batch = torch.zeros(maxlen,1,2)
    else:
      nbrs_batch = torch.zeros(maxlen,nbr_batch_size,2)
    
    pos = [0, 0]  # pos in the 13x3 grid
    nbr_inds_batch = torch.zeros(
      num_samples,  # eg 2
      self.grid_size[1], self.grid_size[0],  # 3, 13
      self.enc_size*self.enc_fac  # 64*1 = 64
    )
    nbr_inds_batch = nbr_inds_batch.byte()

    hist_batch = torch.zeros(maxlen, num_samples, 2)  # 15, 2, 2
    fut_batch  = torch.zeros(self.t_f//self.d_s, num_samples, 2)  # 30, 2, 2
    mask_batch = torch.zeros(self.t_f//self.d_s, num_samples, 2)  # 30, 2, 2
    if self.use_context:
      context_batch = torch.zeros(num_samples, im_crop.shape[0], im_crop.shape[1], im_crop.shape[2] )
    else:
      context_batch = None # removed typing for py35 compat

    if self.use_yaws:
      yaws_batch = torch.zeros(num_samples) # [ego_yaw, actor_yaw]
    else:
      yaws_batch = None # removed typing for py35 compat

    nbrs_infos = []
    count = 0
    samples_so_far = 0
    for sampleId, (hist, fut, nbrs, context, yaws) in enumerate(samples):            
      # nbrs[ind] maps from
      #     0-index'd vehicle_id
      # to
      #     list of [(0-index'd neighbor's vehicle_id, neighbor's vehicle_id, neighbor's grid pos), ...]
      num = len(nbrs)

      # For each of the vehicle_ids (including self)
      # Populate hist_batch with the history of all the vehicles recorded in the neighbor search
      for j in range(num):
        # clipping the 1st index in case some fut/hist are shorter than others
        # mask=0 indicates that future was shorter than the max
        hist_batch[0:len(hist[j]), samples_so_far+j, :] = torch.from_numpy(hist[j])
        fut_batch[0:len(fut[j]), samples_so_far+j, :] = torch.from_numpy(fut[j]) # 25x2, 30x2
        mask_batch[0:len(fut[j]),samples_so_far+j,:] = 1                
        if self.use_yaws:
          assert yaws.shape == (2,), yaws.shape
          yaws_batch[samples_so_far+j] = torch.Tensor([yaws[j]])                

      samples_so_far += num

      # nbrs[ind] maps from
      #     0-index'd vehicle_id
      # to
      #     list of [(0-index'd neighbor's vehicle_id, neighbor's vehicle_id, neighbor's grid pos), ...]
      nbrs_infos.append(nbrs)  # list of dicts, one for each sample/frame id

      if self.use_context:
        context_batch[sampleId,:,:,:] = torch.from_numpy(context)                
      for batch_ind, list_of_nbr in nbrs.items():
        for batch_id, vehid, grid_ind in list_of_nbr:          
          if batch_id >= 0:
            nbr_hist = hist[batch_id] # get the xy hist of the neighbor
            nbrs_batch[0:len(nbr_hist),count,:] = torch.from_numpy(nbr_hist)
            pos[0] = grid_ind % self.grid_size[0]
            pos[1] = grid_ind // self.grid_size[0]
            nbr_inds_batch[batch_ind,pos[1],pos[0],:] = torch.ones(self.enc_size*self.enc_fac).byte()
            count+=1

    return (hist_batch,
            nbrs_batch,
            nbr_inds_batch,
            fut_batch,
            mask_batch,
            context_batch,
            yaws_batch,
            nbrs_infos)


