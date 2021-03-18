from typing import List, Set, Dict, Tuple, Optional, Union, Any
import numpy as np
np.set_printoptions(suppress=1)
import time
import math
import glob
from attrdict import AttrDict
import gin
import torch
from torch.utils.data import DataLoader
from multiple_futures_prediction.dataset_carla import *
from multiple_futures_prediction.my_utils import *
from multiple_futures_prediction.train_carla import get_mean, Params, import_mfp_net

import matplotlib.pyplot as plt
import os
import subprocess


def generate_video(frames_dir='demo_files_carla2', video_name='demo_video.mp4', delete_frames=True):
  os.chdir(frames_dir)
  subprocess.call([
    'ffmpeg', '-framerate', '8', '-i', 'frame%03d.png', '-r', '30', '-pix_fmt', 'yuv420p',
    video_name
  ])
  if delete_frames:
    for file_name in glob.glob("*.png"):
      os.remove(file_name)

def video_from_images(image_fnames, video_fname, fps=15):
    # https://theailearner.com/2018/10/15/creating-video-from-images-using-opencv-python/
    img_array = []
    for f in image_fnames:
        img = cv2.imread(f)
        height, width, layers = img.shape
        size = (width, height)
        img_array.append(img)
    out = cv2.VideoWriter(video_fname, cv2.VideoWriter_fourcc(*'DIVX'), fps, size)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def load_mfp_model(checkpoint_dir, checkpoint='latest', config_file=None):
    """Load an MFP checkpoint into a torch object"""
    # Load the config that was used to train the model
    if not config_file:
      config_file = os.path.join(checkpoint_dir, 'config.gin')
    print("Loading MFP gin config %s..." % config_file)
    gin.parse_config_file(config_file)
    params = Params()()
    print("...done")
    if checkpoint == 'latest':
        # Find the final checkpoint file
        ckpts = glob.glob(os.path.join(checkpoint_dir, 'checkpoints/carla_*.pth'))
        ckpts = natsort.natsorted(ckpts)
        ckpt_file = ckpts[-1]
    elif checkpoint == 'first':
        # Find the first checkpoint file
        ckpts = glob.glob(os.path.join(checkpoint_dir, 'checkpoints/carla_*.pth'))
        ckpts = natsort.natsorted(ckpts)
        ckpt_file = ckpts[0]
    else:
        # Use the specified checkpoint
        ckpt_file = os.path.join(checkpoint_dir,
          'checkpoints/carla_{}.pth'.format(checkpoint))
    # Load the checkpoint from the model training
    mfpNet = import_mfp_net(params)
    net = mfpNet(params)
    if params.use_cuda:
      print("Moving params to cuda")
      net = net.cuda()
    print("Loading MFP checkpoint %s..." % ckpt_file)
    net.load_state_dict(torch.load(ckpt_file))
    net.eval()
    return net, params, ckpt_file

def demo(checkpoint_dir: str, checkpoint: str, config: str, outdir: str,
         n_frames: int) -> Any :
  """Demo/viz function."""

  ############################  
  net, params, pth_file = load_mfp_model(checkpoint_dir, checkpoint=checkpoint, config_file=config)

  ############################  
  torch.manual_seed( params.seed ) #type: ignore
  np.random.seed( params.seed )

  batch_size = 1
  data_hz                = 10
  ns_between_samples     = (1.0/data_hz)*1e9
  d_s = params.subsampling
  t_h = params.hist_len_orig_hz
  t_f = params.fut_len_orig_hz
  NUM_WORKERS = 1


  ROOT_PATH = 'multiple_futures_prediction/'  
  DATA_PATH = os.path.join(ROOT_PATH, 'carla_data_cfo')
  if params.scenario == 0:
    DATASET_DIR = os.path.join(DATA_PATH, 'Left_Turn_Dataset')
  elif params.scenario == 1:
    DATASET_DIR = os.path.join(DATA_PATH, 'Overtake_Dataset')
  elif params.scenario == 2:
    DATASET_DIR = os.path.join(DATA_PATH, 'Right_Turn_Dataset')
  else:
    raise ValueError(params.scenario)
  print("Loading dataset from:", str(os.path.join(DATASET_DIR, 'train')))
  
  # Loading the dataset.  
  # NOTE: need to turn off shuffling in the Dataset class, AND the pytorch dataloader
  train_set = CarlaDataset(
          str(os.path.join(DATASET_DIR, 'train')),
          t_h, t_f, d_s, params.encoder_size, params.use_gru, params.self_norm,
          params.data_aug, params.use_context, params.nbr_search_depth,
          rotate_fut=params.rotate_pov,
          shuffle=False)

  train_data_loader = DataLoader(
          train_set,batch_size=batch_size, shuffle=0, # NOTE no shuffle for demo
          num_workers=NUM_WORKERS, collate_fn=train_set.collate_fn, drop_last=True) # type: ignore

  # Compute or load existing mean over future trajectories.
  y_mean = get_mean(train_data_loader)
  net.y_mean = y_mean
  y_mean = torch.tensor(net.y_mean)

  train_loss: List = []
  val_loss: List = []
  
  num_updates = 0
  optimizer = None

  for epoch_num in range(1):
    for i, data in enumerate(train_data_loader):
      if i % 10 == 0:
        print("Frame %d / %d" % (i,len(train_data_loader)))
      plt.clf()

      st_time = time.time()
      hist, nbrs, mask, fut, mask, context, yaws, nbrs_info = data

      if params.remove_y_mean:
        # preprocess the (real) future traj before passing to model
        fut = fut-y_mean.unsqueeze(1)
     
      if params.use_cuda:
        hist = hist.cuda()
        nbrs = nbrs.cuda()
        mask = mask.cuda()
        fut = fut.cuda()
        if context is not None:
          context = context.cuda()
        if yaws is not None:
          yaws = yaws.cuda()

      # Forward pass.
      bStepByStep = True # we want interactive rollouts at test
      # Returns:
      #   fut_pred: a list of predictions, one for each mode.
      #   modes_pred: prediction over latent modes.
      #fut_preds, modes_pred = net.forward_mfp(hist, nbrs, mask, context, nbrs_info, fut, bStepByStep) 
      visualize = False
      if params.no_atten_model:
        fut_preds, modes_pred = net.forward_mfp(
          hist, nbrs, mask, context,
          nbrs_info, fut, bStepByStep, visualize=visualize,
          yaws=yaws, rotate_hist=params.rotate_pov)
      else:
        fut_preds, modes_pred = net.forward_mfp(
          hist, nbrs, mask, context,
          nbrs_info, fut, bStepByStep, visualize=visualize)

      ##### Plot the past ######
      hist_np = hist.cpu().detach().numpy()
      ref_pos = hist[-1,:,:] # final (i=0) x,y (i=2) for all vehicles (i=1)
      hist_ref_np = hist_np - ref_pos.view(1,-1,2).cpu().detach().numpy()
  
      fut_np = fut.cpu().detach().numpy()
      # NOTE: revert future to original state (since we preprocessed before passing to model)
      if params.remove_y_mean:
        fut_np += y_mean.unsqueeze(1).cpu().detach().numpy()

      # NOTE need to offset from the last position of history
      ref_pos_np = ref_pos.view(1,-1,2).cpu().detach().numpy()
      fut_np += ref_pos.view(1,-1,2).cpu().detach().numpy()

      ego_hist = hist_np[:,0,:]
      ego_fut = fut_np[:,0,:]
      
      # there may not be more than one vehicle
      if hist_np.shape[1] > 1:
        actor_hist = hist_np[:,1,:]
        actor_fut = fut_np[:,1,:]

      if params.rotate_pov:
        # Need to rotate the generated futures 
        # AND the "ground truth" futures since the
        # histories which were fed as inputs to the model were
        # rotated to POV view
        yaws_np = yaws.cpu().detach().numpy()
        ego_fut = rotate_np(ego_hist[-1,:],ego_fut,-yaws_np[0],degrees=False)
        actor_fut = rotate_np(actor_hist[-1,:],actor_fut,-yaws_np[1],degrees=False)
      else:
        # still needed for legend
        yaws_np = yaws.cpu().detach().numpy()

      plt.scatter(ego_hist[:,0],ego_hist[:,1],c='blue',label='ego (past)')
      plt.scatter(ego_fut[:,0],ego_fut[:,1],c='blue',alpha=0.5,label='ego (fut) [y={:.2f}]'.format(yaws_np[0]))
      if hist_np.shape[1] > 1:
        plt.scatter(actor_hist[:,0],actor_hist[:,1],c='red',label='actor (past)')
        plt.scatter(actor_fut[:,0],actor_fut[:,1],c='red',alpha=0.5,label='actor (fut) [y={:.2f}]'.format(yaws_np[1]))

      ##### Plot the predicted future ######

      assert len(modes_pred) == 2, len(modes_pred)
      ego_modes_pred = modes_pred[0].cpu().detach().numpy()
      actor_modes_pred = modes_pred[1].cpu().detach().numpy()

      for k in range(len(fut_preds)):
      
        assert fut_preds[k].shape == (30,2,5), fut_preds[k].shape
        fut_preds_mode_k = fut_preds[k].cpu().detach().numpy()[:,:,:2] # just u_x,u_y (30,2,5->30,2,2)

        if params.remove_y_mean:
          fut_preds_mode_k += y_mean.unsqueeze(1).cpu().detach().numpy() # 30,2,2
        # NOTE need to offset by final hist pos
        fut_preds_mode_k += ref_pos.view(1,-1,2).cpu().detach().numpy() 

        ego_fut_mode_k = fut_preds_mode_k[:,0,:] # 30,2
        actor_fut_mode_k = fut_preds_mode_k[:,1,:] # 30,2

        if params.rotate_pov:
          # Need to rotate the generated futures 
          # AND the "ground truth" futures since the
          # histories which were fed as inputs to the model were
          # rotated to POV view
          ego_fut_mode_k = rotate_np(
            ego_hist[-1,:],ego_fut_mode_k,-yaws_np[0],degrees=False)
          actor_fut_mode_k = rotate_np(
            actor_hist[-1,:],actor_fut_mode_k,-yaws_np[1],degrees=False)

        assert ego_modes_pred.shape == (len(fut_preds),), ego_modes_pred.shape
        ego_prob_mode_k = ego_modes_pred[k]
        assert actor_modes_pred.shape == (len(fut_preds),), actor_modes_pred.shape
        actor_prob_mode_k = actor_modes_pred[k]

        # to avoid clutter, only plot the futures with certain probability
        p_threshold = 0.01
        if ego_prob_mode_k > p_threshold:
          plt.scatter(ego_fut_mode_k[:,0],ego_fut_mode_k[:,1],
                      marker='^',alpha=0.25,
                      label='ego (m={},p={:.2f})'.format(k,ego_prob_mode_k))
        if actor_prob_mode_k > p_threshold:
          plt.scatter(actor_fut_mode_k[:,0],actor_fut_mode_k[:,1],
                      marker='^',alpha=0.25,
                      label='actor (m={},p={:.2f})'.format(k,actor_prob_mode_k))

      DIRNAME = outdir
      plt.legend()
      if params.scenario == 0: # left turn
        PADDING = 50
        plt.xlim(160,230+PADDING)
        plt.ylim(-180,-150+PADDING)
      elif params.scenario == 1: # overtake
        PADDING = 10
        plt.xlim(-90-PADDING,-85+PADDING)
        plt.ylim(-60-PADDING,5+6*PADDING) # needs extra padding in direction both cars are moving
      elif params.scenario == 2: # right turn
        PADDING = 25
        plt.xlim(160-3*PADDING,260) # needs extra padding in direction ego's turning / actor's moving
        plt.ylim(-210,-170+PADDING)

      plt.gca().invert_xaxis()
      plt.title("{}-{}".format(os.path.basename(checkpoint_dir), os.path.basename(pth_file)))
      plt.savefig(os.path.join(DIRNAME,"frame%03d.png") % i)
      if i == n_frames: break

    print("making video")
    generate_video(
      frames_dir=DIRNAME,
      video_name='demo_video_epoch{}_{}-{}.mp4'.format(
        epoch_num, os.path.basename(checkpoint_dir), os.path.basename(pth_file)),
      delete_frames=False
    )
