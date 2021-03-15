import sys
from typing import List, Set, Dict, Tuple, Optional, Union, Any
import torch
import torch.nn as nn
from multiple_futures_prediction.my_utils import *


# Multiple Futures Prediction Network
class mfpNet(nn.Module):    
  def __init__(self, args: Dict) -> None:
    super(mfpNet, self).__init__() #type: ignore
    self.use_cuda = args['use_cuda'] 
    self.encoder_size = args['encoder_size']
    self.decoder_size = args['decoder_size']        
    self.out_length = args['fut_len_orig_hz']//args['subsampling']

    self.dyn_embedding_size = args['dyn_embedding_size']
    self.input_embedding_size = args['input_embedding_size']

    #self.nbr_atten_embedding_size = args['nbr_atten_embedding_size']
    self.nbr_atten_embedding_size = self.input_embedding_size
    self.st_enc_hist_size = self.nbr_atten_embedding_size
    self.st_enc_pos_size = args['dec_nbr_enc_size'] 
    self.use_gru          = args['use_gru']
    self.bi_direc         = args['bi_direc']        
    self.use_context      = args['use_context']
    self.modes            = args['modes']
    self.use_forcing      = args['use_forcing'] # 1: Teacher forcing. 2:classmates forcing.
    
    self.hidden_fac     = 2 if args['use_gru'] else 1
    self.bi_direc_fac   = 2 if args['bi_direc'] else 1
    self.dec_fac        = 2 if args['bi_direc'] else 1   

    #self.init_rbf_state_enc( in_dim=self.encoder_size*self.hidden_fac )     
    #self.posi_enc_dim       = self.st_enc_pos_size
    self.posi_enc_dim       = 2
    self.posi_enc_ego_dim   = 2

    # Input embedding layer
    self.ip_emb = torch.nn.Linear(2,self.input_embedding_size) #type: ignore

    # Encoding RNN.
    if not self.use_gru:            
      self.enc_lstm = torch.nn.LSTM(
        self.input_embedding_size,self.encoder_size,1) # type: ignore
    else:
      self.num_layers=2
      self.enc_lstm = torch.nn.GRU(
        self.input_embedding_size,self.encoder_size, # type: ignore 
        num_layers=self.num_layers, bidirectional=False) 

    # Dynamics embeddings.
    self.dyn_emb = torch.nn.Linear(
      self.encoder_size*self.hidden_fac, self.dyn_embedding_size) #type: ignore

    context_feat_size = 64 if self.use_context else 0
    self.dec_lstm = []
    self.op = []
    for k in range(self.modes):            
      if not self.use_gru:
        self.dec_lstm.append(
          torch.nn.LSTM(
            self.nbr_atten_embedding_size + self.dyn_embedding_size + #type: ignore
            context_feat_size + self.posi_enc_dim + self.posi_enc_ego_dim,
            self.decoder_size))
      else:
        # Decoding RNN (one per mode)
        self.num_layers=2
        self.dec_lstm.append(
          torch.nn.GRU(
            self.nbr_atten_embedding_size + self.dyn_embedding_size +
            context_feat_size + self.posi_enc_dim + self.posi_enc_ego_dim, # type: ignore 
            self.decoder_size,
            num_layers=self.num_layers, bidirectional=self.bi_direc))
      
      self.op.append( torch.nn.Linear(self.decoder_size*self.dec_fac, 5) ) #type: ignore
      
      self.op[k] = self.op[k]
      self.dec_lstm[k] = self.dec_lstm[k]

    self.dec_lstm = torch.nn.ModuleList(self.dec_lstm) # type: ignore 
    self.op       = torch.nn.ModuleList(self.op)       # type: ignore

    self.op_modes = torch.nn.Linear(
      self.nbr_atten_embedding_size + self.dyn_embedding_size + context_feat_size,
      self.modes) #type: ignore

    # Nonlinear activations.
    self.leaky_relu = torch.nn.LeakyReLU(0.1) #type: ignore
    self.relu = torch.nn.ReLU() #type: ignore
    self.softmax = torch.nn.Softmax(dim=1) #type: ignore

    # NOTE: shapes modified for CARLA context
    if self.use_context:          
      self.context_conv    = torch.nn.Conv2d(1, 16, kernel_size=3, stride=2)  #type: ignore
      self.context_conv2   = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2) #type: ignore
      self.context_maxpool = torch.nn.MaxPool2d(kernel_size=(4,2))            #type: ignore
      self.context_conv3   = torch.nn.Conv2d(16, 16, kernel_size=3, stride=2) #type: ignore
      self.context_fc      = torch.nn.Linear(16*21*1, context_feat_size)      #type: ignore
  
  def forward_mfp(self,
        hist:torch.Tensor,
        nbrs:torch.Tensor,
        masks:torch.Tensor,
        context:Any, 
        nbrs_info:List,
        fut:torch.Tensor,
        bStepByStep:bool, 
        use_forcing:Optional[Union[None,int]]=None,
        visualize:Optional[bool]=False,
        rotate_hist:Optional[bool]=False,
        yaws:Optional[Union[None,torch.Tensor]]=None, # NOTE: needed if we want to rotate
    ) -> Tuple[List[torch.Tensor], Any]:
    """Forward propagation function for the MFP
    
    Computes dynamic state encoding with precomputed attention tensor and the 
    RNN based encoding.
    Args:
      hist: Trajectory history.
      nbrs: Neighbors.
      masks: Neighbors mask.
      context: contextual information in image form (if used).
      nbrs_info: information as to which other agents are neighbors.
      fut: Future Trajectory.
      bStepByStep: During rollout, interactive or independent.
      use_forcing: Teacher-forcing or classmate forcing.

    Returns:
      fut_pred: a list of predictions, one for each mode.
      modes_pred: prediction over latent modes.    
    """
    use_forcing = self.use_forcing if use_forcing==None else use_forcing
    if yaws is not None:
      assert yaws.shape == (2,), yaws.shape
    assert hist.shape[1] == 2, "currently only supports 2 agents"

    # Normalize to reference position.
    ref_pos = hist[-1,:,:]
    hist_shifted = hist - ref_pos.view(1,-1,2)
    if rotate_hist:
      assert yaws is not None
      # Rotate the hist so that each ego if facing forward
      # where 'forward' indicates +x-axis direction
      # NOTE: rotate around the ego's yaw and ref pos
      hist_shifted = torch.stack([
        rotate_torch(
          hist_shifted[-1,i,:], # NOTE: this should be (0,0) for all
          hist_shifted[:,i,:],
          yaws[i],
          degrees=False)
        for i in range(hist_shifted.shape[1])
        ], dim=1)
      assert hist_shifted.shape == hist.shape, (hist_shifted.shape,hist.shape)

    ### Encode history trajectories.
    if isinstance(self.enc_lstm, torch.nn.modules.rnn.GRU):
      _, hist_enc = self.enc_lstm(
        self.leaky_relu(self.ip_emb(hist_shifted)))
    else:
      _,(hist_enc,_) = self.enc_lstm(
        self.leaky_relu(self.ip_emb(hist_shifted)))

    if self.use_gru:
      hist_enc = hist_enc.permute(1,0,2).contiguous()
      hist_enc = self.leaky_relu(self.dyn_emb(
        hist_enc.view(hist_enc.shape[0], -1) ))
    else:
      hist_enc = self.leaky_relu(self.dyn_emb(
        hist_enc.view(hist_enc.shape[1],hist_enc.shape[2])))

    ### Encode neighbor history trajectories.
    # Need to shift (and potentially rotate) the neighbor's history
    # with each ego's perspective
    nbr_hist = torch.flip(hist, [1]) # 0's neighbor is 1, 1's neighbor is 0
    nbr_hist_shifted = nbr_hist - ref_pos.view(1,-1,2) # shifting on nbr's ref_pos
    if rotate_hist:
      # Rotate the hist of the neighbor/other agent
      # based on the initial rotation of the ego
      # NOTE: rotate around the ego's yaw and ref pos
      nbr_hist_shifted = torch.stack([
        rotate_torch(
          hist_shifted[-1,i,:], # NOTE: this should be (0,0) for all
          nbr_hist_shifted[:,i,:],
          yaws[i],
          degrees=False)
        for i in range(hist_shifted.shape[1])
        ], dim=1)
      assert nbr_hist_shifted.shape == hist.shape, (nbr_hist_shifted.shape,hist.shape)

    # Encode the neighbor's (ie other agent's) hist the same way
    # No need for attention since there's only one other agent
    if isinstance(self.enc_lstm, torch.nn.modules.rnn.GRU):
      _, nbr_enc = self.enc_lstm(
        self.leaky_relu(self.ip_emb(nbr_hist_shifted)))
    else:
      _,(nbr_enc,_) = self.enc_lstm(
        self.leaky_relu(self.ip_emb(nbr_hist_shifted)))

    if self.use_gru:
      nbr_enc = nbr_enc.permute(1,0,2).contiguous()
      nbr_enc = self.leaky_relu(self.dyn_emb(
        nbr_enc.view(nbr_enc.shape[0], -1) ))
    else:
      nbr_enc = self.leaky_relu(self.dyn_emb(
        nbr_enc.view(nbr_enc.shape[1],nbr_enc.shape[2])))

    ### Concat all the inputs for the decoder (which outputs trajectories)
    if self.use_context == True:
      context_enc = self.relu(self.context_conv( context ))        
      context_enc = self.context_maxpool( self.context_conv2( context_enc ))
      context_enc = self.relu(self.context_conv3(context_enc))            
      context_enc = self.context_fc( context_enc.view( context_enc.shape[0], -1) )
      enc = torch.cat((nbr_enc, hist_enc, context_enc),1)
    else:
      # concat ([2,80], [2,32])
      enc = torch.cat((nbr_enc, hist_enc),1)
   
    ############################################################################
    modes_pred = None if self.modes==1 else self.softmax(self.op_modes(enc))
    fut_pred = self.decode(
        enc, None, None, ref_pos, fut, bStepByStep, use_forcing,
        ref_yaws=yaws)
    # Returns:
    #
    #   fut_pred: a list of predictions, one for each mode.
    #             len(fut_pred) = N_MODES
    #             fut_pred[MODE].shape = (HORIZON, N_ACTORS, 5)
    #             5 because [horizon, n_agents, 5 pair x_u/y_u/x_sig/y_sig/rho]
    #
    #   modes_pred: prediction over latent modes.
    #               len(modes_pred) = N_ACTORS
    #               modes_pred[ACTOR].shape = (N_MODES,)
    #               modes_pred[ACTOR].sum() = 1.0

    # Visualizing the pre- and post-shift trajs for debug
    #visualize=True
    if visualize:
      # Copy from GPU to numpy
      hist_np = hist.cpu().detach().numpy()
      hist_shifted_np = hist_shifted.cpu().detach().numpy()
      nbrs_hist_np = nbr_hist.cpu().detach().numpy()
      nbrs_hist_shifted_np = nbr_hist_shifted.cpu().detach().numpy()
      fut_pred_np = [t.cpu().detach().numpy() for t in fut_pred]

      # Plot the different views that were fed into the encoder (pre and postshift)
      fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(1, 5)
      ax0.set_title('raw hist') 
      ax0.scatter(hist_np[:,0,0],hist_np[:,0,1],c='blue',label='ego (hist)')
      ax0.scatter(hist_np[:,1,0],hist_np[:,1,1],c='red',label='actor (hist)')

      batch_axis = 0
      ax1.set_title('ego POV (preshift)') 
      ax1.scatter(hist_np[:,batch_axis,0],hist_np[:,batch_axis,1],c='blue',label='ego (hist)')
      ax1.scatter(nbrs_hist_np[:,batch_axis,0],nbrs_hist_np[:,batch_axis,1],
                  c='purple',label='nbr0',marker='^')

      ax2.set_title('ego POV (postshift r={:.2f})'.format(yaws[0]))
      ax2.scatter(hist_shifted_np[:,batch_axis,0],hist_shifted_np[:,batch_axis,1],
                  c='blue',label='ego (hist)')
      ax2.scatter(nbrs_hist_shifted_np[:,batch_axis,0],nbrs_hist_shifted_np[:,batch_axis,1],
                  c='purple',label='nbr0',marker='^')

      for m,fut in enumerate(fut_pred_np):
        # fut_pred[MODE].shape = (HORIZON, N_ACTORS, 5)
        ax2.scatter(fut[:,batch_axis,0],
                    fut[:,batch_axis,1],
                    label='ego (fut-m{})'.format(m),alpha=0.5)

      batch_axis = 1
      ax3.set_title('actor POV (preshift)') 
      ax3.scatter(hist_np[:,batch_axis,0],hist_np[:,batch_axis,1],c='red',label='actor (hist)')
      ax3.scatter(nbrs_hist_np[:,batch_axis,0],nbrs_hist_np[:,batch_axis,1],
                  c='orange',label='nbr1',marker='^')

      ax4.set_title('actor POV (postshift r={:.2f})'.format(yaws[1])) 
      ax4.scatter(hist_shifted_np[:,batch_axis,0],hist_shifted_np[:,batch_axis,1],
                  c='red',label='actor (hist)')
      ax4.scatter(nbrs_hist_shifted_np[:,batch_axis,0],nbrs_hist_shifted_np[:,batch_axis,1],
                  c='orange',label='nbr1',marker='^')

      for m,fut in enumerate(fut_pred_np):
        # fut_pred[MODE].shape = (HORIZON, N_ACTORS, 5)
        ax4.scatter(fut[:,batch_axis,0],
                    fut[:,batch_axis,1],
                    label='actor (fut-m{})'.format(m),alpha=0.5)

      # preshift bounds
      for ax in [ax0,ax1,ax3]:
        x_min = 150
        x_max = 240
        y_min = -175
        y_max = -145
        ax.set_xlim(x_min,x_max)
        ax.set_ylim(y_min,y_max)
      # postshift bounds
      for ax in [ax2,ax4]:
        x_scale = x_max-x_min
        y_scale = y_max-y_min
        ax.set_xlim(-x_scale//2,x_scale//2)
        ax.set_ylim(-y_scale//2,y_scale//2)
      ax1.legend()
      ax2.legend()
      ax3.legend()
      ax4.legend()
      plt.show()

    return fut_pred, modes_pred
 
  def decode(self,
    enc: torch.Tensor,
    attens:List,
    nbrs_info_this:List,
    ref_pos:torch.Tensor,
    fut:torch.Tensor,
    bStepByStep:bool,
    use_forcing:Any,
    ref_yaws:torch.Tensor, # needed if StepByStep=True
    visualize:Optional[bool]=False,
    ) -> List[torch.Tensor]:    
    #visualize=True; use_forcing=1 # for testing
    #ref_yaws=None # for isolating rotation vs shifting
    """Decode the future trajectory using RNNs.
    
    Given computed feature vector, decode the future with multimodes, using
    dynamic attention and either interactive or non-interactive rollouts.
    Args:
      enc: encoded features, one per agent.
      attens: attentional weights, list of objs, each with dimenstion of [8 x 4] (e.g.)
      nbrs_info_this: information on who are the neighbors
      ref_pos: the current postion (reference position) of the agents.
      fut: future trajectory (only useful for teacher or classmate forcing)
      bStepByStep: interactive or non-interactive rollout
      use_forcing: 0: None. 1: Teacher-forcing. 2: classmate forcing.

    Returns:
      fut_pred: a list of predictions, one for each mode.
      modes_pred: prediction over latent modes.    
    """
    if not bStepByStep: # Non-interactive rollouts
      # NOTE: since there's no real "input sequence" (instead,
      # the encoder just generated a fixed-len feature),
      # this means that we just stack the encoding to the size
      # that we want the RNN outputs to be, i.e., use
      # it as the 'ground-truth' input for each time t
      enc = enc.repeat(self.out_length, 1, 1)
      pos_enc = torch.zeros(
        self.out_length,
        enc.shape[1], # this is from every agent's perspective
        2 + 2, # NOTE: xy for actor/neighbor, and xy for ego
        device=enc.device)
      enc2 = torch.cat( (enc, pos_enc), dim=2)                

      fut_preds = []
      for k in range(self.modes):
        h_dec, _ = self.dec_lstm[k](enc2)
        h_dec = h_dec.permute(1, 0, 2)
        fut_pred = self.op[k](h_dec)
        fut_pred = fut_pred.permute(1, 0, 2) #torch.Size([nSteps, num_agents, 5])

        fut_pred = Gaussian2d(fut_pred)
        fut_preds.append(fut_pred)
      return fut_preds

    else: # Interactive rollout
      batch_sz = enc.shape[0]
      fut_preds = []

      if visualize:
        pred_fut_t_global_povs = np.zeros((self.modes,self.out_length,2,2)) # K, T, n_agents, xy
        pred_fut_t_ego_povs = np.zeros((self.modes,self.out_length,2,1,2)) # K, T, n_ego, n_other, xy
        ego_fut_ts = np.zeros((self.modes,self.out_length,2,2)) # K, T, n_ego, xy

      for k in range(self.modes):
        direc = 2 if self.bi_direc else 1
        # Start with zero'd hidden state
        hidden = torch.zeros(self.num_layers*direc, batch_sz, self.decoder_size).to(fut.device)
        preds = []

        for t in range(self.out_length):
          '''
          No forcing:
            Use the yt values generated by the RNN as the input to step t+1
          Teacher forcing:
            Use ground truth ('fut') values yt as the input to step t+1
          Classmate forcing:
            At time t, for agent n, use ground truth observations as inputs
            for all other agents ym_t, where m!=n.
            However, for agent n itself, use its previous predicted state
            instead of the true observations xn_t as its input
          '''
          if t == 0: # Intial timestep.
            # No forcing
            if use_forcing == 0:
              pred_fut_t = torch.zeros_like(fut[t,:,:]) # no predictions for t=0
              ego_fut_t = pred_fut_t
            # Teacher forcing
            elif use_forcing == 1:
              pred_fut_t = fut[t,:,:] # can use ground-truth at t=0
              ego_fut_t = pred_fut_t
            # Classmate forcing
            elif use_forcing == 2:
              pred_fut_t = fut[t,:,:] # use ground truth
              ego_fut_t =  torch.zeros_like(pred_fut_t) # use real predictions, but none to use
            else:
              raise ValueError(use_forcing)
          else:
            # No forcing
            if use_forcing == 0:
              pred_fut_t = preds[-1][:,:,:2].squeeze() # this is in xy format
              ego_fut_t = pred_fut_t # use t-1 preds as input
            # Teacher forcing
            elif use_forcing == 1:
              pred_fut_t = fut[t,:,:]
              ego_fut_t = pred_fut_t
            # Classmate forcing
            elif use_forcing == 2:
              pred_fut_t = fut[t,:,:]
              ego_fut_t = preds[-1][:,:,:2] 
            else:
              raise ValueError(use_forcing)

          assert batch_sz == 2, batch_sz
          assert pred_fut_t.shape == (batch_sz, 2), pred_fut_t.shape
          assert ref_pos.shape == (batch_sz, 2), ref_pos.shape

          # (1) convert back to global coords
          pred_fut_t_global_pov = pred_fut_t + ref_pos 
          if ref_yaws is not None:
            # NOTE: rotate the predicted point around its own start (0,0), using its own yaw
            pred_fut_t_global_pov = torch.stack([
              rotate_torch(
                ref_pos[i,:], # origin
                pred_fut_t_global_pov[i,:], # point(s)
                -ref_yaws[i], # angle
                degrees=False)
              for i in range(batch_sz)
            ], dim=0) # dim=0 or 1?
          assert pred_fut_t_global_pov.shape == (batch_sz, 2)

          # (2) for each agent (ego), convert the other agents' coords into the ego's ref_pos
          # Since we only have two agents (ego and actor), we can just set each to the opposite
          pred_fut_t_ego_pov = torch.zeros((
            batch_sz, # n_agents
            batch_sz-1, # n_agents-1 (ie, the number of neighbors each agent has)
            2, # xy
          ), device=enc.device) # 2,1,2
          # NOTE: if we're going to rotate w.r.t. an origin in global space,
          # we need to rotate FIRST, THEN shift back to 0-centered space

          pred_fut_t_ego_pov[0,0,:] = pred_fut_t_global_pov[1,:]
          if ref_yaws is not None:
            pred_fut_t_ego_pov[0,0,:] = rotate_torch(
              ref_pos[0,:], # origin
              pred_fut_t_ego_pov[0,0,:], # points
              ref_yaws[0], # angle
              degrees=False)
          pred_fut_t_ego_pov[0,0,:] -= ref_pos[0,:]

          pred_fut_t_ego_pov[1,0,:] = pred_fut_t_global_pov[0,:]
          if ref_yaws is not None:
            pred_fut_t_ego_pov[1,0,:] = rotate_torch(
              ref_pos[1,:], # origin
              pred_fut_t_ego_pov[1,0,:], # points
              ref_yaws[1], # angle
              degrees=False)
          pred_fut_t_ego_pov[1,0,:] -= ref_pos[1,:]

          if visualize:
            pred_fut_t_global_povs[k,t,:,:] = pred_fut_t_global_pov.detach().cpu().numpy()
            pred_fut_t_ego_povs[k,t,:,:,:] = pred_fut_t_ego_pov.detach().cpu().numpy()
            ego_fut_ts[k,t,:,:] = ego_fut_t.detach().cpu().numpy()
          
          enc_large = torch.cat((
            enc.view(1,enc.shape[0],enc.shape[1]), 
            pred_fut_t_ego_pov.view(1, batch_sz, 2), # everyone else's predicted xy (for each ego)
            ego_fut_t.view(1, batch_sz, 2) # your own predicted xy (for each ego)
          ), dim=2)

          out, hidden = self.dec_lstm[k](enc_large, hidden)
          pred = Gaussian2d(self.op[k](out))
          preds.append(pred)

        fut_pred_k = torch.cat(preds,dim=0)
        fut_preds.append(fut_pred_k)

      if visualize:
        for k in range(self.modes):
          # plot the predicted futures from each POV
          plt.clf()
          fig, axs = plt.subplots(1, 3)

          axs[0].set_title('mode %d, global POV' % k)
          axs[0].scatter(pred_fut_t_global_povs[k,:,0,0],
                         pred_fut_t_global_povs[k,:,0,1],
                         c='blue',label='agent0 (global POV)')
          axs[0].scatter(pred_fut_t_global_povs[k,:,1,0],
                         pred_fut_t_global_povs[k,:,1,1],
                         c='red',label='agent1 (global POV)')
          axs[0].scatter(pred_fut_t_global_povs[k,0,0,0],
                         pred_fut_t_global_povs[k,0,0,1],
                         c='yellow') # starting point
          axs[0].scatter(pred_fut_t_global_povs[k,0,1,0],
                         pred_fut_t_global_povs[k,0,1,1],
                         c='yellow') # starting point
          axs[0].legend()

          agent = 0
          axs[1].set_title('mode %d, agent0 POV (y=%f)' % (k,ref_yaws[agent]))

          axs[1].scatter(ego_fut_ts[k,:,agent,0],
                         ego_fut_ts[k,:,agent,1],
                         c='blue',label='agent0 (agent0 POV)')
          axs[1].scatter(ego_fut_ts[k,0,agent,0],
                         ego_fut_ts[k,0,agent,1],
                         c='yellow')
          axs[1].scatter(pred_fut_t_ego_povs[k,:,agent,:,0],
                         pred_fut_t_ego_povs[k,:,agent,:,1],
                         c='red',label='agent1 (agent0 POV)')
          axs[1].scatter(pred_fut_t_ego_povs[k,0,agent,:,0],
                         pred_fut_t_ego_povs[k,0,agent,:,1],
                         c='yellow')
          axs[1].legend()

          agent = 1
          axs[2].set_title('mode %d, agent1 POV (y=%f)' % (k,ref_yaws[agent]))
          axs[2].scatter(ego_fut_ts[k,:,agent,0],
                         ego_fut_ts[k,:,agent,1],
                         c='red',label='agent1 (agent1 POV)')
          axs[2].scatter(ego_fut_ts[k,0,agent,0],
                         ego_fut_ts[k,0,agent,1],
                         c='yellow')
          axs[2].scatter(pred_fut_t_ego_povs[k,:,agent,:,0],
                         pred_fut_t_ego_povs[k,:,agent,:,1],
                         c='blue',label='agent0 (agent1 POV)')
          axs[2].scatter(pred_fut_t_ego_povs[k,0,agent,:,0],
                         pred_fut_t_ego_povs[k,0,agent,:,1],
                         c='yellow')
          axs[2].legend()

          plt.show()

      return fut_preds

