from typing import List, Set, Dict, Tuple, Optional, Union, Any
import os
from multiple_futures_prediction.demo_carla import demo, Params
import gin
import argparse

def parse_args() -> Any:
  parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument('--checkpoint-dir', type=str, default='') # eg CARLA_2020.x.x....
  parser.add_argument('--checkpoint', type=str, default='latest') # eg 'latest', 004999, ...
  parser.add_argument('--outdir', type=str, default='')
  parser.add_argument('--config', type=str, default='')
  parser.add_argument('--frames', type=int, default=200) # number of feed_dicts/frames to use
  return parser.parse_args()

def main() -> None:
  args = parse_args()

  if os.path.isdir(args.outdir):
    print("Warning: dir %s already exists!" % args.outdir)
  else:
    # create
    os.mkdir(args.outdir)

  checkpoint_dir = os.path.join('multiple_futures_prediction','checkpts',args.checkpoint_dir)
  demo(checkpoint_dir, args.checkpoint, args.config, args.outdir, args.frames)

# python -m multiple_futures_prediction.cmd.train_carla_cmd --config multiple_futures_prediction/configs/mfp2_carla.gin
if __name__ == '__main__':
  main()
