from flax.training import checkpoints
import shutil
import os

def save_checkpoint(state, ckpt_dir, keep=2):
    if os.path.exists(ckpt_dir):
        shutil.rmtree(ckpt_dir)
    checkpoints.save_checkpoint(ckpt_dir=ckpt_dir, 
                                target=state, 
                                step=state.step, 
                                keep=keep)
    
def load_checkpoint(state, ckpt_dir):
  return checkpoints.restore_checkpoint(ckpt_dir, state)