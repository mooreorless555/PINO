import argparse
import copy
import os.path
import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning as L
import scipy.ndimage.filters as filters
import numpy as np
from os.path import join as pjoin
from models import *
from collections import OrderedDict
from configs import get_config
from utils.plot_script import *
from utils.preprocess import *
from utils import paramUtil
from utils.utils import MotionNormalizerTorch
from utils.preprocess import MotionNormalizer

from torch import optim
from torch.optim.lr_scheduler import LambdaLR

from atomic_lib.math_utils import *
from atomic_lib.basic_loss import *

class LitGenModel(L.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        self.cfg = cfg

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model
        self.normalizer = MotionNormalizer()

        self.nfeats = cfg.OPT.INPUT_DIM
        self.lr = cfg.OPT.LR
        self.iter = cfg.OPT.ITER
        self.early_stop = float(cfg.OPT.EARLY_STOP)


    def plot_t2m(self, mp_data, result_path, caption):
        mp_joint = []
        for i, data in enumerate(mp_data):
            if i == 0:
                joint = data[:,:22*3].reshape(-1,22,3)
            else:
                joint = data[:,:22*3].reshape(-1,22,3)

            mp_joint.append(joint)

        plot_3d_motion(result_path, paramUtil.t2m_kinematic_chain, mp_joint, title=caption, fps=30)

    def generate_one_sample(self, prompt, name, motion_data=None, output_dir="results"):
        self.model.eval()
        batch = OrderedDict({})

        batch["motion_lens"] = torch.zeros(1, 1).long().to(self.device)
        batch["prompt"] = prompt
        
        joints = []
        if motion_data is not None:
            # Pivoting on the first person (index 0) for generation
            sliced_motion_data = motion_data[:, :, :self.nfeats*2]
            batch["target_coords"] = sliced_motion_data
            # Set the indices for the features of the first person
            batch["coord_indices"] = list(range(0, self.nfeats))

            # Prepare existing motion for plotting
            joints_list = motion_prepare(motion_data, self.device)
            joints = [t.cpu().numpy() for t in joints_list]

        window_size = 210 
        
        output_tensor = self.generate_loop(batch, window_size, motion_data)
        os.makedirs(output_dir, exist_ok=True)
        
        if motion_data is not None:
            # Case: Add a new person to existing motion
            num_existing_people = motion_data.shape[2] // self.nfeats
            new_person_motion = output_tensor[:, :, self.nfeats : self.nfeats * 2]
            combined_motion = torch.cat([motion_data.cpu(), new_person_motion], dim=2)
            
            total_people = num_existing_people + 1
            save_filename = pjoin(output_dir, f"{name}_{total_people}.pt")
            torch.save(combined_motion, save_filename)
            print(f"Saved combined motion for {total_people} people to {save_filename}")
        else:
            # Case: Generate a new 2-person motion from scratch
            save_filename = pjoin(output_dir, f"{name}_2.pt")
            torch.save(output_tensor, save_filename)
            print(f"Saved generated motion for 2 people to {save_filename}")

        sequences = [[], []]
        motion_output_both = output_tensor.reshape(output_tensor.shape[1], 2, -1)
        motion_output_both = self.normalizer.backward(motion_output_both.numpy())

        for j in range(2):
            motion_output = motion_output_both[:, j]
            
            joints3d = motion_output[:, :22*3].reshape(-1, 22, 3) 
            joints3d = filters.gaussian_filter1d(joints3d, 1, axis=0, mode='nearest')
            sequences[j].append(joints3d)

        sequences[0] = np.concatenate(sequences[0], axis=0)
        sequences[1] = np.concatenate(sequences[1], axis=0)

        joints_to_plot = []
        if motion_data is not None:
            joints_to_plot.extend(joints) 
            joints_to_plot.append(sequences[1]) # Add the newly generated person
            
            total_people = len(joints_to_plot)
            result_path = pjoin(output_dir, f"{name}_{total_people}.mp4")
        else:
            joints_to_plot.append(sequences[0]) # Add person 1
            joints_to_plot.append(sequences[1]) # Add person 2
            
            result_path = pjoin(output_dir, f"{name}_2.mp4")

        self.plot_t2m(joints_to_plot,
                      result_path,
                      batch["prompt"])

    def generate_loop(self, batch, window_size, motion_data=None):
        prompt = batch["prompt"]
        batch = copy.deepcopy(batch)
        batch["motion_lens"][0, 0] = window_size

        shape = (1, window_size, self.nfeats * 2)
        device = next(self.model.parameters()).device
        
        noise = torch.randn(1, window_size, self.nfeats*2, device=device)
        
        if motion_data is not None:
            mask = torch.zeros_like(noise)
            mask[:, :, batch["coord_indices"]] = 1.0
            sliced_motion_data = batch["target_coords"]
            noise = (noise * (1.0 - mask)) + (sliced_motion_data.to(device) * mask)

        noise.requires_grad = True
        batch["noise"] = noise
        
        optimizer = optim.Adam([noise], lr=self.lr)

        batch["text"] = [prompt]

        # noise optimization
        for it in range(self.iter):
            lr_current = optimizer.param_groups[0]["lr"]
            optimizer.zero_grad()
            batch = self.model.forward_test(batch)
            output = batch["output"]
            motion_output_both = output.reshape(output.shape[1], -1, self.nfeats)
            motion_output_both = self.normalizer.backward(motion_output_both)

            loss = self.f_loss(motion_output_both, motion_data=motion_data)

            print(f"iter={it}, lr={lr_current:.4f}, loss={loss.item():.6f}", flush=True)

            if loss.item() < self.early_stop:
                break

            loss.backward(retain_graph=True)
            optimizer.step()
            
        output_tensor = batch["output"].detach().cpu()
        return output_tensor
    
    def f_loss(self, motion_output_both, motion_data=None):
        #(210, 2, 262) (frame, human_idx, feature)
        
        length = motion_output_both.shape[0]

        joints_list = []
        for j in range(2):
            motion_output = motion_output_both[:, j]
            joints3d = motion_output[:, :22*3].reshape(-1, 22, 3)
            joints_list.append(joints3d)
        joints1, joints2 = joints_list
        device = joints1.device
        root_1 = dimXZ(get_joint(joints1, PELVIS))
        root_2 = dimXZ(get_joint(joints2, PELVIS))

        # overlap
        loss = motion_overlap_loss(joints1,joints2,motion_data=motion_data, inter_motion_dist_min=0.5, inter_motion_dist_max=4.0, other_motion_distance=0.5)
        
        # root position
        # loss += start_end_pos_loss(
        #     joints1, joints2,
        #     gt_start_pos1=(-1.0, 0.0),   # person1 start XZ
        #     gt_start_pos2=(1.0, 0.0),   # person2 start XZ
        #     margin=0.01
        # )
    
        # region
        # loss += calc_region_loss(joints1, region_type='rectangle',inside=True, center=torch.tensor([0.0, 0.0]), width=6.0, height=6.0) + \
        #             calc_region_loss(joints2, region_type='rectangle',inside=True, center=torch.tensor([0.0, 0.0]), width=6.0, height=6.0)

        return loss

def build_models(cfg):
    if cfg.NAME == "InterGen":
        model = InterGen(cfg, optimization=True)
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate motion from a text prompt.")
    parser.add_argument('--prompt', type=str, required=True, 
                        help='The text prompt describing the motion to generate.')
    
    parser.add_argument('--motion_path', type=str, default=None,
                        help='(Optional) Path to a pre-existing motion tensor (.pt file) to add a person to.')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='(Optional) Directory to save the generated motion tensor and video.')
    
    args = parser.parse_args()

    # torch.manual_seed(37)
    model_cfg = get_config("configs/model.yaml")
    infer_cfg = get_config("configs/infer.yaml")

    model = build_models(model_cfg)

    if model_cfg.CHECKPOINT:
        ckpt = torch.load(model_cfg.CHECKPOINT, map_location="cpu")
        new_state_dict = OrderedDict()
        for k, v in ckpt["state_dict"].items():
            if "model." in k:
                name = k.replace("model.", "")
                new_state_dict[name] = v
            else:
                 new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict=False)
        print("checkpoint state loaded!")

    litmodel = LitGenModel(model, infer_cfg).to(torch.device("cuda:0"))

    motion_data_tensor = None
    if args.motion_path:
        if os.path.exists(args.motion_path):
            try:
                motion_data_tensor = torch.load(args.motion_path, map_location="cpu")
                print(f"Successfully loaded motion data from {args.motion_path} with shape: {motion_data_tensor.shape}")
            except Exception as e:
                print(f"Error loading motion data from {args.motion_path}: {e}")
                sys.exit(1)
        else:
            print(f"Error: Motion file not found at {args.motion_path}")
            sys.exit(1)

    text = args.prompt
    name = text.replace(' ', '_').replace('.', '').replace(',', '')[:48]
    
    litmodel.generate_one_sample(text, name, motion_data=motion_data_tensor, output_dir=args.output_dir)