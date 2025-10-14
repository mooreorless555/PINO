import torch

from models.utils import *
from models.cfg_sampler import ClassifierFreeSampleModel
from models.blocks import *
from utils.utils import *

from models.gaussian_diffusion import (
    MotionDiffusion,
    space_timesteps,
    get_named_beta_schedule,
    create_named_schedule_sampler,
    ModelMeanType,
    ModelVarType,
    LossType
)
import random

class MotionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.input_feats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION

        self.query_token = nn.Parameter(torch.randn(1, self.latent_dim))

        self.embed_motion = nn.Linear(self.input_feats*2, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=2000)

        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation,
                                                          batch_first=True)
        self.transformer = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=self.num_layers)
        self.out_ln = nn.LayerNorm(self.latent_dim)
        self.out = nn.Linear(self.latent_dim, 512)


    def forward(self, batch):
        x, mask = batch["motions"], batch["mask"]
        B, T, D  = x.shape

        x = x.reshape(B, T, 2, -1)[..., :-4].reshape(B, T, -1)

        x_emb = self.embed_motion(x)

        emb = torch.cat([self.query_token[torch.zeros(B, dtype=torch.long, device=x.device)][:,None], x_emb], dim=1)

        seq_mask = (mask>0.5)
        token_mask = torch.ones((B, 1), dtype=bool, device=x.device)
        valid_mask = torch.cat([token_mask, seq_mask], dim=1)

        h = self.sequence_pos_encoder(emb)
        h = self.transformer(h, src_key_padding_mask=~valid_mask)
        h = self.out_ln(h)
        motion_emb = self.out(h[:,0])

        batch["motion_emb"] = motion_emb

        return batch


class InterDenoiser(nn.Module):
    def __init__(self,
                 input_feats,
                 latent_dim=512,
                 num_frames=240,
                 ff_size=1024,
                 num_layers=8,
                 num_heads=8,
                 dropout=0.1,
                 activation="gelu",
                 cfg_weight=0.,
                 **kargs):
        super().__init__()

        self.cfg_weight = cfg_weight
        self.num_frames = num_frames
        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.activation = activation
        self.input_feats = input_feats
        self.time_embed_dim = latent_dim

        self.text_emb_dim = 768

        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, dropout=0)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        # Input Embedding
        self.motion_embed = nn.Linear(self.input_feats, self.latent_dim)
        self.text_embed = nn.Linear(self.text_emb_dim, self.latent_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(TransformerBlock(num_heads=num_heads,latent_dim=latent_dim, dropout=dropout, ff_size=ff_size))
        # Output Module
        self.out = zero_module(FinalLayer(self.latent_dim, self.input_feats))



    def forward(self, x, timesteps, mask=None, cond=None):
        """
        x: B, T, D
        """
        B, T = x.shape[0], x.shape[1]
        x_a, x_b = x[...,:self.input_feats], x[...,self.input_feats:]

        if mask is not None:
            mask = mask[...,0]

        emb = self.embed_timestep(timesteps) + self.text_embed(cond)

        a_emb = self.motion_embed(x_a)
        b_emb = self.motion_embed(x_b)
        h_a_prev = self.sequence_pos_encoder(a_emb)
        h_b_prev = self.sequence_pos_encoder(b_emb)

        if mask is None:
            mask = torch.ones(B, T).to(x_a.device)
        key_padding_mask = ~(mask > 0.5)

        for i,block in enumerate(self.blocks):
            h_a = block(h_a_prev, h_b_prev, emb, key_padding_mask)
            h_b = block(h_b_prev, h_a_prev, emb, key_padding_mask)
            h_a_prev = h_a
            h_b_prev = h_b

        output_a = self.out(h_a)
        output_b = self.out(h_b)

        output = torch.cat([output_a, output_b], dim=-1)

        return output



class InterDiffusion(nn.Module):
    def __init__(self, cfg, sampling_strategy: str = "ddim50", optimization: bool = False):
        super().__init__()
        self.cfg = cfg
        self.nfeats = cfg.INPUT_DIM
        self.latent_dim = cfg.LATENT_DIM
        self.ff_size = cfg.FF_SIZE
        self.num_layers = cfg.NUM_LAYERS
        self.num_heads = cfg.NUM_HEADS
        self.dropout = cfg.DROPOUT
        self.activation = cfg.ACTIVATION
        self.motion_rep = cfg.MOTION_REP

        self.cfg_weight = cfg.CFG_WEIGHT
        self.diffusion_steps = cfg.DIFFUSION_STEPS
        self.beta_scheduler = cfg.BETA_SCHEDULER
        self.sampling_strategy = sampling_strategy

        self.net = InterDenoiser(
            self.nfeats,
            self.latent_dim,
            ff_size=self.ff_size,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            dropout=self.dropout,
            activation=self.activation,
            cfg_weight=self.cfg_weight,
        )

        betas = get_named_beta_schedule(self.beta_scheduler, self.diffusion_steps)
        self.register_buffer("betas", torch.as_tensor(betas, dtype=torch.float32), persistent=False)

        self.diffusion = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, str(self.diffusion_steps)),
            betas=self.betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.schedule_sampler_name = cfg.SAMPLER
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_name, self.diffusion)

        self.diffusion_test: Optional[MotionDiffusion] = None
        self._last_sampling_strategy: Optional[str] = None
        self._build_test_diffuser(device=None, carry_over=False) 

        self.cfg_model = None
        self._last_cfg_weight = None

        self.optimization = optimization


    def mask_cond(self, cond, cond_mask_prob: float = 0.1, force_mask: bool = False):
        if cond is None:
            return None, None

        bs = cond.shape[0]
        device = cond.device
        view_shape = [bs] + [1] * (cond.ndim - 1)

        if force_mask:
            zero_cond = torch.zeros_like(cond)
            zero_mask = torch.zeros(view_shape, device=device, dtype=cond.dtype)
            return zero_cond, zero_mask

        if cond_mask_prob > 0.0:
            drop = torch.bernoulli(torch.full((bs,), cond_mask_prob, device=device)).view(view_shape) 
            cond_mask = (1.0 - drop).to(cond.dtype)
            return cond * cond_mask, cond_mask

        ones = torch.ones(view_shape, device=device, dtype=cond.dtype)
        return cond, ones

    def generate_src_mask(self, T: int, length):
        if not torch.is_tensor(length):
            length = torch.as_tensor(length, dtype=torch.long)
        device = length.device
        B = length.shape[0]
        t = torch.arange(T, device=device).unsqueeze(0)       # (1,T)
        valid = (t < length.unsqueeze(1)).float()             # (B,T)
        return valid.unsqueeze(-1).repeat(1, 1, 2)            # (B,T,2)

    def compute_loss(self, batch):
        x_start = batch["motions"]  # (B,T,feat)
        B, T = x_start.shape[:2]
        cond = batch.get("cond", None)


        cond_mask = None
        if cond is not None:
            cond, cond_mask = self.mask_cond(cond, cond_mask_prob=0.1)

        seq_mask = self.generate_src_mask(T, batch["motion_lens"]).to(x_start.device)

        t, _ = self.schedule_sampler.sample(B, x_start.device)

        output = self.diffusion.training_losses(
            model=self.net,
            x_start=x_start,
            t=t,
            mask=seq_mask,
            t_bar=getattr(self.cfg, "T_BAR", None),
            cond_mask=cond_mask,
            model_kwargs={
                "mask": seq_mask,
                "cond": cond,
            },
        )
        return output

    def _build_test_diffuser(self, device=None, carry_over: bool = True):
        timestep_respacing = self.sampling_strategy
        betas = self.betas if device is None else self.betas.to(device)

        new_diff = MotionDiffusion(
            use_timesteps=space_timesteps(self.diffusion_steps, timestep_respacing),
            betas=betas,
            motion_rep=self.motion_rep,
            model_mean_type=ModelMeanType.START_X,
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        if carry_over and self.diffusion_test is not None:
            new_diff.target_coords = getattr(self.diffusion_test, "target_coords", None)
            new_diff.coord_indices = getattr(self.diffusion_test, "coord_indices", None)

        self.diffusion_test = new_diff
        self._last_sampling_strategy = self.sampling_strategy

    def _maybe_refresh_test_diffuser(self, device):
        need_rebuild = (self.diffusion_test is None) or (self._last_sampling_strategy != self.sampling_strategy)

        curr_device = None
        if (self.diffusion_test is not None) and hasattr(self.diffusion_test, "betas"):
            try:
                curr_device = self.diffusion_test.betas.device
            except Exception:
                curr_device = None
        if (curr_device is not None) and (device is not None) and (curr_device != device):
            need_rebuild = True

        if need_rebuild:
            self._build_test_diffuser(device=device, carry_over=True)

    def _ensure_cfg_wrapper(self):
        if (self.cfg_model is None) or (self._last_cfg_weight != self.cfg_weight):
            self.cfg_model = ClassifierFreeSampleModel(self.net, self.cfg_weight)
            self._last_cfg_weight = self.cfg_weight

    def forward(self, batch):
        cond = batch.get("cond", None)
        motion_lens = batch["motion_lens"]
        noise = batch.get("noise", None)
        target_coords = batch.get("target_coords", None)
        coord_indices = batch.get("coord_indices", None)

        if not torch.is_tensor(motion_lens):
            motion_lens = torch.as_tensor(motion_lens, dtype=torch.long)

        B = int(motion_lens.shape[0]) if cond is None else int(cond.shape[0])
        T = int(motion_lens[0])

        device = cond.device if cond is not None else self.betas.device

        self._maybe_refresh_test_diffuser(device)
        self._ensure_cfg_wrapper()

        if target_coords is not None and coord_indices is not None:
            self.set_inpainting_motion(target_coords=target_coords, coord_indices=coord_indices)
        else:
            self.reset_inpainting_motion()

        output = self.diffusion_test.ddim_sample_loop(
            self.cfg_model,
            (B, T, self.nfeats * 2),
            noise=noise,
            clip_denoised=False,
            progress=False,
            model_kwargs={
                "mask": None,
                "cond": cond,
            },
            x_start=None,
            optimization=self.optimization,
        )
        return {"output": output}

    def set_inpainting_motion(self, target_coords=None, coord_indices=None):
        if self.diffusion_test is None:
            self._build_test_diffuser(device=None, carry_over=False)
        self.diffusion_test.target_coords = target_coords
        self.diffusion_test.coord_indices = coord_indices

    def reset_inpainting_motion(self):
        if self.diffusion_test is None:
            self._build_test_diffuser(device=None, carry_over=False)
        self.diffusion_test.target_coords = None
        self.diffusion_test.coord_indices = None

    def set_sampling_strategy(self, sampling_strategy: str):
        self.sampling_strategy = sampling_strategy
        self._build_test_diffuser(device=None, carry_over=True)


