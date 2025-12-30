import torch
import torch.nn as nn
from typing import List
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from skills.autoencoder import Autoencoder
import torch.nn.functional as F
from skills.skill_interface import Skill
import numpy as np
import math


def get_embedding_for_context(
        observations: torch.Tensor,
        encoder
    ) -> torch.Tensor:
        """Extract context for routing decisions"""

        with torch.no_grad():
            z = encoder(observations)
            z = torch.reshape(z, (z.size(0), -1))

        return z


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        skills: List[Skill] = None,
        device="cpu",
    ):
        super().__init__(observation_space, features_dim)

        self.skills = skills

        # TODO: change spatial adapters

        # [hardcoded] adapters using 1x1 conv
        # this is to obtain fixed size spatial embeddings from skills that output spatial embeddings
        # torch.Size([x, x, 16, 16]) (env, stacked frames, height, width)
        self.__vobj_seg_adapter = nn.Sequential(
            nn.Conv2d(20, 16, 1),
            nn.Conv2d(16, 16, 5, 5),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.__kpt_enc_adapter = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 6),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.__kpt_key_adapter = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.Conv2d(16, 16, 6),
            nn.ReLU(),
            # nn.Sigmoid()
        )
        self.adapters = {
            "obj_key_enc": self.__kpt_enc_adapter,
            "obj_key_key": self.__kpt_key_adapter,
            "vid_obj_seg": self.__vobj_seg_adapter,
        }
        self.__vobj_seg_adapter.to(device)
        self.__kpt_enc_adapter.to(device)
        self.__kpt_key_adapter.to(device)

        self.skills_embeddings = []

        # self.num_channels = 0
        # for el in self.skills_embeddings:
        #     if el.ndim == 4:
        #         self.num_channels += el.shape[1]

    def preprocess_input(
        self, observations: torch.Tensor, skill_indices: List[int] = None
    ):
        """
        :param observations: torch tensor of shape (n_envs, n_stacked_frames, height, width)
        :param skill_indices: list of skill indices to process (None = process all skills)
        """
        self.skills_embeddings = []

        # If skill_indices not provided, process all skills (for WSA compatibility)
        skills_to_process = (
            skill_indices if skill_indices is not None else range(len(self.skills))
        )

        for idx in skills_to_process:
            skill = self.skills[idx]
            # this apply a skill to the observations
            with torch.no_grad():
                so = skill.input_adapter(observations)
                so = skill.skill_output(
                    skill.skill_model, so
                )  # can return linear or spatial embeddings

            if skill.name in self.adapters:
                adapter = self.adapters[skill.name]
                so = adapter(so)

            # flatten skill out to linear embedding
            if len(so.shape) > 2:
                so = torch.reshape(so, (so.size(0), -1))

            self.skills_embeddings.append(so)

    def get_dimension(self, observations: torch.Tensor) -> int:
        out = self.forward(observations)
        return out.shape[1]


class WeightSharingAttentionExtractor(FeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        skills: List[Skill] | None = None,
        device="cpu",
    ):
        """
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations. This corresponds to the number of units for the last layer.
        :param skills: List of skill objects.
        :param device: Device used for computation.
        """
        super().__init__(observation_space, features_dim, skills, device)

        self.device = device

        sample = observation_space.sample()  # 4x84x84
        sample = np.expand_dims(sample, axis=0)  # 1x4x84x84
        sample = torch.from_numpy(sample) / 255
        sample = sample.to(device)

        # dropout_p = 0.1

        self.preprocess_input(sample)  # this will populate self.skills_embeddings

        # linear layers to learn a representation of the skills
        self.mlp_layers = nn.ModuleList()
        for i in range(len(self.skills_embeddings)):
            seq_layer = nn.Sequential(
                nn.Linear(
                    self.skills_embeddings[i].shape[1], features_dim, device=device
                ),
                nn.ReLU(),
                # nn.Sigmoid(),
                # nn.Dropout(p=dropout_p)
                # nn.BatchNorm1d(features_dim, device=device),
            )
            self.mlp_layers.append(seq_layer)

        # linear layer for context in the attention
        model_path = "skills/torch_models/nature-encoder-all-envs.pt"
        model = Autoencoder().to(device)
        model = torch.compile(model, mode="reduce-overhead")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()

        self.encoder = model.encoder

        z = get_embedding_for_context(sample, self.encoder)
        self.input_size = z.shape[-1]

        self.encoder_lin_layer = nn.Sequential(
            nn.Linear(self.input_size, features_dim, device=device),
            nn.ReLU(),
            # nn.Sigmoid(),
            # nn.Dropout(p=dropout_p)
            # nn.BatchNorm1d(features_dim, device=device),
        )

        # linear layers for attention weights
        self.weights = nn.Sequential(
            nn.Linear((2 * features_dim), 1, device=device), nn.ReLU()
        )

        # self.dropout = nn.Dropout(p=dropout_p)

        # ---------- saving info ---------- #

        self.att_weights = {}
        self.spatial_adapters = []
        self.linear_adapters = []
        self.training_weights = []

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # print("forward observation shape", observations.shape)
        # -------------- saving stats -------------- #
        
        
        weights: List[torch.Tensor] = []

        self.preprocess_input(observations)  # this will populate self.skills_embeddings

        encoded_frame = get_embedding_for_context(observations, self.encoder)
        encoded_frame = self.encoder_lin_layer(encoded_frame)  # query

        for i in range(len(self.skills_embeddings)):
            seq_layer = self.mlp_layers[i]

            self.skills_embeddings[i] = seq_layer(
                self.skills_embeddings[i]
            )  # pass through a mlp layer to reduce and fix the dimension

            concatenated = torch.cat([encoded_frame, self.skills_embeddings[i]], 1)

            weight: torch.Tensor = self.weights(concatenated)
            weights.append(weight)

        weights = torch.stack(weights, 1)
        weights = torch.softmax(weights, 1) # weights shape torch.Size([8, 4, 1])
        
        # Store on GPU - only transfer when needed for logging
        # Squeeze last dimension to match monitoring format: [batch, num_experts]
        self.training_weights.append(weights.squeeze(-1).detach())
        # weights = self.dropout(weights)

        # save attention weights to plot them in evaluation
        for i, s in enumerate(self.skills):
            self.att_weights[s.name] = [w[i] for w in weights]

        # now stack the skill outputs to obtain a sequence of tokens
        stacked_skills = torch.stack(self.skills_embeddings, 0).permute(1, 0, 2)

        # sum product of weights and skills
        att_out = weights * stacked_skills
        att_out = torch.sum(att_out, 1)
        return att_out


    
    
    
    
    
    
    
    
    
class SoftHardMOE(FeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        skills: List[Skill] | None = None,
        device="cpu",
        initial_temperature: float = 1.0,  # Start with soft routing
        min_temperature: float = 0.1,      # End with nearly-hard routing
        temperature_decay: float = 0.99995,  # Gradual annealing (tune this!)
        router_warmup_steps: int = 100,  # Steps before starting annealing
    ):
        """
        Mixture of Experts with soft-to-hard routing transition.
        
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations
        :param skills: List of skill objects (experts)
        :param device: Device used for computation
        :param initial_temperature: Starting temperature (1.0 = soft, uniform routing)
        :param min_temperature: Minimum temperature (0.1 = nearly hard routing)
        :param temperature_decay: Decay rate per forward pass (0.99995 recommended for 1M steps)
        
        Temperature annealing schedule:
        - At temp=1.0: Router weights are soft, all experts contribute
        - At temp=0.1: Router weights are sharp, dominant expert(s) take over
        - Decay of 0.99995 reaches min_temp after ~460k forward passes
        """

        super().__init__(observation_space, features_dim, skills, device)

        self.device = device
                
        # Temperature annealing parameters
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.router_warmup_steps = router_warmup_steps
        self.step_count = 0
        
        self.p_keep = 0.8  # Probability to keep each expert active during training
        # EMA for temporal smoothing
        self.ema_alpha = 0.95  # Smoothing factor (0.9 = heavy smoothing)
        self.register_buffer('ema_weights', None)  # Persistent across forward passes
        self.smoothing_factor = 0.8

        self.num_experts = len(skills)

        sample = observation_space.sample()  # 4x84x84
        sample = np.expand_dims(sample, axis=0)  # 1x4x84x84
        sample = torch.from_numpy(sample) / 255
        sample = sample.to(device)

        # Context encoder for routing decisions
        model_path = "skills/torch_models/nature-encoder-all-envs.pt"
        model = Autoencoder().to(device)
        model = torch.compile(model, mode="reduce-overhead")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        self.encoder = model.encoder
        
        
        z = get_embedding_for_context(sample, self.encoder)
        self.input_size = z.shape[-1]

        # Router network: takes context and outputs logits for each expert
        self.router = nn.Sequential(
            nn.Linear(self.input_size, features_dim, device=device),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(features_dim, len(self.skills), device=device),
        )
        
        # Initialize router to output near-uniform logits
        with torch.no_grad():
            self.router[-1].weight.data *= 0.1  # Small random weights
            self.router[-1].bias.data.zero_()    # Zero bias = uniform start

        # MLP layers for each skill to project to features_dim
        self.preprocess_input(sample)  # populate self.skills_embeddings
        
        self.mlp_layers = nn.ModuleList()
        for i in range(len(self.skills_embeddings)):
            seq_layer = nn.Sequential(
                nn.Linear(
                    self.skills_embeddings[i].shape[1], features_dim, device=device
                ),
                nn.LayerNorm(features_dim),
                nn.ReLU(),
            )

            self.mlp_layers.append(seq_layer)

        # Tracking
        self.training_weights = []
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(self.device)
        batch_size = observations.shape[0]

        # Get routing logits from context
        context = get_embedding_for_context(observations, self.encoder)
        router_logits = self.router(context)  # (batch_size, num_experts)
        
        # if self.step_count < self.router_warmup_steps:
        #     # Uniform routing - all skills contribute equally
        #     router_logits = router_logits.detach()
        #     router_weights = torch.ones_like(router_logits) / self.num_experts
            
        #     # here self.routing_entropy is maximum (uniform distribution)
        #     self.routing_entropy = torch.log(torch.tensor(float(self.num_experts), device=self.device))
            
        # else:
        router_weights = F.gumbel_softmax(router_logits, tau=self.temperature, hard=False)
        
        # Calculate entropy of routing distribution (per sample in batch)
        entropy = -torch.sum(router_weights * torch.log(router_weights + 1e-10), dim=1)
        
        # Store for auxiliary loss
        self.routing_entropy = entropy.mean()
        
        # Apply EMA smoothing (prevents drastic changes)
        if self.ema_weights is None:
            # First step after warmup: initialize with current weights
            self.ema_weights = router_weights.detach().mean(dim=0)  # Average over batch
        else:
            # Smooth: new_weights = alpha * old + (1-alpha) * new
            batch_mean_weights = router_weights.detach().mean(dim=0)
            self.ema_weights = self.ema_alpha * self.ema_weights + (1 - self.ema_alpha) * batch_mean_weights
    
        
        router_weights = self.smoothing_factor * self.ema_weights.unsqueeze(0) + (1 - self.smoothing_factor) * router_weights
        
        if self.training:
            # Apply dropout to router logits during training for exploration
            drop_mask = torch.bernoulli(
                torch.full_like(router_logits, self.p_keep)
            )
            router_logits = router_logits.masked_fill(drop_mask == 0, -1e9)    
            
        self.temperature = max(
            self.min_temperature,
            self.temperature * self.temperature_decay
        )
        
        self.step_count += 1

        # Process ALL experts (soft routing - no information loss)
        self.preprocess_input(observations)  # Processes all skills

        # Weighted combination of expert outputs
        output = torch.zeros(batch_size, self.features_dim, device=self.device)
        
        for i in range(self.num_experts):
            # Get skill embedding (already computed for all experts)
            skill_embedding = self.skills_embeddings[i]  # (batch_size, embedding_dim)
            
            # Project through trainable MLP
            skill_embedding = self.mlp_layers[i](skill_embedding)  # (batch_size, features_dim)
            
            # Weight by router decision
            expert_weight = router_weights[:, i].unsqueeze(1)  # (batch_size, 1)
            weighted_output = skill_embedding * expert_weight  # (batch_size, features_dim)
            
            output += weighted_output

        # Store weights for visualization
        self.training_weights.append(router_weights.detach())
    
        return output
    
    def get_auxiliary_loss(self) -> torch.Tensor:
        """Entropy and load balancing loss for the router.
        
        Returns a non-negative penalty that:
        - Is HIGH when routing collapses (low entropy, imbalanced usage)
        - Is LOW when routing is diverse (high entropy, balanced usage)
        """
        
        if not hasattr(self, 'training_weights') or len(self.training_weights) == 0 or not hasattr(self, 'routing_entropy'):
            return torch.tensor(0.0, device=self.device)

        # 1. Entropy regularization: penalize distance from maximum entropy
        # Maximum entropy for uniform distribution over N experts: log(N)
        max_entropy = torch.log(torch.tensor(float(self.num_experts), device=self.device))
        
        # Penalty is 0 when entropy is maximum (diverse), increases as entropy drops (collapsed)
        entropy_penalty = max_entropy - self.routing_entropy
        entropy_coefficient = 0.01  # Tune this: higher = stronger diversity pressure
        entropy_loss = entropy_coefficient * entropy_penalty
        
        # # 2. Load balancing: penalize deviation from uniform expert usage
        # recent_weights = torch.cat(self.training_weights[-100:], dim=0)  # Shape: [batch*N, num_experts]
        # avg_expert_usage = recent_weights.mean(dim=0)  # Shape: [num_experts]
        
        # # Target: each expert used equally (1/num_experts)
        # target_usage = 1.0 / self.num_experts
        # load_balance_loss = torch.sum((avg_expert_usage - target_usage) ** 2)
        # load_balance_coefficient = 0.1  # Tune this: higher = stronger balancing pressure
        
        # total_aux_loss = entropy_loss + load_balance_coefficient * load_balance_loss
        
        return entropy_loss 