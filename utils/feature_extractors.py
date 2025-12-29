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


class MixtureOfExpertsExtractor(FeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        skills: List[Skill] | None = None,
        device="cpu",
        top_k: int = 2,  # number of experts to activate
        exploration_noise_std: float = 0.1,  # noise level for exploration (0.01-0.1 recommended)
        load_balance_coef: float = 0.01,  # coefficient for load balancing loss
    ):
        """

        Mixture of Experts feature extractor that selectively activates only top-k skills based on router decision.
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations
        :param skills: List of skill objects (experts)
        :param device: Device used for computation
        :param top_k: Number of top experts to activate (default: 2)
        :param exploration_noise_std: Standard deviation of Gaussian noise added to router logits during training (default: 0.1)
        :param load_balance_coef: Coefficient for load balancing auxiliary loss to encourage expert diversity (default: 0.01)
        """

        super().__init__(observation_space, features_dim, skills, device)

        self.device = device

        self.top_k = min(top_k, len(skills)) if skills else top_k
        self.exploration_noise_std = exploration_noise_std
        self.load_balance_coef = load_balance_coef
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
            nn.Linear(features_dim, len(self.skills), device=device),
        )

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
        self.expert_weights = {}
        self.selected_experts = []
        self.training_weights = []
        self.load_balance_loss = torch.tensor(0.0, device=device)  # stores last computed load balance loss
         


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        observations = observations.to(self.device) 
         
        batch_size = observations.shape[0]

        context = get_embedding_for_context(observations, self.encoder)
        router_logits = self.router(context)  # (batch_size, num_experts)
        
        # Store router probabilities for load balancing loss (before noise)
        router_probs = torch.softmax(router_logits, dim=1)  # (batch_size, num_experts)
        
        if self.training and self.exploration_noise_std > 0:
            # Add Gaussian noise to logits for exploration
            # Reduced from 1.0 to 0.1 default - high noise destabilizes learning
            noise = torch.randn_like(router_logits) * self.exploration_noise_std
            router_logits = router_logits + noise

        top_k_values, top_k_indices = torch.topk(router_logits, self.top_k, dim=1)

        # Normalize weights using softmax only over selected experts
        top_k_weights = torch.softmax(top_k_values, dim=1)  # (batch_size, top_k)

        # Store weights for tracking (sparse representation)
        all_weights = torch.zeros(batch_size, len(self.skills), device=self.device)
        all_weights.scatter_(1, top_k_indices, top_k_weights)
        
        # Find unique experts selected across the entire batch
        unique_experts = torch.unique(top_k_indices).tolist()

        # Process all unique experts for the ENTIRE batch at once using preprocess_input
        self.preprocess_input(observations, skill_indices=unique_experts)

        # Now combine expert outputs using per-sample weights
        output = torch.zeros(batch_size, self.features_dim, device=self.device)


        for i, expert_idx in enumerate(unique_experts):
            # Get the skill embedding for all samples (already computed)
            skill_embedding = self.skills_embeddings[i]  # (batch_size, embedding_dim)

            # Project through MLP (trainable)
            skill_embedding = self.mlp_layers[expert_idx](skill_embedding)  # (batch_size, features_dim)

            # Get per-sample weights for this expert
            expert_weights = all_weights[:, expert_idx]  # (batch_size,)

            # Weight and accumulate
            weighted_output = skill_embedding * expert_weights.unsqueeze(1)  # (batch_size, features_dim)

            output += weighted_output

        # Compute load balancing loss to encourage expert diversity
        if self.training:
            self.load_balance_loss = self._compute_load_balance_loss(router_probs, top_k_indices)
        
        # Store weights for visualization
        self.training_weights.append(all_weights.detach())
        for i, skill in enumerate(self.skills):
            self.expert_weights[skill.name] = all_weights[:, i].detach().cpu().tolist()

        return output
    
    def _compute_load_balance_loss(
        self, 
        router_probs: torch.Tensor, 
        top_k_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute load balancing auxiliary loss to encourage uniform expert utilization.
        
        Based on "Switch Transformers" paper (Fedus et al., 2021):
        loss = num_experts * sum_i(f_i * P_i)
        
        where:
        - f_i = fraction of samples routed to expert i (how often expert i is selected)
        - P_i = mean routing probability for expert i (router's confidence for expert i)
        
        This loss is minimized when experts are used equally often AND 
        the router assigns similar probabilities to all experts.
        
        :param router_probs: Router softmax probabilities (batch_size, num_experts)
        :param top_k_indices: Indices of selected top-k experts (batch_size, top_k)
        :return: Load balancing loss scalar
        """
        batch_size = router_probs.shape[0]
        
        # Compute P_i: mean router probability for each expert across the batch
        mean_router_probs = router_probs.mean(dim=0)  # (num_experts,)
        
        # Compute f_i: fraction of samples that route to each expert
        # Create one-hot encoding of selected experts
        expert_mask = torch.zeros(
            batch_size, 
            self.num_experts, 
            device=self.device
        )
        
        # Mark which experts were selected (any of top-k)
        for k in range(self.top_k):
            expert_mask.scatter_(1, top_k_indices[:, k:k+1], 1.0)
        
        # Compute fraction: how many samples selected each expert
        expert_usage_fraction = expert_mask.sum(dim=0) / batch_size  # (num_experts,)
        
        # Load balance loss: encourages both uniform usage AND uniform probabilities
        # Multiply by num_experts to scale appropriately
        load_balance_loss = self.num_experts * torch.sum(
            expert_usage_fraction * mean_router_probs
        )
        
        # Scale by coefficient (typically 0.01)
        return self.load_balance_coef * load_balance_loss
    
    def get_auxiliary_loss(self) -> torch.Tensor:
        """
        Get the auxiliary load balancing loss to be added to the main PPO loss.
        
        Usage in PPO training loop:
            # After computing PPO loss
            aux_loss = model.policy.features_extractor.get_auxiliary_loss()
            total_loss = ppo_loss + aux_loss
        
        :return: Load balancing loss tensor (scalar)
        """
        return self.load_balance_loss
    
    
    
    
    
    
    
    
    
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
        router_warmup_steps: int = 50000,  # Steps before starting annealing
        exploration_noise_std: float = 0.3, # noise level for exploration
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
        
        self.exploration_noise_std = exploration_noise_std
        
        # Temperature annealing parameters
        self.temperature = initial_temperature
        self.min_temperature = min_temperature
        self.temperature_decay = temperature_decay
        self.router_warmup_steps = router_warmup_steps
        self.step_count = 0
        
        self.p_keep = 0.7  # Probability to keep each expert active during training

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
        
        if self.step_count < self.router_warmup_steps:
            # Uniform routing - all skills contribute equally
            router_logits = router_logits.detach()
            router_weights = torch.ones_like(router_logits) / self.num_experts
        else:
            
                
            noise = torch.randn_like(router_logits) * self.exploration_noise_std
            router_logits = router_logits + noise

            scaled_logits = router_logits / self.temperature
            
            # Apply dropout to router logits during training for exploration
            drop_mask = torch.bernoulli(
                torch.full_like(scaled_logits, self.p_keep)
            )
            router_logits = router_logits.masked_fill(drop_mask == 0, -1e9)
    
            
            #router_weights = torch.softmax(scaled_logits, dim=1)  # (batch_size, num_experts)
            router_weights = F.gumbel_softmax(scaled_logits, tau=self.temperature, hard=False)
            

            # Anneal temperature during training
            if self.training:
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