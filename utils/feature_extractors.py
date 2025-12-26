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

        #TODO: change spatial adapters

        # [hardcoded] adapters using 1x1 conv
        # this is to obtain fixed size spatial embeddings from skills that output spatial embeddings
        # torch.Size([x, x, 16, 16]) (env, stacked frames, height, width) 
        self.__vobj_seg_adapter = nn.Sequential(
            nn.Conv2d(20, 16, 1),
            nn.Conv2d(16, 16, 5, 5),
            nn.ReLU(),
            #nn.Sigmoid()
        )
        self.__kpt_enc_adapter = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 6),
            nn.ReLU(),
            #nn.Sigmoid()
        )
        self.__kpt_key_adapter = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.Conv2d(16, 16, 6),
            nn.ReLU()
            #nn.Sigmoid()
        )
        self.adapters = {
            "obj_key_enc": self.__kpt_enc_adapter,
            "obj_key_key": self.__kpt_key_adapter,
            "vid_obj_seg": self.__vobj_seg_adapter
        }
        self.__vobj_seg_adapter.to(device)
        self.__kpt_enc_adapter.to(device)
        self.__kpt_key_adapter.to(device)

        self.skills_embeddings = []

        # self.num_channels = 0
        # for el in self.skills_embeddings:
        #     if el.ndim == 4:
        #         self.num_channels += el.shape[1]

    def preprocess_input(self, observations: torch.Tensor, skill_indices: List[int] = None):
        """
        :param observations: torch tensor of shape (n_envs, n_stacked_frames, height, width)
        :param skill_indices: list of skill indices to process (None = process all skills)
        """
        self.skills_embeddings = []

        # If skill_indices not provided, process all skills (for WSA compatibility)
        skills_to_process = skill_indices if skill_indices is not None else range(len(self.skills))

        for idx in skills_to_process:
            skill = self.skills[idx]
            # this apply a skill to the observations
            with torch.no_grad():
                so = skill.input_adapter(observations)
                so = skill.skill_output(skill.skill_model, so) # can return linear or spatial embeddings
                       
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
        device="cpu"
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

        #dropout_p = 0.1

        self.preprocess_input(sample) # this will populate self.skills_embeddings
            
        # linear layers to learn a representation of the skills
        self.mlp_layers = nn.ModuleList()
        for i in range(len(self.skills_embeddings)):
            seq_layer = nn.Sequential(
                nn.Linear(self.skills_embeddings[i].shape[1], features_dim, device=device),
                nn.ReLU(),
                #nn.Sigmoid(),
                #nn.Dropout(p=dropout_p)
                #nn.BatchNorm1d(features_dim, device=device),
            )
            self.mlp_layers.append(seq_layer)

        # linear layer for context in the attention
        model_path = "skills/torch_models/nature-encoder-all-envs.pt"
        model = Autoencoder().to(device)
        model = torch.compile(model, mode='default')
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()

        self.encoder = model.encoder
                
        z = self.get_last_frame_embedding_for_context(sample)    
        self.input_size = z.shape[-1]

        self.encoder_lin_layer = nn.Sequential(
            nn.Linear(self.input_size, features_dim, device=device),
            nn.ReLU(),
            #nn.Sigmoid(),
            #nn.Dropout(p=dropout_p)
            #nn.BatchNorm1d(features_dim, device=device),
        )

        # linear layers for attention weights
        self.weights = nn.Sequential(
            nn.Linear((2 * features_dim), 1, device=device),
            nn.ReLU()
        )
        
        #self.dropout = nn.Dropout(p=dropout_p)

        # ---------- saving info ---------- #

        self.att_weights = {}
        self.spatial_adapters = []
        self.linear_adapters = []
        self.training_weights = []

    def get_last_frame_embedding_for_context(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations[:, -1:, :, :]
        with torch.no_grad():
            z = self.encoder(x)
            z = torch.reshape(z, (z.size(0), -1))
        
        return z

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        #print("forward observation shape", observations.shape)
        # -------------- saving stats -------------- #
        weights = []

        self.preprocess_input(observations) # this will populate self.skills_embeddings

        encoded_frame = self.get_last_frame_embedding_for_context(observations)
        encoded_frame = self.encoder_lin_layer(encoded_frame)  # query

        for i in range(len(self.skills_embeddings)):
            seq_layer = self.mlp_layers[i]
            
            self.skills_embeddings[i] = seq_layer(self.skills_embeddings[i])  # pass through a mlp layer to reduce and fix the dimension
            
            concatenated = torch.cat([encoded_frame, self.skills_embeddings[i]], 1)

            weight = self.weights(concatenated)
            weights.append(weight)

        weights = torch.stack(weights, 1)
        weights = torch.softmax(weights, 1)

        self.training_weights.append(weights.detach().cpu().numpy())
        #weights = self.dropout(weights)

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
        use_soft_routing: bool = True,  # use soft routing for better gradients
        router_noise: float = 0.1,  # noise std for exploration during training
    ):
        """
        Mixture of Experts feature extractor that selectively activates only top-k skills based on router decision.
        
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations
        :param skills: List of skill objects (experts)
        :param device: Device used for computation
        :param top_k: Number of top experts to activate (default: 2)
        :param use_soft_routing: If True, use soft top-k for gradient flow (default: True)
        :param router_noise: Standard deviation of noise added to router logits during training
        """
        super().__init__(observation_space, features_dim, skills, device)

        self.device = device
        self.top_k = min(top_k, len(skills)) if skills else top_k
        self.use_soft_routing = use_soft_routing
        self.router_noise = router_noise
        
        sample = observation_space.sample()  # 4x84x84
        sample = np.expand_dims(sample, axis=0)  # 1x4x84x84
        sample = torch.from_numpy(sample) / 255
        sample = sample.to(device)

        # Context encoder for routing decisions
        model_path = "skills/torch_models/nature-encoder-all-envs.pt"
        model = Autoencoder().to(device)
        model = torch.compile(model, mode='default')
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        self.encoder = model.encoder
        
        z = self.get_last_frame_embedding_for_context(sample)    
        self.input_size = z.shape[-1]

        # Router network: takes context and outputs logits for each expert (Stage 1: Selection)
        self.router = nn.Sequential(
            nn.Linear(self.input_size, features_dim, device=device),
            nn.ReLU(),
            nn.Linear(features_dim, len(self.skills), device=device)
        )

        # Context encoder for weight refinement (Stage 2)
        self.encoder_lin_layer = nn.Sequential(
            nn.Linear(self.input_size, features_dim, device=device),
            nn.ReLU(),
        )

        # MLP layers for each skill to project to features_dim
        self.preprocess_input(sample)  # populate self.skills_embeddings
        
        self.mlp_layers = nn.ModuleList()
        for i in range(len(self.skills_embeddings)):
            seq_layer = nn.Sequential(
                nn.Linear(self.skills_embeddings[i].shape[1], features_dim, device=device),
                nn.ReLU(),
            )
            self.mlp_layers.append(seq_layer)

        # Weight refinement MLP (Stage 2: Refinement)
        # Takes concatenated [context, skill_embedding] and outputs weight for that skill
        self.weight_refiner = nn.Sequential(
            nn.Linear(2 * features_dim, 1, device=device),
            nn.ReLU()
        )

        # Tracking
        self.expert_weights = {}
        self.selected_experts = []
        self.training_weights = []

    def get_last_frame_embedding_for_context(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract context from the last frame for routing decisions"""
        x = observations[:, -1:, :, :]
        with torch.no_grad():
            z = self.encoder(x)
            z = torch.reshape(z, (z.size(0), -1))
        return z

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        
        # Get context for routing and weight refinement
        context_raw = self.get_last_frame_embedding_for_context(observations)
        
        # ========== STAGE 1: Expert Selection (Router) ==========
        # Router outputs logits for each expert
        router_logits = self.router(context_raw)  # (batch_size, num_experts)
        
        # Add noise during training for exploration
        if self.training and self.router_noise > 0:
            noise = torch.randn_like(router_logits) * self.router_noise
            router_logits = router_logits + noise
        
        # Select top-k experts per sample
        # Shape: (batch_size, top_k)
        _, top_k_indices = torch.topk(router_logits, self.top_k, dim=1)
        
        # Find unique experts selected across the entire batch for efficient inference
        unique_experts = torch.unique(top_k_indices).tolist()
        
        # ========== Process Selected Experts (Batch-wise) ==========
        # Process all unique experts for the ENTIRE batch at once using preprocess_input
        self.preprocess_input(observations, skill_indices=unique_experts)
        
        # Project skill embeddings through their respective MLPs
        # Create mapping from expert_idx to position in skills_embeddings
        expert_to_embedding = {}
        for i, expert_idx in enumerate(unique_experts):
            skill_embedding = self.skills_embeddings[i]  # (batch_size, embedding_dim)
            skill_embedding = self.mlp_layers[expert_idx](skill_embedding)  # (batch_size, features_dim)
            expert_to_embedding[expert_idx] = skill_embedding
        
        # ========== STAGE 2: Weight Refinement ==========
        # Process context through linear layer for refinement
        context_encoded = self.encoder_lin_layer(context_raw)  # (batch_size, features_dim)
        
        # Build a list to collect (batch_idx, expert_idx, weight_logit) for selected pairs only
        refined_logits = torch.full((batch_size, len(self.skills)), float('-inf'), device=self.device)
        
        # For each sample, compute refined weights only for its selected experts
        for batch_idx in range(batch_size):
            sample_experts = top_k_indices[batch_idx]  # (top_k,) - experts selected by this sample
            sample_logits = []
            
            for expert_idx in sample_experts:
                expert_idx_val = expert_idx.item()
                # Only compute weight for this specific (sample, expert) pair
                concatenated = torch.cat([
                    context_encoded[batch_idx:batch_idx+1], 
                    expert_to_embedding[expert_idx_val][batch_idx:batch_idx+1]
                ], dim=1)  # (1, 2*features_dim)
                weight_logit = self.weight_refiner(concatenated)  # (1, 1)
                sample_logits.append(weight_logit.squeeze())
            
            # Softmax over this sample's selected experts only
            sample_weights = torch.softmax(torch.stack(sample_logits), dim=0)
            
            # Assign weights to the selected experts for this sample
            for i, expert_idx in enumerate(sample_experts):
                refined_logits[batch_idx, expert_idx] = torch.log(sample_weights[i] + 1e-10)
        
        # Apply softmax (log-softmax-exp pattern for numerical stability)
        refined_weights = torch.softmax(refined_logits, dim=1)  # (batch_size, num_experts)
        
        # ========== Combine Expert Outputs ==========
        output = torch.zeros(batch_size, self.features_dim, device=self.device)
        
        for expert_idx in unique_experts:
            skill_embedding = expert_to_embedding[expert_idx]  # (batch_size, features_dim)
            expert_weights = refined_weights[:, expert_idx]  # (batch_size,)
            
            # Weight and accumulate
            weighted_output = skill_embedding * expert_weights.unsqueeze(1)  # (batch_size, features_dim)
            output += weighted_output
        
        # Store weights for visualization
        self.training_weights.append(refined_weights.detach().cpu().numpy())
        for i, skill in enumerate(self.skills):
            self.expert_weights[skill.name] = refined_weights[:, i].detach().cpu().tolist()
        
        return output