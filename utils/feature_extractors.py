import torch
import torch.nn as nn
from typing import List
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from skills.autoencoder import Autoencoder
import torch.nn.functional as F
from skills.skill_interface import Skill
import numpy as np

def get_embedding_for_context(
        observations: torch.Tensor,
        encoder,
        detach: bool = True
    ) -> torch.Tensor:
        """Extract context for routing decisions"""

        if detach:
            with torch.no_grad():
                z = encoder(observations)
        else:
            z = encoder(observations)

        z = torch.reshape(z, (z.size(0), -1))

        return z


class FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        skills: List[Skill] | None = None,
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

        self.skills_embeddings: List[torch.Tensor] = []

        # self.num_channels = 0
        # for el in self.skills_embeddings:
        #     if el.ndim == 4:
        #         self.num_channels += el.shape[1]

    def preprocess_input(
        self, 
        observations: torch.Tensor, 
        skill_indices: List[int] | None = None
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

        self.preprocess_input(sample)  # this will populate self.skills_embeddings

        # linear layers to learn a representation of the skills
        self.mlp_layers = nn.ModuleList()
        for i in range(len(self.skills_embeddings)):
            seq_layer = nn.Sequential(
                nn.LayerNorm(self.skills_embeddings[i].shape[1], device=device),
                nn.Linear(
                    self.skills_embeddings[i].shape[1], features_dim, device=device
                ),
                nn.ReLU(),                
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
            nn.LayerNorm(self.input_size, device=device),
            nn.Linear(self.input_size, features_dim, device=device),
            nn.ReLU(),
        )
        self.weights = nn.Linear((2 * features_dim), 1, device=device)
        
        self.final_layer_norm = nn.LayerNorm(features_dim, device=device)

        # ---------- saving info ---------- #

        self.att_weights = {}
        self.spatial_adapters = []
        self.linear_adapters = []
        self.training_weights = []  # For monitoring only (detached)
        self.current_batch_weights = None  # For auxiliary loss gradients (not detached)
        
        # Track number of experts (skills) for auxiliary loss
        self.num_experts = len(self.skills)
        
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
            
            self.skills_embeddings[i] = F.normalize(self.skills_embeddings[i], dim=1)  # L2 normalize skill embeddings
            
            concatenated = torch.cat([encoded_frame, self.skills_embeddings[i]], 1)

            weight: torch.Tensor = self.weights(concatenated)
            weights.append(weight)

        weights = torch.stack(weights, 1) # weights shape torch.Size([8, 4, 1])
        
        # Before softmax, add gaussian noise for exploration (only during training)
        if self.training:
            noise = torch.randn_like(weights) * 0.1
            weights = weights + noise
        
        weights = torch.softmax(weights, 1) 
        
        # Store weights for load balancing loss (NOT detached for gradients)
        weights_2d = weights.squeeze(-1)  # [batch, num_experts]
        self.current_batch_weights = weights_2d  # Keep gradients for auxiliary loss
        
        # Store detached version for monitoring/logging only
        self.training_weights.append(weights_2d.mean(dim=0).detach())

        # save attention weights to plot them in evaluation
        for i, s in enumerate(self.skills):
            self.att_weights[s.name] = [w[i] for w in weights]

        # now stack the skill outputs to obtain a sequence of tokens
        stacked_skills = torch.stack(self.skills_embeddings, 0).permute(1, 0, 2)

        # sum product of weights and skills
        att_out = weights * stacked_skills
        att_out = torch.sum(att_out, 1)
        final_out = self.final_layer_norm(att_out)
        return final_out

class MixtureOfExpertsExtractor(FeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        skills: List[Skill] | None = None,
        device="cpu",
    ):
        super().__init__(observation_space, features_dim, skills, device)

        self.device = device
        self._moe_features_dim = features_dim
        self.top_k = 3

        sample = observation_space.sample()
        sample = np.expand_dims(sample, axis=0)
        sample = torch.from_numpy(sample) / 255
        sample = sample.to(device)

        # Trainable context encoder with the same architecture as the pretrained autoencoder encoder.
        self.encoder = Autoencoder().encoder.to(device)

        self.preprocess_input(sample)

        self.mlp_layers = nn.ModuleList()
        for i in range(len(self.skills_embeddings)):
            seq_layer = nn.Sequential(
                nn.LayerNorm(self.skills_embeddings[i].shape[1], device=device),
                nn.Linear(
                    self.skills_embeddings[i].shape[1], features_dim, device=device
                ),
                nn.ReLU(),
            )
            self.mlp_layers.append(seq_layer)

        z = get_embedding_for_context(sample, self.encoder, detach=False)
        self.input_size = z.shape[-1]

        self.encoder_lin_layer = nn.Sequential(
            nn.LayerNorm(self.input_size, device=device),
            nn.Linear(self.input_size, features_dim, device=device),
            nn.ReLU(),
        )

        self.router = nn.Sequential(
            nn.LayerNorm(features_dim, device=device),
            nn.Linear(features_dim, len(self.skills), device=device),
        )

        self.final_layer_norm = nn.LayerNorm(features_dim, device=device)

        self.att_weights = {}
        self.training_weights = []
        self.current_batch_weights = None
        self.num_experts = len(self.skills)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        encoded_frame = get_embedding_for_context(
            observations,
            self.encoder,
            detach=False,
        )
        encoded_frame = self.encoder_lin_layer(encoded_frame)

        router_logits = self.router(encoded_frame)

        if self.training:
            noise = torch.randn_like(router_logits) * 0.1
            router_logits = router_logits + noise

        top_k = min(self.top_k, router_logits.shape[1])
        topk_values, topk_indices = torch.topk(router_logits, top_k, dim=1)
        topk_weights = torch.softmax(topk_values, dim=1)

        weights = torch.zeros_like(router_logits)
        weights.scatter_(1, topk_indices, topk_weights)

        self.current_batch_weights = weights
        self.training_weights.append(weights.mean(dim=0).detach())

        for i, s in enumerate(self.skills):
            self.att_weights[s.name] = [w[i] for w in weights]

        selected_experts = torch.unique(topk_indices).detach().cpu().tolist()
        self.preprocess_input(observations, skill_indices=selected_experts)

        expert_outputs = torch.zeros(
            observations.shape[0],
            self.num_experts,
            self._moe_features_dim,
            device=observations.device,
            dtype=encoded_frame.dtype,
        )

        for skill_position, expert_idx in enumerate(selected_experts):
            seq_layer = self.mlp_layers[expert_idx]
            expert_embedding = seq_layer(self.skills_embeddings[skill_position])
            expert_embedding = F.normalize(expert_embedding, dim=1)
            expert_outputs[:, expert_idx, :] = expert_embedding

        att_out = weights.unsqueeze(-1) * expert_outputs
        att_out = torch.sum(att_out, dim=1)
        final_out = self.final_layer_norm(att_out)
        return final_out
    