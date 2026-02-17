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
        encoder
    ) -> torch.Tensor:
        """Extract context for routing decisions"""

        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            z = encoder(observations)
            z = torch.reshape(z, (z.size(0), -1))
        
        # Convert back to float32 for downstream processing
        return z.float()


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
        
        # Store device and create CUDA streams for parallel processing
        self.device = device
        self.use_cuda_streams = False #device != "cpu" and torch.cuda.is_available()
        if self.use_cuda_streams and skills is not None:
            print("Using CUDA streams for parallel skill processing")
            # Create one stream per skill for parallel execution
            self.cuda_streams = [torch.cuda.Stream() for _ in range(len(skills))]
        else:
            print("CUDA streams not available, using sequential processing")
            self.cuda_streams = None

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
        # If skill_indices not provided, process all skills (for WSA compatibility)
        skills_to_process = (
            list(skill_indices) if skill_indices is not None else list(range(len(self.skills)))
        )
        
        # Parallel processing with CUDA streams (when available)
        if self.use_cuda_streams:
            # Create temporary storage for results (indexed by position in skills_to_process)
            temp_embeddings = [None] * len(skills_to_process)
            
            # Launch all skill computations in parallel on separate streams
            for i, idx in enumerate(skills_to_process):
                skill = self.skills[idx]
                stream = self.cuda_streams[idx]
                
                with torch.cuda.stream(stream):
                    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
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
                    
                    # Convert back to float32 for downstream processing
                    so = so.float()
                    temp_embeddings[i] = so
            
            # Synchronize all streams to ensure all computations are complete
            for idx in skills_to_process:
                self.cuda_streams[idx].synchronize()
            
            # Store results in order
            self.skills_embeddings = temp_embeddings
            
        else:
            # Sequential processing (CPU or when CUDA streams not available)
            self.skills_embeddings = []
            
            for idx in skills_to_process:
                skill = self.skills[idx]
                # this apply a skill to the observations
                with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
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
                
                # Convert back to float32 for downstream processing
                so = so.float()
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
        :param entropy_coef: Coefficient for entropy regularization in auxiliary loss.
        :param load_balance_coef: Coefficient for load balancing in auxiliary loss.
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
        #weights = torch.softmax(weights / self.temperature, 1) # weights shape torch.Size([8, 4, 1])
        
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
    
    # def get_auxiliary_loss(self) -> torch.Tensor:
    #     """Auxiliary loss to encourage skill diversity over long time horizons.
        
    #     Uses a two-component approach:
    #     1. Long-term load balancing: Tracks skill usage via EMA across many batches,
    #        encouraging uniform usage over episodes/training (allows per-timestep specialization)
    #     2. Current batch guidance: Gently nudges current batch toward the long-term target
    #        to prevent sudden collapse while allowing dynamic, context-dependent routing
        
    #     This enables:
    #     - Per-timestep specialization: t=[0.10, 0.60, 0.05, 0.25], t+1=[0.70, 0.10, 0.10, 0.10]
    #     - Per-batch adaptation: Batch can be biased toward relevant skills for current game states
    #     - Long-term balance: Over thousands of steps, all skills used roughly equally
    #     """
        
    #     if self.current_batch_weights is None:
    #         return torch.tensor(0.0, device=self.device, requires_grad=True)
        
    #     # Shape: [batch_size, num_experts]
    #     batch_weights = self.current_batch_weights
        
    #     # Average usage in current batch (with gradients)
    #     avg_batch_usage = batch_weights.mean(dim=0)  # [num_experts]
        
    #     # Target: uniform distribution (1/num_experts for each expert)
    #     uniform_target = 1.0 / self.num_experts
        

    #     # Gradient flows through avg_batch_usage -> current_batch_weights
    #     load_balance_loss = torch.mean((avg_batch_usage - uniform_target) ** 2)
        
    #     return self.load_balance_coef * load_balance_loss


    
    
class SoftHardMOE(FeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        skills: List[Skill] | None = None,
        device="cpu",
    ):
        """
        Mixture of Experts with soft-to-hard routing transition.
        
        :param observation_space: Gymnasium observation space
        :param features_dim: Number of features extracted from the observations
        :param skills: List of skill objects (experts)
        :param device: Device used for computation
        """

        super().__init__(observation_space, features_dim, skills, device)

        self.device = device
        
        # temperature annealing parameters
        self.temperature = 1.0  # Start with soft routing
        self.min_temperature = 0.1      # End with nearly-hard routing
        self.temperature_decay = 0.99998  # Gradual annealing (tune this!)
        self.step_count = 0
               
        # expert dropout parameters
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
        
        # Compute routing weights with Gumbel-Softmax
        router_weights = F.gumbel_softmax(router_logits, tau=self.temperature, hard=False)
        
        # Calculate entropy of routing distribution (per sample in batch)
        entropy = -torch.sum(router_weights * torch.log(router_weights + 1e-10), dim=1)
        
        # Store for auxiliary loss
        self.routing_entropy = entropy.mean()
        
        #Apply EMA smoothing (prevents drastic changes)
        if self.ema_weights is None:
            # First step after warmup: initialize with current weights
            self.ema_weights = router_weights.detach().mean(dim=0)  # Average over batch
        else:
            # Smooth: new_weights = alpha * old + (1-alpha) * new
            batch_mean_weights = router_weights.detach().mean(dim=0)
            self.ema_weights = self.ema_alpha * self.ema_weights + (1 - self.ema_alpha) * batch_mean_weights
    
        router_weights = self.smoothing_factor * self.ema_weights.unsqueeze(0) + (1 - self.smoothing_factor) * router_weights
        
        # Apply expert dropout AFTER EMA smoothing (only during training)
        # This ensures dropped experts are truly zeroed out
        if self.training:
            drop_mask = torch.bernoulli(
                torch.full((batch_size, self.num_experts), self.p_keep, device=self.device)
            )
            router_weights = router_weights * drop_mask
            # Re-normalize so weights still sum to 1
            router_weights = router_weights / (router_weights.sum(dim=1, keepdim=True) + 1e-10)
            
        # Anneal temperature for soft-to-hard transition
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
        entropy_coefficient = 0.01
        entropy_loss = entropy_coefficient * entropy_penalty
        
        return entropy_loss 
    
    # def get_auxiliary_loss(self) -> torch.Tensor:
    #     """Load balancing loss to prevent expert collapse"""
    #     if not hasattr(self, 'training_weights') or len(self.training_weights) == 0:
    #         return torch.tensor(0.0, device=self.device)
        
    #     # Get recent routing decisions (last N batches)
    #     recent_weights = torch.cat(self.training_weights[-500:], dim=0)  # Shape: [batch*10, num_experts]
        
    #     # Calculate average usage per expert
    #     avg_expert_usage = recent_weights.mean(dim=0)  # Shape: [num_experts]
        
    #     # Target: each expert used equally (1/num_experts)
    #     target_usage = 1.0 / self.num_experts
        
    #     # Penalize deviation from uniform usage
    #     load_balance_loss = torch.sum((avg_expert_usage - target_usage) ** 2)
        
    #     return 0.01 * load_balance_loss  # Tune coefficient