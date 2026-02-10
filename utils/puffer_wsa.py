import torch
import torch.nn as nn
from typing import List
from skills.skill_interface import Skill

from skills.autoencoder import Autoencoder


class AttentionFeatureExtractor(nn.Module):
    """
    Attention-based feature extractor combining multiple pretrained skills.
    This replaces the SB3 BaseFeaturesExtractor.
    """
    def __init__(
        self,
        skills: List[Skill],
        features_dim: int = 256,
        device="cpu",
    ):
        super().__init__()
        
        self.skills = skills
        self.device = device
        self.features_dim = features_dim
        
        # Spatial adapters (same as your original code)
        self.__vobj_seg_adapter = nn.Sequential(
            nn.Conv2d(20, 16, 1),
            nn.Conv2d(16, 16, 5, 5),
            nn.ReLU(),
        )
        self.__kpt_enc_adapter = nn.Sequential(
            nn.Conv2d(128, 32, 1),
            nn.Conv2d(32, 32, 6),
            nn.ReLU(),
        )
        self.__kpt_key_adapter = nn.Sequential(
            nn.Conv2d(4, 16, 1),
            nn.Conv2d(16, 16, 6),
            nn.ReLU(),
        )
        
        self.adapters = {
            "obj_key_enc": self.__kpt_enc_adapter,
            "obj_key_key": self.__kpt_key_adapter,
            "vid_obj_seg": self.__vobj_seg_adapter,
        }
        
        self.__vobj_seg_adapter.to(device)
        self.__kpt_enc_adapter.to(device)
        self.__kpt_key_adapter.to(device)
        
        # Compile adapters for speed
        self.__vobj_seg_adapter = torch.compile(self.__vobj_seg_adapter, mode="default")
        self.__kpt_enc_adapter = torch.compile(self.__kpt_enc_adapter, mode="default")
        self.__kpt_key_adapter = torch.compile(self.__kpt_key_adapter, mode="default")
        
        # Initialize with dummy input to determine skill output sizes
        self._initialized = False
        
        # Encoder for context
        model_path = "skills/torch_models/nature-encoder-all-envs.pt"
        model = Autoencoder().to(device)
        model = torch.compile(model, mode="reduce-overhead")
        model.load_state_dict(torch.load(model_path, map_location=device), strict=True)
        model.eval()
        self.encoder = model.encoder
        
        # These will be initialized lazily
        self.mlp_layers = None
        self.encoder_lin_layer = None
        self.weights = None
        
    def _lazy_init(self, observations: torch.Tensor):
        """Initialize layers after seeing first observation"""
        if self._initialized:
            return
            
        # Process skills to get dimensions
        skills_embeddings = self._preprocess_input(observations)
        
        # Initialize MLP layers for each skill
        self.mlp_layers = nn.ModuleList()
        for i, emb in enumerate(skills_embeddings):
            seq_layer = nn.Sequential(
                nn.Linear(emb.shape[1], self.features_dim, device=self.device),
                nn.ReLU(),
            )
            self.mlp_layers.append(seq_layer)
        
        # Initialize encoder linear layer
        with torch.no_grad():
            z = self.encoder(observations)
            z = torch.reshape(z, (z.size(0), -1))
        
        self.encoder_lin_layer = nn.Sequential(
            nn.Linear(z.shape[-1], self.features_dim, device=self.device),
            nn.ReLU(),
        )
        
        # Initialize attention weights layer
        self.weights = nn.Sequential(
            nn.Linear((2 * self.features_dim), 1, device=self.device),
            nn.ReLU()
        )
        
        # Storage for attention weights (for monitoring)
        self.att_weights = {}
        self.training_weights = []
        
        self._initialized = True
    
    def _preprocess_input(self, observations: torch.Tensor) -> List[torch.Tensor]:
        """Process observations through all skills"""
        skills_embeddings = []
        
        for skill in self.skills:
            with torch.no_grad():
                so = skill.input_adapter(observations)
                so = skill.skill_output(skill.skill_model, so)
            
            if skill.name in self.adapters:
                adapter = self.adapters[skill.name]
                so = adapter(so)
            
            # Flatten to linear embedding
            if len(so.shape) > 2:
                so = torch.reshape(so, (so.size(0), -1))
            
            skills_embeddings.append(so)
        
        return skills_embeddings
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass combining skill embeddings with attention.
        
        Args:
            observations: Shape (batch, channels, height, width) or (batch, 1, channels, height, width)
            
        Returns:
            Combined feature vector of shape (batch, features_dim)
        """
        
        # Handle PufferLib's extra dimension: (batch, 1, 4, 84, 84) -> (batch, 4, 84, 84)
        if observations.ndim == 5 and observations.shape[1] == 1:
            observations = observations.squeeze(1)
        
        # Convert uint8 to float32 and normalize to [0, 1]
        if observations.dtype == torch.uint8:
            observations = observations.float() / 255.0
        
        # Lazy initialization on first forward pass
        if not self._initialized:
            self._lazy_init(observations)
        
        weights = []
        
        # Get skill embeddings
        skills_embeddings = self._preprocess_input(observations)
        
        # Get context from encoder
        with torch.no_grad():
            encoded_frame = self.encoder(observations)
            encoded_frame = torch.reshape(encoded_frame, (encoded_frame.size(0), -1))
        encoded_frame = self.encoder_lin_layer(encoded_frame)
        
        # Process each skill through MLP and compute attention weights
        for i, skill_emb in enumerate(skills_embeddings):
            skill_emb = self.mlp_layers[i](skill_emb)
            skills_embeddings[i] = skill_emb
            
            concatenated = torch.cat([encoded_frame, skill_emb], 1)
            weight = self.weights(concatenated)
            weights.append(weight)
        
        # Compute attention
        weights = torch.stack(weights, 1)
        weights = torch.softmax(weights, 1)
        
        # Store attention weights for monitoring (matching SB3 implementation)
        self.training_weights.append(weights.squeeze(-1).detach())
        
        # Save attention weights per skill
        for i, skill in enumerate(self.skills):
            self.att_weights[skill.name] = [w[i] for w in weights]
        
        # Stack skills and apply attention
        stacked_skills = torch.stack(skills_embeddings, 0).permute(1, 0, 2)
        att_out = weights * stacked_skills
        att_out = torch.sum(att_out, 1)
        
        return att_out


class AttentionPolicy(nn.Module):
    """
    PufferLib-compatible policy with attention-based feature extraction.
    This follows PufferLib's pure PyTorch policy design.
    """
    def __init__(
        self,
        env,
        skills: List[Skill],
        features_dim: int = 256,
        hidden_size: int = 512,
        device="cpu",
    ):
        super().__init__()
        
        # Feature extractor
        self.feature_extractor = AttentionFeatureExtractor(
            skills=skills,
            features_dim=features_dim,
            device=device,
        )
        
        # Get action space info from environment
        if hasattr(env, 'single_action_space'):
            action_space = env.single_action_space
        else:
            action_space = env.action_space
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(features_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_space.n),
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(features_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )
    
    def forward(self, observations, state=None):
        """
        Forward pass for training.
        
        Args:
            observations: Environment observations
            state: Optional dict with LSTM state (for RNN policies, ignored here)
            
        Returns:
            Tuple of (logits, value) - PufferLib expects only 2 values
        """
        # Extract features (handles PufferLib's extra dimension internally)
        features = self.feature_extractor(observations)
        
        # Get policy logits and value
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        # Return only (logits, value) - state is not returned for non-recurrent policies
        return logits, value
    
    def get_value(self, observations):
        """Get value estimate for observations"""
        features = self.feature_extractor(observations)
        return self.critic(features).squeeze(-1)
    
    def get_action_and_value(self, observations, action=None):
        """
        Get action and value for observations.
        This is the main method used during training.
        """
        features = self.feature_extractor(observations)
        
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        # Sample action if not provided
        probs = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), value
    
    def forward_eval(self, observations, state=None):
        """
        Forward pass for evaluation (required by PuffeRL).
        
        Args:
            observations: Environment observations
            state: RNN state (not used for non-recurrent policies)
            
        Returns:
            Tuple of (logits, value) - note: does NOT return state for eval
        """
        # Extract features
        features = self.feature_extractor(observations)
        
        # Get policy logits and value
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        
        # Return only (logits, value) for evaluation - PufferLib expects 2 values
        return logits, value


