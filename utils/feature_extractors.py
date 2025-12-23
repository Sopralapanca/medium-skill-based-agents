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

    def preprocess_input(self, observations: torch.Tensor):
        """
        :param observations: torch tensor of shape (n_envs, n_stacked_frames, height, width)
        """
        self.skills_embeddings = []

        for skill in self.skills:
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