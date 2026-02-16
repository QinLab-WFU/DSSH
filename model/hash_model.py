import os
import torch
import logging
import torch.nn as nn
import numpy as np
from typing import Union

from model.model import build_model
from stochman.nnj import L2Norm


class LinearHash(nn.Module):

    def __init__(self, inputDim=2048, outputDim=64):
        super(LinearHash, self).__init__()
        self.fc = nn.Linear(inputDim, outputDim)
        self.drop_out = nn.Dropout(p=0.2)

    def forward(self, data):
        result = self.fc(data)
        return torch.tanh(self.drop_out(result))


class LAM(nn.Module):

    def __init__(self,
                 num_class=80,
                 outputDim=64,
                 clipPath="./ViT-B-32.pt",
                 writer=None,
                 saveDir="./result/log"):
        super(LAM, self).__init__()
        os.makedirs(saveDir, exist_ok=True)
        self.embedDim, self.clip = self.load_clip(clipPath)
        self.image_hash = LinearHash(inputDim=self.embedDim, outputDim=outputDim)
        self.text_hash = LinearHash(inputDim=self.embedDim, outputDim=outputDim)
        # self.image_linear = nn.Sequential(nn.Linear(outputDim, outputDim), L2Norm())
        # self.text_linear = nn.Sequential(nn.Linear(outputDim, outputDim), L2Norm())

    def freezen(self):
        for name, param in self.clip.named_parameters():
            # print(name)
            if name.find("ln_final.") == 0 or name.find("text_projection") == 0 or name.find("logit_scale") == 0 \
                    or name.find("visual.ln_post.") == 0 or name.find("visual.proj") == 0:
                # print("1")
                continue
            elif name.find("visual.transformer.resblocks.") == 0 or name.find("transformer.resblocks.") == 0:
                layer_num = int(name.split(".resblocks.")[1].split(".")[0])
                if layer_num >= 12:
                    # print("2")
                    continue
            if name.find("conv2.") == 0:
                # print("3")
                continue
            else:
                # paramenters which < freeze_layer_num will be freezed
                param.requires_grad = False

    def load_clip(self, clipPath: str) -> tuple:
        try:
            model = torch.jit.load(clipPath, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(clipPath, map_location="cpu")

        return state_dict["text_projection"].shape[1], build_model(state_dict)

    def encode(self, image, text):

        image_embed = self.clip.encode_image(image)  # b * 512

        text_embed = self.clip.encode_text(text)

        image_embed = self.image_hash(image_embed)

        text_embed = self.text_hash(text_embed)

        return image_embed, text_embed

    def linear(self, image_embed, text_embed):

        image_line = self.image_linear(image_embed)

        text_line = self.text_linear(text_embed)

        return image_line, text_line

    def eval(self):
        self.image_hash.eval()
        self.text_hash.eval()
        # self.image_linear.eval()
        # self.text_linear.eval()

        # self.clip.eval()

    def train(self):
        self.image_hash.train()
        self.text_hash.train()
        # self.image_linear.train()
        # self.text_linear.train()

    def forward(self, image, text):

        image_embed, text_embed = self.encode(image, text)
        return image_embed, text_embed

    # def forward(self, image, text, label, index):
    #
    #     image_embed, text_embed = self.encode(image, text)
    #
    #     if self.training:
    #         p = self.proxy(image, label, index)
    #
    #     return image_embed, text_embed
