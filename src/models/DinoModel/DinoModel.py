from .Encoders import EncoderDino
import torch
import torch.nn as nn


class TokenClassifier(nn.Module):
    def __init__(self, embedding_size, num_classes, **kwargs):
        super().__init__()

        # self.norm = nn.LayerNorm(embedding_size)
        self.output_layer = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        return self.output_layer(x)


class DinoModel(nn.Module):
    def __init__(self, **kwargs):
        super(DinoModel, self).__init__()

        # general params
        self.kwargs = kwargs
        self.checkpoint_dino_one = self.kwargs["checkpoint_dino_one"]
        self.checkpoint_dino_two = self.kwargs["checkpoint_dino_two"]
        self.device_one = self.kwargs["device_one"]
        self.device_two = self.kwargs["device_two"]
        # build encoder model for 4k images
        self.encoder_dino = EncoderDino(
            **{
                "model256_path": self.checkpoint_dino_one,
                "device256": self.device_one,
                "model4k_path": self.checkpoint_dino_two,
                "device4k": self.device_one,
            }
        )
        self.classifier_dino = TokenClassifier(
            **{"embedding_size": 192, "num_classes": 4}
        )

        self.init_grads()

    def init_grads(self):

        for p in self.encoder_dino.parameters():
            p.requires_grad = True

        for p in self.classifier_dino.parameters():
            p.requires_grad = True

    def encode(self, x, **kwargs):
        x = eval_transforms()(x).unsqueeze(dim=0)
        x = self.encoder_dino(x)
        return x

    def forward(self, x, **kwargs):
        e = self.encode(x)
        p = self.classifier_dino(e)
        return p
