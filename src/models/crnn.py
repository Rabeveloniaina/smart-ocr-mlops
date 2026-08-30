import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional
import yaml


def load_params(params_path: str = "params.yaml") -> dict:
    with open(params_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class CNN_FeatureExtractor(nn.Module):
    def __init__(self, pretrained: bool = True, in_channels: int = 1):
        super(CNN_FeatureExtractor, self).__init__()

        resnet = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT if pretrained else None
        )

        if in_channels == 1:
            resnet.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        if hasattr(resnet.layer3[0], "conv1"):
            resnet.layer3[0].conv1.stride = (2, 1)
        if hasattr(resnet.layer3[0], "downsample") and resnet.layer3[0].downsample is not None:
            resnet.layer3[0].downsample[0].stride = (2, 1)

        if hasattr(resnet.layer4[0], "conv1"):
            resnet.layer4[0].conv1.stride = (2, 1)
        if hasattr(resnet.layer4[0], "downsample") and resnet.layer4[0].downsample is not None:
            resnet.layer4[0].downsample[0].stride = (2, 1)

        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
        )

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None))
        self.output_channels = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.adaptive_pool(x)
        return x


class BidirectionalLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super(BidirectionalLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=False,
        )

        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        recurrent, _ = self.lstm(x)
        recurrent = self.dropout(recurrent)
        output = self.linear(recurrent)
        return output


class CRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        img_height: int = 32,
        hidden_size: int = 256,
        num_rnn_layers: int = 2,
        rnn_dropout: float = 0.3,
        pretrained: bool = True,
    ):
        super(CRNN, self).__init__()

        self.num_classes = num_classes
        self.img_height = img_height

        self.cnn = CNN_FeatureExtractor(pretrained=pretrained, in_channels=1)
        cnn_output_size = self.cnn.output_channels

        self.projection = nn.Sequential(
            nn.Linear(cnn_output_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(rnn_dropout),
        )

        self.rnn = BidirectionalLSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=num_classes,
            num_layers=num_rnn_layers,
            dropout=rnn_dropout,
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for module in [self.projection, self.rnn]:
            for m in module.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
                elif isinstance(m, nn.LSTM):
                    for name, param in m.named_parameters():
                        if "weight" in name:
                            nn.init.orthogonal_(param)
                        elif "bias" in name:
                            nn.init.zeros_(param)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.cnn(x)
        B, C, H, W = features.size()
        features = features.squeeze(2)
        features = features.permute(2, 0, 1)
        features = self.projection(features)
        logits = self.rnn(features)
        log_probs = nn.functional.log_softmax(logits, dim=2)
        return log_probs

    def get_sequence_length(self, input_width: int) -> int:
        return max(1, input_width // 4)

    @staticmethod
    def from_params(params_path: str = "params.yaml") -> "CRNN":
        params = load_params(params_path)
        charset = params["data"]["charset"]
        num_classes = len(charset) + 1
        return CRNN(
            num_classes=num_classes,
            img_height=params["data"]["image_height"],
            hidden_size=params["model"]["rnn_hidden_size"],
            num_rnn_layers=params["model"]["rnn_num_layers"],
            rnn_dropout=params["model"]["rnn_dropout"],
            pretrained=params["model"]["pretrained_backbone"],
        )

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def freeze_backbone(self):
        for param in self.cnn.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.cnn.parameters():
            param.requires_grad = True
