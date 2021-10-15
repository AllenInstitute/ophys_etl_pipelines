import torch
import torchvision
from torch import nn


class VggBackbone(torch.nn.Module):
    def __init__(self, model, truncate_to_layer, classifier_cfg, dropout_prob=0.5, freeze_layers=True):
        super().__init__()
        conv_layers = self._truncate_to_layer(model=model, layer=truncate_to_layer)
        self.features = torch.nn.Sequential(*conv_layers)

        if freeze_layers:
            for p in self.features.parameters():
                p.requires_grad = False

        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))

        last_conv_filter_num = self._get_last_filter_num()
        in_features = last_conv_filter_num * 7 * 7
        self.classifier = self._make_classifier_layers(cfg=classifier_cfg, in_features=in_features,
                                                       dropout_prob=dropout_prob)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _truncate_to_layer(model, layer):
        conv_layers = list(model.children())[0]
        return conv_layers[:layer]

    def _get_last_filter_num(self):
        idx = -1
        while idx > -1 * len(self.features):
            if hasattr(self.features[idx], 'out_channels'):
                return self.features[idx].out_channels
            idx -= 1

        raise Exception('Could not find number of filters in last conv layer')

    @staticmethod
    def _make_classifier_layers(cfg, in_features, dropout_prob, num_classes=1):
        layers = [
            nn.Linear(in_features=in_features, out_features=cfg[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        ]
        for i, v in enumerate(cfg[1:], start=1):
            layers += [
                nn.Linear(cfg[i - 1], cfg[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob)
            ]

        layers.append(nn.Linear(cfg[-1], num_classes))
        return nn.Sequential(*layers)