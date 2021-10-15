import torch
import torchvision
from torch import nn


class ResnetBackbone(nn.Module):
    def __init__(self, classifier_cfg, truncate_to_layer=None, freeze_all_layers=True, freeze_up_to_layer=None):
        if freeze_all_layers and freeze_up_to_layer is not None:
            raise ValueError('Use either freeze_all_layers or freeze_up_to_layer')

        super().__init__()
        model = torchvision.models.resnet34(pretrained=True, progress=False)
        if truncate_to_layer is not None:
            layers = self._truncate_to_layer(model=model, layer=truncate_to_layer)
        else:
            layers = list(model.children())[:-2]            # exclude classifier and avg pool
        self.features = torch.nn.Sequential(*layers)

        if freeze_all_layers:
            for p in self.features.parameters():
                p.requires_grad = False

        elif freeze_up_to_layer is not None:
            for layer in layers[:freeze_up_to_layer]:
                for p in layer.parameters():
                    p.requires_grad = False

        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

        last_conv_filter_num = self._get_last_filter_num()
        in_features = last_conv_filter_num
        self.classifier = self._make_classifier_layers(cfg=classifier_cfg, in_features=in_features)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _truncate_to_layer(model, layer):
        layers = list(model.children())
        return layers[:layer]

    def _get_last_filter_num(self):
        return [x for x in self.features[-1][-1].children()][-1].num_features

    @staticmethod
    def _make_classifier_layers(cfg, in_features, dropout_prob=0.0, num_classes=1):
        cfg.insert(0, in_features)
        layers = []
        for i, v in enumerate(cfg[1:], start=1):
            layers += [
                nn.Linear(cfg[i - 1], cfg[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob)
            ]

        layers.append(nn.Linear(cfg[-1], num_classes))
        return nn.Sequential(*layers)


if __name__ == '__main__':
    ResnetBackbone(classifier_cfg=[512], truncate_to_layer=-4, freeze_all_layers=False, freeze_up_to_layer=-1)