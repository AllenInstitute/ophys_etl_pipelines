from torch import nn


class CNN(nn.Module):
    def __init__(self, conv_cfg, classifier_cfg, batch_norm=True, batch_norm_before_nonlin=True, num_classes=1,
                 dropout_prob=0.5):
        super().__init__()
        self.features = self._make_conv_layers(cfg=conv_cfg, batch_norm=batch_norm,
                                               batch_norm_before_nonlin=batch_norm_before_nonlin)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        last_conv_filter_num = conv_cfg[-2]
        in_features = last_conv_filter_num * 7 * 7
        self.classifier = self._make_classifier_layers(cfg=classifier_cfg, in_features=in_features,
                                                       dropout_prob=dropout_prob, num_classes=num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.reshape(x.size(0), -1)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_conv_layers(cfg, batch_norm=True, batch_norm_before_nonlin=True):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    if batch_norm_before_nonlin:
                        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                    else:
                        layers += [conv2d, nn.ReLU(inplace=True), nn.BatchNorm2d(v)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    @staticmethod
    def _make_classifier_layers(cfg, in_features, dropout_prob, num_classes=1):
        layers = [
            nn.Linear(in_features=in_features, out_features=cfg[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob)
        ]
        for i, v in enumerate(cfg[1:], start=1):
            layers += [
                nn.Linear(cfg[i-1], cfg[i]),
                nn.ReLU(inplace=True),
                nn.Dropout(p=dropout_prob)
            ]

        layers.append(nn.Linear(cfg[-1], num_classes))
        return nn.Sequential(*layers)



