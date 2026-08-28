"""
seresnet18_vitals_11lead.py

SE-ResNet-18 for 11-lead ECG + vital-sign fusion.
Adapted from seresnet18.py to match MobileNetV2_1D_Vitals interface.

ECG input   : (batch, 11, segment_samples)
Vitals input: (batch, 3)  [RR, HR, Temp]
"""

import torch
import torch.nn as nn


# ── SE BLOCK ─────────────────────────────────────────────────
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        return x * self.fc(y).view(b, c, 1).expand_as(x)


# ── CONV HELPERS ─────────────────────────────────────────────
def conv7x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=7,
                     stride=stride, padding=3, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=1,
                     stride=stride, bias=False)


# ── BASIC BLOCK ───────────────────────────────────────────────
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dropout=0.2):
        super().__init__()
        self.conv1      = conv7x1(inplanes, planes, stride)
        self.bn1        = nn.BatchNorm1d(planes)
        self.relu       = nn.ReLU(inplace=True)
        self.dropout    = nn.Dropout(dropout)
        self.conv2      = conv7x1(planes, planes)
        self.bn2        = nn.BatchNorm1d(planes)
        self.se         = SELayer(planes)
        self.downsample = downsample
        self.stride     = stride

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.se(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


# ── SE-RESNET-18 WITH VITALS FUSION ──────────────────────────
class SEResNet18_1D_Vitals(nn.Module):
    """
    SE-ResNet-18 for 11-lead ECG + vitals fusion.

    Matches the MobileNetV2_1D_Vitals interface:
        logits = model(ecg, vitals)
        ecg    : (batch, 11, segment_samples)
        vitals : (batch, 3)
    """

    def __init__(
        self,
        input_channels=11,
        num_classes=19,
        vitals_dim=3,
        vitals_hidden_dim=16,
        dropout_rate=0.2,
        layers=(2, 2, 2, 2),
        zero_init_residual=False,
    ):
        super().__init__()
        self.inplanes = 64

        # ── Stem ─────────────────────────────────────────────
        self.stem = nn.Sequential(
            nn.Conv1d(input_channels, 64, kernel_size=15,
                      stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # ── Residual stages ──────────────────────────────────
        self.layer1 = self._make_layer(BasicBlock,  64, layers[0], dropout=dropout_rate)
        self.layer2 = self._make_layer(BasicBlock, 128, layers[1], stride=2, dropout=dropout_rate)
        self.layer3 = self._make_layer(BasicBlock, 256, layers[2], stride=2, dropout=dropout_rate)
        self.layer4 = self._make_layer(BasicBlock, 512, layers[3], stride=2, dropout=dropout_rate)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # ── Vitals branch (mirrors MobileNetV2) ──────────────
        self.vitals_fc = nn.Sequential(
            nn.Linear(vitals_dim, vitals_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        # ── Classifier ───────────────────────────────────────
        self.dropout    = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(512 * BasicBlock.expansion + vitals_hidden_dim,
                                    num_classes)

        # ── Weight init ──────────────────────────────────────
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dropout=0.2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample, dropout)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x, vitals):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)           # (B, 512)

        v = self.vitals_fc(vitals.float()) # (B, vitals_hidden_dim)

        x = self.dropout(x)
        x = torch.cat((x, v), dim=1)      # (B, 512 + vitals_hidden_dim)
        return self.classifier(x)


def seresnet18_1d(**kwargs):
    return SEResNet18_1D_Vitals(
        input_channels=11,
        vitals_dim=3,
        **kwargs,
    )