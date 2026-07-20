import torch
import torch.nn as nn


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    return new_v + divisor if new_v < 0.9 * v else new_v


class DepthwiseConv1D(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )

    def forward(self, x):
        return self.depthwise(x)


class InvertedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expansion, kernel_size, dropout_rate, block_id):
        super().__init__()

        self.use_residual = (in_channels == out_channels) and (stride == 1)

        mid_channels = in_channels * expansion
        padding = kernel_size // 2 if stride == 1 else 0

        layers = []
        if block_id != 0:
            layers.extend([
                nn.Conv1d(in_channels, mid_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(mid_channels),
                nn.ReLU6(inplace=True),
            ])

        layers.extend([
            DepthwiseConv1D(mid_channels, kernel_size, stride, padding),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU6(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            return x + out
        return out


class MobileNetV2_1D_Vitals(nn.Module):
    """
    1D MobileNetV2 for ECG + vital-sign fusion.

    ECG input shape:
        x: (batch_size, 11, segment_samples)

    Vitals input shape:
        vitals: (batch_size, 3), ordered as [RR, HR, Temp]

    Important:
        The model uses AdaptiveAvgPool1d, so segment_samples can be any length.
        For 10-second ECG windows, set segment_samples = sampling_rate * 10
        in your dataloader/transforms.
    """

    def __init__(
        self,
        input_channels=11,
        alpha=1.0,
        num_classes=10,
        vitals_dim=3,
        vitals_hidden_dim=16,
        stride_size=4,
        kernel_size=9,
        dropout_rate=0.3,
        include_top=True,
        pooling=None,
    ):
        super().__init__()

        if isinstance(stride_size, int):
            stride_size = [stride_size] * 5
        elif len(stride_size) != 5:
            raise ValueError("stride_size must be an integer or a tuple/list of length 5.")

        self.include_top = include_top
        self.pooling = pooling
        self.vitals_dim = vitals_dim

        first_block_filters = make_divisible(32 * alpha, 8)

        self.initial = nn.Sequential(
            nn.Conv1d(input_channels, first_block_filters, kernel_size=3, stride=stride_size[0], padding=1, bias=False),
            nn.BatchNorm1d(first_block_filters),
            nn.ReLU6(inplace=True),
        )

        block_params = [
            (16, 1, 1), (24, stride_size[1], 6), (24, 1, 6),
            (32, stride_size[2], 6), (32, 1, 6), (32, 1, 6),
            (64, stride_size[3], 6), (64, 1, 6), (64, 1, 6), (64, 1, 6),
            (96, 1, 6), (96, 1, 6), (96, 1, 6),
            (160, stride_size[4], 6), (160, 1, 6), (160, 1, 6),
            (320, 1, 6),
        ]

        blocks = []
        in_channels = first_block_filters
        for i, (filters, stride, expansion) in enumerate(block_params):
            out_channels = make_divisible(filters * alpha, 8)
            blocks.append(InvertedResidualBlock(in_channels, out_channels, stride, expansion, kernel_size, 0.1, i))
            in_channels = out_channels

        self.blocks = nn.Sequential(*blocks)

        last_block_filters = make_divisible(1280 * alpha, 8) if alpha > 1.0 else 1280
        self.final = nn.Sequential(
            nn.Conv1d(in_channels, last_block_filters, kernel_size=1, bias=False),
            nn.BatchNorm1d(last_block_filters),
            nn.ReLU6(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1) if include_top or pooling == "avg" else (
            nn.AdaptiveMaxPool1d(1) if pooling == "max" else nn.Identity()
        )

        self.vitals_fc = nn.Sequential(
            nn.Linear(vitals_dim, vitals_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )

        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(last_block_filters + vitals_hidden_dim, num_classes)

    def forward(self, x, vitals=None):
        x = self.initial(x)
        x = self.blocks(x)
        x = self.final(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)

        if self.include_top:
            if vitals is None:
                raise ValueError("vitals must be provided with shape (batch_size, 3): [RR, HR, Temp].")
            vitals = vitals.float()
            vitals_features = self.vitals_fc(vitals)
            x = self.dropout(x)
            x = torch.cat((x, vitals_features), dim=1)
            x = self.classifier(x)

        return x


def mobilenetv2_1d(**kwargs):
    """
    Constructs a 1D MobileNetV2 model with RR/HR/temperature fusion.

    Default ECG input channels: 11 leads.
    Vitals vector: [RR, HR, Temp].

    Example for 10 seconds at 500 Hz:
        ecg = torch.randn(8, 11, 5000)
        vitals = torch.randn(8, 3)
        logits = model(ecg, vitals)
    """
    model = MobileNetV2_1D_Vitals(
        alpha=1.0,
        stride_size=(2, 2, 2, 2, 2),
        kernel_size=9,
        input_channels=11,
        vitals_dim=3,
        **kwargs,
    )
    return model
