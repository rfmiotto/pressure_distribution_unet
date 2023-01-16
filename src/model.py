import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip_connection = self.double_conv(x)
        encoder_output = self.pool(skip_connection)
        return encoder_output, skip_connection


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super().__init__()
        if up_sample_mode == "conv_transpose":
            self.up_sample = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        elif up_sample_mode == "bilinear":
            self.up_sample = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=True
            )
        else:
            raise ValueError(
                "Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)"
            )
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up_sample(x)
        x = torch.cat([skip_connection, x], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    # pylint: disable=too-many-instance-attributes
    def __init__(self, in_channels=3, out_classes=3, up_sample_mode="conv_transpose"):
        super().__init__()
        self.up_sample_mode = up_sample_mode

        n_filters = [64, 128, 256, 512, 1024]

        # Downsampling Path
        self.down_conv1 = EncoderBlock(in_channels, n_filters[0])
        self.down_conv2 = EncoderBlock(n_filters[0], n_filters[1])
        self.down_conv3 = EncoderBlock(n_filters[1], n_filters[2])
        self.down_conv4 = EncoderBlock(n_filters[2], n_filters[3])
        # Bottleneck
        self.bottleneck = DoubleConv(n_filters[3], n_filters[4])
        # Upsampling Path
        self.up_conv4 = DecoderBlock(n_filters[4], n_filters[3], self.up_sample_mode)
        self.up_conv3 = DecoderBlock(n_filters[3], n_filters[2], self.up_sample_mode)
        self.up_conv2 = DecoderBlock(n_filters[2], n_filters[1], self.up_sample_mode)
        self.up_conv1 = DecoderBlock(n_filters[1], n_filters[0], self.up_sample_mode)
        # Final Convolution
        self.final_conv = nn.Conv2d(n_filters[0], out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1 = self.down_conv1(x)
        x, skip2 = self.down_conv2(x)
        x, skip3 = self.down_conv3(x)
        x, skip4 = self.down_conv4(x)
        x = self.bottleneck(x)
        x = self.up_conv4(x, skip4)
        x = self.up_conv3(x, skip3)
        x = self.up_conv2(x, skip2)
        x = self.up_conv1(x, skip1)
        x = self.final_conv(x)

        return x


def _test():
    batch_size = 1
    x = torch.rand(batch_size, 3, 512, 512)
    model = UNet()
    predictions = model(x)
    assert predictions.shape == x.shape


if __name__ == "__main__":
    _test()
