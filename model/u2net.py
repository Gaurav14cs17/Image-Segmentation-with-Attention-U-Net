"""
Segmentation Models

Available models:
- AttentionUNet: Pure attention-based with learnable downsampling (recommended)
- U2Net: Full U²-Net (~44M params)
- U2NetSmall: Lightweight U²-Net (~1.1M params)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Building Blocks
# =============================================================================

class ConvBNReLU(nn.Module):
    """Conv2d + BatchNorm + ReLU"""
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel, stride, padding, dilation)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ChannelAttention(nn.Module):
    """Channel Attention - learns which channels are important"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    """Spatial Attention - learns where to focus"""
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv(torch.cat([avg_out, max_out], dim=1)))


class CBAM(nn.Module):
    """CBAM: Channel + Spatial Attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()

    def forward(self, x):
        x = x * self.channel_att(x)
        x = x * self.spatial_att(x)
        return x


class LearnableDownsample(nn.Module):
    """Learnable downsampling (strided conv) - better than MaxPool"""
    def __init__(self, in_ch, out_ch=None):
        super().__init__()
        out_ch = out_ch or in_ch
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class AttentionBlock(nn.Module):
    """Conv + Conv + CBAM with residual connection"""
    def __init__(self, in_ch, out_ch, reduction=8):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = ConvBNReLU(out_ch, out_ch)
        self.attention = CBAM(out_ch, reduction)
        self.residual = (nn.Identity() if in_ch == out_ch
                         else nn.Conv2d(in_ch, out_ch, 1, bias=False))

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention(x)
        return x + res


# =============================================================================
# AttentionUNet - Pure Attention + Learnable Downsampling (Recommended)
# =============================================================================

class AttentionUNet(nn.Module):
    """
    Pure Attention-based U-Net with Learnable Downsampling

    Features:
    - NO MaxPool - uses learnable strided convolutions
    - CBAM attention blocks throughout
    - Attention-gated skip connections
    - Multi-scale fusion with learnable weights
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        ch = [32, 64, 128, 256, 256]  # Channel progression

        # Initial conv
        self.init_conv = ConvBNReLU(in_channels, ch[0])

        # Encoder
        self.enc1 = AttentionBlock(ch[0], ch[0])
        self.down1 = LearnableDownsample(ch[0], ch[1])
        self.enc2 = AttentionBlock(ch[1], ch[1])
        self.down2 = LearnableDownsample(ch[1], ch[2])
        self.enc3 = AttentionBlock(ch[2], ch[2])
        self.down3 = LearnableDownsample(ch[2], ch[3])
        self.enc4 = AttentionBlock(ch[3], ch[3])
        self.down4 = LearnableDownsample(ch[3], ch[4])

        # Bottleneck
        self.bottleneck = AttentionBlock(ch[4], ch[4])

        # Decoder
        self.up4 = nn.Sequential(nn.Conv2d(ch[4], ch[3], 1), nn.BatchNorm2d(ch[3]), nn.ReLU(True))
        self.dec4 = AttentionBlock(ch[3] * 2, ch[3])
        self.up3 = nn.Sequential(nn.Conv2d(ch[3], ch[2], 1), nn.BatchNorm2d(ch[2]), nn.ReLU(True))
        self.dec3 = AttentionBlock(ch[2] * 2, ch[2])
        self.up2 = nn.Sequential(nn.Conv2d(ch[2], ch[1], 1), nn.BatchNorm2d(ch[1]), nn.ReLU(True))
        self.dec2 = AttentionBlock(ch[1] * 2, ch[1])
        self.up1 = nn.Sequential(nn.Conv2d(ch[1], ch[0], 1), nn.BatchNorm2d(ch[0]), nn.ReLU(True))
        self.dec1 = AttentionBlock(ch[0] * 2, ch[0])

        # Skip attention gates
        self.skip_att4 = CBAM(ch[3], 8)
        self.skip_att3 = CBAM(ch[2], 8)
        self.skip_att2 = CBAM(ch[1], 8)
        self.skip_att1 = CBAM(ch[0], 8)

        # Side outputs
        self.side1 = nn.Conv2d(ch[0], out_channels, 1)
        self.side2 = nn.Conv2d(ch[1], out_channels, 1)
        self.side3 = nn.Conv2d(ch[2], out_channels, 1)
        self.side4 = nn.Conv2d(ch[3], out_channels, 1)
        self.side5 = nn.Conv2d(ch[4], out_channels, 1)

        # Fusion
        self.dropout = nn.Dropout2d(0.1)
        self.fusion = nn.Conv2d(5 * out_channels, out_channels, 1)
        self.fusion_weights = nn.Parameter(torch.ones(5) / 5)

    def forward(self, x):
        size = x.shape[2:]

        # Encoder
        x = self.init_conv(x)
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        e4 = self.enc4(self.down3(e3))
        b = self.bottleneck(self.down4(e4))

        # Decoder
        d4 = self.up4(F.interpolate(b, e4.shape[2:], mode='bilinear', align_corners=True))
        d4 = self.dec4(torch.cat([d4, self.skip_att4(e4)], 1))

        d3 = self.up3(F.interpolate(d4, e3.shape[2:], mode='bilinear', align_corners=True))
        d3 = self.dec3(torch.cat([d3, self.skip_att3(e3)], 1))

        d2 = self.up2(F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=True))
        d2 = self.dec2(torch.cat([d2, self.skip_att2(e2)], 1))

        d1 = self.up1(F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=True))
        d1 = self.dec1(torch.cat([d1, self.skip_att1(e1)], 1))

        # Side outputs
        s1 = self.side1(self.dropout(d1))
        s2 = F.interpolate(self.side2(self.dropout(d2)), size, mode='bilinear', align_corners=True)
        s3 = F.interpolate(self.side3(self.dropout(d3)), size, mode='bilinear', align_corners=True)
        s4 = F.interpolate(self.side4(self.dropout(d4)), size, mode='bilinear', align_corners=True)
        s5 = F.interpolate(self.side5(self.dropout(b)), size, mode='bilinear', align_corners=True)

        # Fusion
        w = F.softmax(self.fusion_weights, dim=0)
        weighted = w[0]*s1 + w[1]*s2 + w[2]*s3 + w[3]*s4 + w[4]*s5
        fused = self.fusion(torch.cat([s1, s2, s3, s4, s5], 1))
        out = fused + weighted

        # Return 7 outputs for training compatibility
        return (torch.sigmoid(out), torch.sigmoid(s1), torch.sigmoid(s2),
                torch.sigmoid(s3), torch.sigmoid(s4), torch.sigmoid(s5), torch.sigmoid(s5))


# =============================================================================
# RSU Blocks for U2Net
# =============================================================================

class RSU7(nn.Module):
    """Residual U-block height 7"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        self.conv1 = ConvBNReLU(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = ConvBNReLU(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv3 = ConvBNReLU(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = ConvBNReLU(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv5 = ConvBNReLU(mid_ch, mid_ch)
        self.pool5 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv6 = ConvBNReLU(mid_ch, mid_ch)
        self.conv7 = ConvBNReLU(mid_ch, mid_ch, dilation=2, padding=2)

        self.conv6d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv5d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv4d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv3d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv2d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv1d = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.conv_in(x)
        hx1 = self.conv1(hxin)
        hx2 = self.conv2(self.pool1(hx1))
        hx3 = self.conv3(self.pool2(hx2))
        hx4 = self.conv4(self.pool3(hx3))
        hx5 = self.conv5(self.pool4(hx4))
        hx6 = self.conv6(self.pool5(hx5))
        hx7 = self.conv7(hx6)

        hx6d = self.conv6d(torch.cat([hx7, hx6], 1))
        hx5d = self.conv5d(torch.cat([F.interpolate(hx6d, hx5.shape[2:], mode='bilinear', align_corners=True), hx5], 1))
        hx4d = self.conv4d(torch.cat([F.interpolate(hx5d, hx4.shape[2:], mode='bilinear', align_corners=True), hx4], 1))
        hx3d = self.conv3d(torch.cat([F.interpolate(hx4d, hx3.shape[2:], mode='bilinear', align_corners=True), hx3], 1))
        hx2d = self.conv2d(torch.cat([F.interpolate(hx3d, hx2.shape[2:], mode='bilinear', align_corners=True), hx2], 1))
        hx1d = self.conv1d(torch.cat([F.interpolate(hx2d, hx1.shape[2:], mode='bilinear', align_corners=True), hx1], 1))
        return hx1d + hxin


class RSU6(nn.Module):
    """Residual U-block height 6"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        self.conv1 = ConvBNReLU(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = ConvBNReLU(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv3 = ConvBNReLU(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = ConvBNReLU(mid_ch, mid_ch)
        self.pool4 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv5 = ConvBNReLU(mid_ch, mid_ch)
        self.conv6 = ConvBNReLU(mid_ch, mid_ch, dilation=2, padding=2)

        self.conv5d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv4d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv3d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv2d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv1d = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.conv_in(x)
        hx1 = self.conv1(hxin)
        hx2 = self.conv2(self.pool1(hx1))
        hx3 = self.conv3(self.pool2(hx2))
        hx4 = self.conv4(self.pool3(hx3))
        hx5 = self.conv5(self.pool4(hx4))
        hx6 = self.conv6(hx5)

        hx5d = self.conv5d(torch.cat([hx6, hx5], 1))
        hx4d = self.conv4d(torch.cat([F.interpolate(hx5d, hx4.shape[2:], mode='bilinear', align_corners=True), hx4], 1))
        hx3d = self.conv3d(torch.cat([F.interpolate(hx4d, hx3.shape[2:], mode='bilinear', align_corners=True), hx3], 1))
        hx2d = self.conv2d(torch.cat([F.interpolate(hx3d, hx2.shape[2:], mode='bilinear', align_corners=True), hx2], 1))
        hx1d = self.conv1d(torch.cat([F.interpolate(hx2d, hx1.shape[2:], mode='bilinear', align_corners=True), hx1], 1))
        return hx1d + hxin


class RSU5(nn.Module):
    """Residual U-block height 5"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        self.conv1 = ConvBNReLU(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = ConvBNReLU(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv3 = ConvBNReLU(mid_ch, mid_ch)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv4 = ConvBNReLU(mid_ch, mid_ch)
        self.conv5 = ConvBNReLU(mid_ch, mid_ch, dilation=2, padding=2)

        self.conv4d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv3d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv2d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv1d = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.conv_in(x)
        hx1 = self.conv1(hxin)
        hx2 = self.conv2(self.pool1(hx1))
        hx3 = self.conv3(self.pool2(hx2))
        hx4 = self.conv4(self.pool3(hx3))
        hx5 = self.conv5(hx4)

        hx4d = self.conv4d(torch.cat([hx5, hx4], 1))
        hx3d = self.conv3d(torch.cat([F.interpolate(hx4d, hx3.shape[2:], mode='bilinear', align_corners=True), hx3], 1))
        hx2d = self.conv2d(torch.cat([F.interpolate(hx3d, hx2.shape[2:], mode='bilinear', align_corners=True), hx2], 1))
        hx1d = self.conv1d(torch.cat([F.interpolate(hx2d, hx1.shape[2:], mode='bilinear', align_corners=True), hx1], 1))
        return hx1d + hxin


class RSU4(nn.Module):
    """Residual U-block height 4"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)

        self.conv1 = ConvBNReLU(out_ch, mid_ch)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv2 = ConvBNReLU(mid_ch, mid_ch)
        self.pool2 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.conv3 = ConvBNReLU(mid_ch, mid_ch)
        self.conv4 = ConvBNReLU(mid_ch, mid_ch, dilation=2, padding=2)

        self.conv3d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv2d = ConvBNReLU(mid_ch * 2, mid_ch)
        self.conv1d = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.conv_in(x)
        hx1 = self.conv1(hxin)
        hx2 = self.conv2(self.pool1(hx1))
        hx3 = self.conv3(self.pool2(hx2))
        hx4 = self.conv4(hx3)

        hx3d = self.conv3d(torch.cat([hx4, hx3], 1))
        hx2d = self.conv2d(torch.cat([F.interpolate(hx3d, hx2.shape[2:], mode='bilinear', align_corners=True), hx2], 1))
        hx1d = self.conv1d(torch.cat([F.interpolate(hx2d, hx1.shape[2:], mode='bilinear', align_corners=True), hx1], 1))
        return hx1d + hxin


class RSU4F(nn.Module):
    """Residual U-block with dilated convolutions (no pooling)"""
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch, out_ch)
        self.conv1 = ConvBNReLU(out_ch, mid_ch)
        self.conv2 = ConvBNReLU(mid_ch, mid_ch, dilation=2, padding=2)
        self.conv3 = ConvBNReLU(mid_ch, mid_ch, dilation=4, padding=4)
        self.conv4 = ConvBNReLU(mid_ch, mid_ch, dilation=8, padding=8)
        self.conv3d = ConvBNReLU(mid_ch * 2, mid_ch, dilation=4, padding=4)
        self.conv2d = ConvBNReLU(mid_ch * 2, mid_ch, dilation=2, padding=2)
        self.conv1d = ConvBNReLU(mid_ch * 2, out_ch)

    def forward(self, x):
        hxin = self.conv_in(x)
        hx1 = self.conv1(hxin)
        hx2 = self.conv2(hx1)
        hx3 = self.conv3(hx2)
        hx4 = self.conv4(hx3)
        hx3d = self.conv3d(torch.cat([hx4, hx3], 1))
        hx2d = self.conv2d(torch.cat([hx3d, hx2], 1))
        hx1d = self.conv1d(torch.cat([hx2d, hx1], 1))
        return hx1d + hxin


# =============================================================================
# U2Net Models
# =============================================================================

class U2Net(nn.Module):
    """U²-Net Full (~44M params)"""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.stage1 = RSU7(in_channels, 32, 64)
        self.pool12 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage2 = RSU6(64, 32, 128)
        self.pool23 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage3 = RSU5(128, 64, 256)
        self.pool34 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage4 = RSU4(256, 128, 512)
        self.pool45 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage5 = RSU4F(512, 256, 512)
        self.pool56 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage6 = RSU4F(512, 256, 512)

        # Decoder
        self.stage5d = RSU4F(1024, 256, 512)
        self.stage4d = RSU4(1024, 128, 256)
        self.stage3d = RSU5(512, 64, 128)
        self.stage2d = RSU6(256, 32, 64)
        self.stage1d = RSU7(128, 16, 64)

        # Side outputs
        self.side1 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side3 = nn.Conv2d(128, out_channels, 3, padding=1)
        self.side4 = nn.Conv2d(256, out_channels, 3, padding=1)
        self.side5 = nn.Conv2d(512, out_channels, 3, padding=1)
        self.side6 = nn.Conv2d(512, out_channels, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_channels, out_channels, 1)

    def forward(self, x):
        size = x.shape[2:]

        # Encoder
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))

        # Decoder
        hx5d = self.stage5d(torch.cat([F.interpolate(hx6, hx5.shape[2:], mode='bilinear', align_corners=True), hx5], 1))
        hx4d = self.stage4d(torch.cat([F.interpolate(hx5d, hx4.shape[2:], mode='bilinear', align_corners=True), hx4], 1))
        hx3d = self.stage3d(torch.cat([F.interpolate(hx4d, hx3.shape[2:], mode='bilinear', align_corners=True), hx3], 1))
        hx2d = self.stage2d(torch.cat([F.interpolate(hx3d, hx2.shape[2:], mode='bilinear', align_corners=True), hx2], 1))
        hx1d = self.stage1d(torch.cat([F.interpolate(hx2d, hx1.shape[2:], mode='bilinear', align_corners=True), hx1], 1))

        # Side outputs
        d1 = self.side1(hx1d)
        d2 = F.interpolate(self.side2(hx2d), size, mode='bilinear', align_corners=True)
        d3 = F.interpolate(self.side3(hx3d), size, mode='bilinear', align_corners=True)
        d4 = F.interpolate(self.side4(hx4d), size, mode='bilinear', align_corners=True)
        d5 = F.interpolate(self.side5(hx5d), size, mode='bilinear', align_corners=True)
        d6 = F.interpolate(self.side6(hx6), size, mode='bilinear', align_corners=True)
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))

        return (torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2),
                torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6))


class U2NetSmall(nn.Module):
    """U²-Net Small (~1.1M params)"""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        # Encoder
        self.stage1 = RSU7(in_channels, 16, 64)
        self.pool12 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage2 = RSU6(64, 16, 64)
        self.pool23 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage3 = RSU5(64, 16, 64)
        self.pool34 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage4 = RSU4(64, 16, 64)
        self.pool45 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage5 = RSU4F(64, 16, 64)
        self.pool56 = nn.MaxPool2d(2, 2, ceil_mode=True)
        self.stage6 = RSU4F(64, 16, 64)

        # Decoder
        self.stage5d = RSU4F(128, 16, 64)
        self.stage4d = RSU4(128, 16, 64)
        self.stage3d = RSU5(128, 16, 64)
        self.stage2d = RSU6(128, 16, 64)
        self.stage1d = RSU7(128, 16, 64)

        # Side outputs
        self.side1 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side2 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side3 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side4 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side5 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.side6 = nn.Conv2d(64, out_channels, 3, padding=1)
        self.outconv = nn.Conv2d(6 * out_channels, out_channels, 1)

    def forward(self, x):
        size = x.shape[2:]

        # Encoder
        hx1 = self.stage1(x)
        hx2 = self.stage2(self.pool12(hx1))
        hx3 = self.stage3(self.pool23(hx2))
        hx4 = self.stage4(self.pool34(hx3))
        hx5 = self.stage5(self.pool45(hx4))
        hx6 = self.stage6(self.pool56(hx5))

        # Decoder
        hx5d = self.stage5d(torch.cat([F.interpolate(hx6, hx5.shape[2:], mode='bilinear', align_corners=True), hx5], 1))
        hx4d = self.stage4d(torch.cat([F.interpolate(hx5d, hx4.shape[2:], mode='bilinear', align_corners=True), hx4], 1))
        hx3d = self.stage3d(torch.cat([F.interpolate(hx4d, hx3.shape[2:], mode='bilinear', align_corners=True), hx3], 1))
        hx2d = self.stage2d(torch.cat([F.interpolate(hx3d, hx2.shape[2:], mode='bilinear', align_corners=True), hx2], 1))
        hx1d = self.stage1d(torch.cat([F.interpolate(hx2d, hx1.shape[2:], mode='bilinear', align_corners=True), hx1], 1))

        # Side outputs
        d1 = self.side1(hx1d)
        d2 = F.interpolate(self.side2(hx2d), size, mode='bilinear', align_corners=True)
        d3 = F.interpolate(self.side3(hx3d), size, mode='bilinear', align_corners=True)
        d4 = F.interpolate(self.side4(hx4d), size, mode='bilinear', align_corners=True)
        d5 = F.interpolate(self.side5(hx5d), size, mode='bilinear', align_corners=True)
        d6 = F.interpolate(self.side6(hx6), size, mode='bilinear', align_corners=True)
        d0 = self.outconv(torch.cat([d1, d2, d3, d4, d5, d6], 1))

        return (torch.sigmoid(d0), torch.sigmoid(d1), torch.sigmoid(d2),
                torch.sigmoid(d3), torch.sigmoid(d4), torch.sigmoid(d5), torch.sigmoid(d6))


# =============================================================================
# Model Factory
# =============================================================================

def get_model(model_name='attention_unet', in_channels=3, out_channels=1):
    """
    Get segmentation model by name.

    Args:
        model_name: 'attention_unet' (recommended), 'u2net', or 'u2net_small'
        in_channels: Input channels (default: 3)
        out_channels: Output channels (default: 1)

    Returns:
        Model instance
    """
    models = {
        'attention_unet': AttentionUNet,
        'u2net': U2Net,
        'u2net_small': U2NetSmall,
    }
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")
    return models[model_name](in_channels, out_channels)


# =============================================================================
# Test
# =============================================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    x = torch.randn(1, 3, 320, 320).to(device)

    print("Model Comparison:")
    print("-" * 50)
    for name in ['attention_unet', 'u2net_small', 'u2net']:
        model = get_model(name).to(device)
        out = model(x)
        params = sum(p.numel() for p in model.parameters())
        print(f"{name:15} | {params:>12,} params | output: {out[0].shape}")
