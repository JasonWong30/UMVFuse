import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import models


class Fusion_loss2(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_func_grad = GradientMaxLoss()
        self.resnet_loss = ResNet18PerceptualLoss()
        self.loss_func_consist = IntensityConsistencyLoss()
        self.loss_func_temp = TemporalSecondOrderTV()

    def forward(
        self,
        img_f,
        img_ir,
        img_vi,
        ir_compose=1,
        temp_weight=0.5,
    ):
        loss_temp = temp_weight * self.loss_func_temp(img_f, img_ir, img_vi)

        img_vi = rearrange(img_vi, 'b c t h w -> (b t) c h w')
        img_ir = rearrange(img_ir, 'b c t h w -> (b t) c h w')
        img_f = rearrange(img_f, 'b c t h w -> (b t) c h w')

        loss_perceptual = (
            self.resnet_loss(img_f, img_ir)
            + self.resnet_loss(img_f, img_vi)
        )

        loss_gradient = 0.0
        for channel in range(img_f.shape[1]):
            fused_channel = img_f[:, channel:channel + 1, ...]
            visible_channel = img_vi[:, channel:channel + 1, ...]
            infrared_channel = img_ir[:, channel:channel + 1, ...]
            loss_gradient += self.loss_func_grad(
                fused_channel,
                visible_channel,
                infrared_channel,
            )

        loss_consistency = self.loss_func_consist(
            img_f,
            img_vi,
            img_ir,
            ir_compose,
        )
        total_loss = (
            loss_consistency
            + loss_perceptual
            + loss_gradient
            + loss_temp
        )

        return {
            'loss_temp': loss_temp,
            'loss_vgg': loss_perceptual,
            'loss_grad': loss_gradient,
            'loss_pixel_consist': loss_consistency,
            'loss_fusion': total_loss,
        }


class GradientMaxLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.grad_operator = SobelGradient()

    def forward(self, img_f, img_vi, img_ir):
        grad_vi_x, grad_vi_y = self.grad_operator(img_vi)
        grad_ir_x, grad_ir_y = self.grad_operator(img_ir)
        grad_f_x, grad_f_y = self.grad_operator(img_f)

        alpha = 0.4
        visible_weight = torch.sigmoid(
            alpha * ((grad_vi_x.abs() + grad_vi_y.abs())
            - (grad_ir_x.abs() + grad_ir_y.abs()))
        )
        target_x = visible_weight * grad_vi_x + (1 - visible_weight) * grad_ir_x
        target_y = visible_weight * grad_vi_y + (1 - visible_weight) * grad_ir_y

        return F.l1_loss(grad_f_x, target_x) + F.l1_loss(grad_f_y, target_y)


class SobelGradient(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
        ).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def forward(self, image):
        original_dtype = image.dtype
        with torch.amp.autocast('cuda', enabled=False):
            image_float = image.float()
            image_float = F.pad(image_float, (1, 1, 1, 1), mode='replicate')
            grad_x = F.conv2d(image_float, self.sobel_x.float())
            grad_y = F.conv2d(image_float, self.sobel_y.float())
        return grad_x.abs().to(original_dtype), grad_y.abs().to(original_dtype)


class IntensityConsistencyLoss(nn.Module):
    def forward(
        self,
        img_f,
        img_vi,
        img_ir,
        ir_compose,
    ):
        return (
            F.l1_loss(img_vi, img_f)
            + ir_compose * F.l1_loss(img_ir, img_f)
        )


class ResNet18FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        try:
            backbone = models.resnet18(
                weights=models.ResNet18_Weights.IMAGENET1K_V1
            )
        except Exception:
            print(
                'Failed to load pretrained ResNet18 weights; '
                'using random initialization.'
            )
            backbone = models.resnet18(weights=None)

        self.slice1 = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
        )
        self.slice2 = nn.Sequential(backbone.maxpool, backbone.layer1)
        self.slice3 = backbone.layer2
        self.slice4 = backbone.layer3
        self.slice5 = backbone.layer4

        for parameter in self.parameters():
            parameter.requires_grad = False

    def forward(self, image):
        feature1 = self.slice1(image)
        feature2 = self.slice2(feature1)
        feature3 = self.slice3(feature2)
        feature4 = self.slice4(feature3)
        feature5 = self.slice5(feature4)
        return [feature1, feature2, feature3, feature4, feature5]


class ResNet18PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet18FeatureExtractor().eval()
        self.weights = [1, 1 / 4, 1 / 8, 1 / 16, 1 / 32]

    @torch.no_grad()
    def extract(self, image):
        return self.resnet(image)

    def forward(self, prediction, target):
        mean = prediction.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = prediction.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        prediction_features = self.extract((prediction - mean) / std)
        target_features = self.extract((target - mean) / std)

        return sum(
            weight * F.l1_loss(prediction_feature, target_feature)
            for weight, prediction_feature, target_feature in zip(
                self.weights,
                prediction_features,
                target_features,
            )
        )


class TemporalSecondOrderTV(nn.Module):
    @staticmethod
    def _second_order_diff(video):
        previous = video[:, :, :-2]
        current = video[:, :, 1:-1]
        following = video[:, :, 2:]
        return following - 2 * current + previous

    def forward(self, video, ref_ir, ref_vi):
        if video.size(2) < 3:
            return video.new_tensor(0.0)

        fused_difference = self._second_order_diff(video)
        infrared_difference = self._second_order_diff(ref_ir)
        visible_difference = self._second_order_diff(ref_vi)
        return (
            (fused_difference - infrared_difference).abs().mean()
            + (fused_difference - visible_difference).abs().mean()
        ) / 2
