import torchvision.models as models
from torch import nn
from torch.nn.functional import normalize

from stochman.nnj import L2Norm


class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        model_resnet = models.resnet50(pretrained=pretrained)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layer = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool,
                                           self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)
        self.tanh = nn.Tanh()
        # nn.Sequential(nn.Linear(1024, latent_dim), L2Norm())
        self.linear = nn.Sequential(
            nn.Linear(model_resnet.fc.in_features, 4096),
            L2Norm()
        )
        # if config['without_BN']:
        #     self.hash_layer = nn.Linear(model_resnet.fc.in_features, 4096)
        #     self.hash_layer.weight.data.normal_(0, 0.01)
        #     self.hash_layer.bias.data.fill_(0.0)
        # else:
        #     self.layer_hash = nn.Linear(model_resnet.fc.in_features, 4096)
        #     self.layer_hash.weight.data.normal_(0, 0.01)
        #     self.layer_hash.bias.data.fill_(0.0)
        #     self.hash_layer = nn.Sequential(self.layer_hash, nn.BatchNorm1d(4096, momentum=0.1))

    def forward(self, x):

        feat = self.feature_layer(x)
        feat = feat.view(feat.shape[0], -1)
        feat = self.tanh(feat)
        # x = self.hash_layer(feat)
        # x = self.tanh(x)

        return feat