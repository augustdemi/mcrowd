# This code is from https://github.com/stefanknegt/Probabilistic-Unet-Pytorch.git
# This code is based on: https://github.com/SimonKohl/probabilistic_unet

from .unet_blocks import *
from .unet import Unet
from .utils import init_weights, init_weights_orthogonal_normal, l2_regularisation
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl
from torch.distributions import RelaxedOneHotCategorical as concrete
from torch.distributions import OneHotCategorical as discrete

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    """
    A convolutional neural network, consisting of len(num_filters) times a block of no_convs_per_block convolutional layers,
    after each block a pooling operation is performed. And after each convolutional layer a non-linear (ReLU) activation function is applied.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, num_classes=2, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.input_channels = input_channels
        self.num_filters = num_filters

        if posterior:
            # To accomodate for the mask that is concatenated at the channel axis, we increase the input_channels.
            self.input_channels += num_classes

        layers = []
        for i in range(len(self.num_filters)):
            """
            Determine input_dim and output_dim of conv layers in this block. The first layer is input x output,
            All the subsequent layers are output x output.
            """
            input_dim = self.input_channels if i == 0 else output_dim
            output_dim = num_filters[i]

            if i != 0:
                layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))

            # for _ in range(no_convs_per_block - 1):
        layers.append(nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True))
        layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=int(padding)))
        layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

        self.layers.apply(init_weights)

    def forward(self, input):
        output = self.layers(input)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A convolutional net that parametrizes a Gaussian distribution with axis aligned covariance matrix.
    """

    def __init__(self, input_channels, num_filters, no_convs_per_block, latent_dim, num_classes=2, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.input_channels = input_channels
        self.channel_axis = 1
        self.num_filters = num_filters
        self.no_convs_per_block = no_convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = 'Posterior'
            self.encoder = Encoder(self.input_channels, self.num_filters, self.no_convs_per_block,
                                   num_classes=num_classes,
                                   posterior=self.posterior)
        else:
            self.name = 'Prior'
            self.encoder = nn.Sequential(
                nn.Conv2d(num_filters[-1], num_filters[-1], (1, 1), stride=1),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=2, stride=2, padding=0, ceil_mode=True),
                nn.Conv2d(num_filters[-1], num_filters[-1], (1, 1), stride=1),
                nn.ReLU(inplace=True),
            )

        self.conv_layer = nn.Conv2d(num_filters[-1], self.latent_dim, (1, 1), stride=1)

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)

    def forward(self, input, segm=None):

        # If segmentation is not none, concatenate the mask to the channel axis of the input
        if segm is not None:
            input = torch.cat((input, segm), dim=1)

        encoding = self.encoder(input)  # [4, 128, 10, 10]
        # self.show_enc = encoding

        # We only want the mean of the resulting hxw image
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)

        # Convert encoding to 2 x latent dim and split up for mu and log_sigma
        logit = self.conv_layer(encoding)

        # We squeeze the second dimension twice, since otherwise it won't work when batch size is equal to 1
        logit = torch.squeeze(logit, dim=2)
        logit = torch.squeeze(logit, dim=2)


        # This is a multivariate normal with diagonal covariance matrix sigma
        # https://github.com/pytorch/pytorch/pull/11178
        # dist = Independent(Normal(loc=mu, scale=torch.sqrt(torch.exp(log_var))), 1)
        dist = discrete(logits=logit)
        relaxed_dist = concrete(logits=logit, temperature=0.66)
        return dist, relaxed_dist



class Fcomb(nn.Module):
    """
    A function composed of no_convs_fcomb times a 1x1 convolution that combines the sample taken from the latent space,
    and output of the UNet (the feature map) by concatenating them along their channel axis.
    """

    def __init__(self, num_filters, latent_dim, num_output_channels, num_classes, no_convs_fcomb, initializers,
                 use_tile=True):
        super(Fcomb, self).__init__()
        self.num_channels = num_output_channels  # output channels
        self.num_classes = num_classes
        self.channel_axis = 1
        self.spatial_axes = [2, 3]
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.no_convs_fcomb = no_convs_fcomb
        self.name = 'Fcomb'

        if self.use_tile:
            layers = []

            # Decoder of N x a 1x1 convolution followed by a ReLU activation function except for the last layer
            layers.append(nn.Conv2d(self.num_filters[-1] + self.latent_dim, self.num_filters[-1], kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

            for _ in range(no_convs_fcomb - 1):
                layers.append(nn.Conv2d(self.num_filters[-1], self.num_filters[-1], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)

            # self.last_layer = nn.Conv2d(self.num_filters[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                # self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                # self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)  # dim = repeat하려는 위치, n_tile= repeat하고픈 dim 수
        repeat_idx = [1] * a.dim()  # a.dim() : a의 차원 수
        repeat_idx[dim] = n_tile  # repeat_idx = [1,1,160]
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
            device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is batch_sizexlatent_dim and feature_map is batch_sizexno_channelsxHxW.
        So broadcast Z to batch_sizexlatent_dimxHxW. Behavior is exactly the same as tf.tile (verified)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, 2)  # (bs, latent_dim) -> (bs, l_d, 1)
            z = self.tile(z, 2, feature_map.shape[
                self.spatial_axes[0]])  # self.spatial_axes = (2,3), feature_map.shape=(4, 32, 160, 160)
            z = torch.unsqueeze(z, 3)
            z = self.tile(z, 3, feature_map.shape[self.spatial_axes[1]])

            # Concatenate the feature map (output of the UNet) and the sample taken from the latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            return self.layers(feature_map)


class ProbabilisticUnet(nn.Module):
    """
    A probabilistic UNet (https://arxiv.org/abs/1806.05034) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: is a list consisint of the amount of filters layer
    latent_dim: dimension of the latent space
    no_cons_per_block: no convs per block in the (convolutional) encoder of prior and posterior
    """

    def __init__(self, input_channels=1, num_classes=1, num_filters=[32, 64, 128, 192], latent_dim=6, no_convs_fcomb=4,
                 no_convs_per_block=3, beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        self.no_convs_per_block = no_convs_per_block
        self.no_convs_fcomb = no_convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        self.beta = beta
        self.z_prior_sample = 0

        self.unet = Unet(self.input_channels, self.num_classes, self.num_filters, apply_last_layer=False,
                         padding=True).to(device)
        self.prior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                             self.latent_dim).to(device)
        self.posterior = AxisAlignedConvGaussian(self.input_channels, self.num_filters, self.no_convs_per_block,
                                                 self.latent_dim, num_classes=num_classes, posterior=True).to(device)
        self.fcomb = Fcomb(self.num_filters, self.latent_dim, self.input_channels, self.num_classes,
                           self.no_convs_fcomb, {'w': 'orthogonal', 'b': 'normal'}, use_tile=True).to(device)

    def forward(self, patch, segm, training=True):
        """
        Construct prior latent space for patch and run patch through UNet,
        in case training is True also construct posterior latent space
        """
        # unet encoder
        self.unet_enc_feat = self.unet.down_forward(patch)


        # latent dist
        if training:
            self.posterior_latent_space, self.relaxed_post = self.posterior.forward(patch, segm)
        self.prior_latent_space, self.relaxed_prior = self.prior.forward(self.unet_enc_feat)

        if training:
            z = self.relaxed_post.rsample()
        else:
            z = self.prior_latent_space.sample()
        '''
        # expand z in spatial dim by replicating
        z = torch.unsqueeze(z, 2)  # (bs, latent_dim) -> (bs, l_d, 1)
        z = self.tile(z, 2, self.unet_enc_feat.shape[2])  # self.spatial_axes = (2,3), feature_map.shape=(4, 32, 160, 160)
        z = torch.unsqueeze(z, 3)
        z = self.tile(z, 3, self.unet_enc_feat.shape[3])
        x = torch.cat([self.unet_enc_feat, z], dim=1)  # channel dim concat
        '''
        x = self.fcomb.forward(self.unet_enc_feat, z)

        return self.unet.up_forward(x)



    def sample(self, testing=False, z_prior=None):
        """
        Sample a segmentation by reconstructing from a prior sample
        and combining this with UNet features
        """
        if z_prior is None:
            if testing:
                # You can choose whether you mean a sample or the mean here. For the GED it is important to take a sample.
                # z_prior = self.prior_latent_space.base_dist.loc
                z_prior = self.prior_latent_space.sample()
                # self.z_prior_sample = z_prior
            else:
                z_prior = self.relaxed_prior.rsample()
                # self.z_prior_sample = z_prior
        x = self.fcomb.forward(self.unet_enc_feat, z_prior)
        return self.unet.up_forward(x)


    def kl_divergence(self, analytic=True, calculate_posterior=False, z_posterior=None):
        """
        Calculate the KL divergence between the posterior and prior KL(Q||P)
        analytic: calculate KL analytically or via sampling from the posterior
        calculate_posterior: if we use samapling to approximate KL we can sample here or supply a sample
        """
        if analytic:
            # Neeed to add this to torch source code, see: https://github.com/pytorch/pytorch/issues/13545
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calculate_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob
        return kl_div

