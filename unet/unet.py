from .unet_blocks import *
import torch.nn.functional as F

class Unet(nn.Module):
    """
    A UNet (https://arxiv.org/abs/1505.04597) implementation.
    input_channels: the number of channels in the image (1 for greyscale and 3 for RGB)
    num_classes: the number of classes to predict
    num_filters: list with the amount of filters per layer
    apply_last_layer: boolean to apply last layer or not (not used in Probabilistic UNet)
    padidng: Boolean, if true we pad the images with 1 so that we keep the same dimensions
    """

    def __init__(self, input_channels, num_classes, num_filters, apply_last_layer=True, padding=True):
        super(Unet, self).__init__()
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.num_filters = num_filters
        self.padding = padding
        self.activation_maps = []
        self.apply_last_layer = True

        ### Encoder
        self.contracting_path = nn.ModuleList()
        for i in range(len(self.num_filters)):
            input = self.input_channels if i == 0 else output
            output = self.num_filters[i]
            if i == 0:
                pool = False
            else:
                pool = True
            self.contracting_path.append(DownConvBlock(input, output, padding, pool=pool))


        ### Decoder
        self.upsampling_path = nn.ModuleList()
        n = len(self.num_filters) - 2
        for i in range(n, -1, -1):
            input = output + self.num_filters[i]
            output = self.num_filters[i]
            self.upsampling_path.append(UpConvBlock(input, output, padding))

        ### To class layer
        if self.apply_last_layer:
            self.last_layer = nn.Conv2d(output, num_classes, kernel_size=1)
            nn.init.kaiming_normal_(self.last_layer.weight, mode='fan_in',nonlinearity='relu')
            nn.init.normal_(self.last_layer.bias)


    def down_forward(self, x):
        self.blocks = []
        for i, down in enumerate(self.contracting_path):
            x = down(x)
            if i != len(self.contracting_path)-1:
                self.blocks.append(x)

        # self.unet_enc_feat = x # x = 128, 10, 10, // blocks[-1] = 64, 20, 20
        return x


    def up_forward(self, x):

        for i, up in enumerate(self.upsampling_path):
            x = up(x, self.blocks[-i-1])

        # del self.blocks

        #Used for saving the activations and plotting
        # if not training:
        #     self.activation_maps.append(x)

        if self.apply_last_layer:
            x =  self.last_layer(x)

        return x # 4, 32, 160, 160 if last layer False



# def tile(a, dim, n_tile):
#     """
#     This function is taken form PyTorch forum and mimics the behavior of tf.tile.
#     Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
#     """
#     init_dim = a.size(dim)  # dim = repeat하려는 위치, n_tile= repeat하고픈 dim 수
#     repeat_idx = [1] * a.dim()  # a.dim() : a의 차원 수
#     repeat_idx[dim] = n_tile  # repeat_idx = [1,1,160]
#     a = a.repeat(*(repeat_idx))
#     order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(
#         a.device)
#     return torch.index_select(a, dim, order_index)