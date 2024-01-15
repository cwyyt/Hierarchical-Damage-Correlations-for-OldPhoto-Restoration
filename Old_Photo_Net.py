"""
## Multi-Stage Progressive Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## https://arxiv.org/abs/2102.02808
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Discriminator import NLayerDiscriminator
import functools
from torch.nn import init
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms
from normalize import MAIN
##########################################################################
def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size//2), bias=bias, stride = stride)

##########################################################################
## Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


##########################################################################
## Channel Attention Block (CAB)

class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        modules_body.append(act)
        modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))

        self.CA = CALayer(n_feat, reduction, bias=bias)
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res = self.CA(res)
        res += x

        return res
############################################################################
class shallow_feat(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction,bias, act):
        super(shallow_feat, self).__init__()
        self.conv_shallow = conv(3, n_feat, kernel_size, bias=bias)
        self.CAB_shallow = CAB(n_feat, kernel_size, reduction, bias, act)
    def forward(self,x):
        x=self.conv_shallow(x)
        x=self.CAB_shallow(x)

        return x


################################################################################
class Encoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, num_heads, ffn_expansion_factor, bias, kernel_size, LayerNorm_type='WithBias'):
        super(Encoder, self).__init__()
        
        self.encoder_level1 = [channel_transformer_encoder(n_feat, num_heads[0], ffn_expansion_factor = ffn_expansion_factor, bias=bias, LayerNorm_type='WithBias') for _ in range(2)]
        self.encoder_level2 = [channel_transformer_encoder(n_feat*2**1, num_heads[1], ffn_expansion_factor = ffn_expansion_factor, bias=bias, LayerNorm_type='WithBias') for _ in range(2)]
        self.encoder_level3 = [channel_transformer_encoder(n_feat*2**2, num_heads[2], ffn_expansion_factor = ffn_expansion_factor, bias=bias, LayerNorm_type='WithBias') for _ in range(2)]
        self.encoder_level4 = [channel_transformer_encoder(n_feat*2**3, num_heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type='WithBias') for _ in range(2)]
       
        self.encoder_level1 = nn.Sequential(*self.encoder_level1)
        self.encoder_level2 = nn.Sequential(*self.encoder_level2)
        self.encoder_level3 = nn.Sequential(*self.encoder_level3)
        self.encoder_level4 = nn.Sequential(*self.encoder_level4)

        self.down12  = DownSample(n_feat, n_feat*2**1)
        self.down23  = DownSample(n_feat*2**1, n_feat*2**2)
        self.down34 =DownSample(n_feat*2**2, n_feat*2**3)

   
    def forward(self, input, scratch_x, mask):

        mix1 = input+scratch_x[0]
        list1 = [mix1, mask[0]]
        enc1 = self.encoder_level1(list1)
        x = self.down12(enc1[0])
        mix2 = x+scratch_x[1]
        list2 = [mix2, mask[1]]
        enc2 = self.encoder_level2(list2)
        x = self.down23(enc2[0])
        mix3 = x+scratch_x[2]
        list3 = [mix3, mask[2]]
        enc3 = self.encoder_level3(list3)
        x=self.down34(enc3[0])
        mix4 = x+scratch_x[3]
        list4 = [mix4, mask[3]]
        enc4 = self.encoder_level4(list4)
        
        # mix1 = input + scratch_x[0]
        # enc1 = self.encoder_level1(mix1, mask[0])
        # x = self.down12(enc1)
        # mix2 = x + scratch_x[1]
        # enc2 = self.encoder_level2(mix2, mask[1])
        # x = self.down23(enc2)
        # mix3 = x + scratch_x[2]
        # enc3 = self.encoder_level3(mix3, mask[2])
        # x = self.down34(enc3)
        # mix4 = x + scratch_x[3]
        # enc4 = self.encoder_level4(mix4, mask[3])
        # x_cat = torch.cat([input,scratch_x[0]], 1)  #concat
        # mix1 = self.conv1(x_cat)
        # enc1 = self.encoder_level1(mix1, mask[0])
        # x = self.down12(enc1)
        # x_cat = torch.cat([x,scratch_x[1]], 1)  #concat
        # mix2 = self.conv2(x_cat)
        # enc2 = self.encoder_level2(mix2, mask[1])
        # x = self.down23(enc2)
       
        # x_cat = torch.cat([x,scratch_x[2]], 1)  #concat
        # mix3 = self.conv3(x_cat)
        # enc3 = self.encoder_level3(mix3, mask[2])
        # x = self.down34(enc3)
       
        # x_cat = torch.cat([x,scratch_x[3]], 1)  #concat
        # mix4 = self.conv4(x_cat)
        # enc4 = self.encoder_level4(mix4, mask[3])
        return [enc1[0], enc2[0], enc3[0], enc4[0]]

class Decoder(nn.Module):
    def __init__(self, n_feat, scale_unetfeats, num_heads, ffn_expansion_factor, bias, LayerNorm_type='WithBias'):
        super(Decoder, self).__init__()

        self.decoder_level1 = [channel_transformer_decoder(n_feat, num_heads[3], ffn_expansion_factor = ffn_expansion_factor, bias=bias, LayerNorm_type='WithBias') for _ in range(2)]
        self.decoder_level2 = [channel_transformer_decoder(n_feat*2**1, num_heads[2], ffn_expansion_factor = ffn_expansion_factor, bias=bias, LayerNorm_type='WithBias') for _ in range(2)]
        self.decoder_level3 = [channel_transformer_decoder(n_feat*2**2, num_heads[1], ffn_expansion_factor = ffn_expansion_factor, bias=bias, LayerNorm_type='WithBias') for _ in range(2)]
        self.decoder_level4 = [channel_transformer_decoder(n_feat*2**3, num_heads[0], ffn_expansion_factor = ffn_expansion_factor, bias=bias, LayerNorm_type='WithBias') for _ in range(2)]

        self.decoder_level1 = nn.Sequential(*self.decoder_level1)
        self.decoder_level2 = nn.Sequential(*self.decoder_level2)
        self.decoder_level3 = nn.Sequential(*self.decoder_level3)
        self.decoder_level4 = nn.Sequential(*self.decoder_level4)

        self.up21  = SkipUpSample(n_feat, n_feat*2**1, kernel_size=3, bias=bias)
        self.up32  = SkipUpSample(n_feat*2**1, n_feat*2**2, kernel_size=3, bias=bias)
        self.up43  = SkipUpSample(n_feat*2**2, n_feat*2**3, kernel_size=3, bias=bias)

    def forward(self, outs, mask):
        enc1, enc2, enc3, enc4 = outs
        list4 = [enc4, mask[3]]
        dec4 = self.decoder_level4(list4)
        x_43 = self.up43(dec4[0], enc3)
        list3 = [x_43, mask[2]]
        dec3 = self.decoder_level3(list3)
        x_32 = self.up32(dec3[0], enc2)
        list2 = [x_32, mask[1]]
        dec2 = self.decoder_level2(list2)
        x_21 = self.up21(dec2[0], enc1)
        list1 = [x_21, mask[0]]
        dec1 = self.decoder_level1(list1)
        return [dec1[0], dec2[0], dec3[0], dec4[0]]



class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=False, innorm=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == 'same-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])

        elif self.innorm:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        elif self.outer:
            out = self.conv(out)
            out[0] = self.bn(out[0])
        else:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            if hasattr(self, 'activation'):
                out[0] = self.activation(out[0])
        return out

class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)
        torch.nn.init.constant_(self.mask_conv.weight, 1.0)
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, inputt):

        input = inputt[0]
        mask = inputt[1]
        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)
        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.byte(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.byte(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.byte(), 0.0)
        out = []
        out.append(output)
        out.append(new_mask)

        return out

class PCconv(nn.Module):
    def __init__(self, channels, kernel_size,  bias=True):
        super(PCconv, self).__init__()

        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(3):
            seuqence_3 += [PCBActiv(channels, channels, innorm=True)]
            seuqence_5 += [PCBActiv(channels, channels, sample='same-5', innorm=True)]
            seuqence_7 += [PCBActiv(channels, channels, sample='same-7', innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)

        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.conv6 = conv(channels*3, channels, kernel_size, bias=bias)

    def forward(self, input, mask):

        mask = torch.add(torch.neg(mask.float()), 1)
        x_1 = [input, mask]
        # Multi Scale PConv fill the Details
        x_DE_3 = self.cov_3(x_1)
        x_DE_5 = self.cov_5(x_1)
        x_DE_7 = self.cov_7(x_1)
        x_DE_fuse = torch.cat([x_DE_3[0], x_DE_5[0], x_DE_7[0]], 1)
        out = self.conv6(x_DE_fuse)+input

        return out

class scratch_net(nn.Module):
    def __init__(self, channels, kernel_size,  bias=True):
        super(scratch_net, self).__init__()
        self.PCconv_1 = PCconv(channels, kernel_size,  bias=bias)
        self.PCconv_2 = PCconv(channels*2, kernel_size, bias=bias)
        self.PCconv_3 =  PCconv(channels*2**2, kernel_size, bias=bias)
        self.PCconv_4 = PCconv(channels*2**3, kernel_size, bias=bias)

        self.down12 = DownSample(channels, channels * 2 ** 1)
        self.down23 = DownSample(channels * 2 ** 1, channels * 2 ** 2)
        self.down34 = DownSample(channels * 2 ** 2, channels * 2 ** 3)

    def forward(self, input, mask):
        x1 = self.PCconv_1(input, mask[0])
        x1_1 = self.down12(x1)
        x2 = self.PCconv_2(x1_1, mask[1])
        x2_2 = self.down23(x2)
        x3 = self.PCconv_3(x2_2, mask[2])
        x3_3 = self.down34(x3)
        x4 = self.PCconv_4(x3_3, mask[3])

        return [x1, x2, x3, x4]

##########################################################################
class DownSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(DownSample, self).__init__()
        self.down = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(in_channels, s_factor, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.down(x)
        return x

class UpSample(nn.Module):
    def __init__(self, in_channels,s_factor):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(s_factor, in_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x

class SkipUpSample(nn.Module):
    def __init__(self, in_channels, s_factor, kernel_size, bias):
        super(SkipUpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(s_factor, in_channels, 1, stride=1, padding=0, bias=False))
        self.conv = conv(in_channels*2, in_channels,kernel_size, bias=bias)
    def forward(self, x,y):
        x = self.up(x)
    #    x_cat = torch.cat([x,y], 1)  #concat
    #    x_cat = self.conv(x_cat)
        x = x + y
        return x


########################################################################
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal(m.weight.data, 1.0, gain)
            init.constant(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
      #  net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net

def define_D(input_nc= 6, ndf=64, n_layers_D=3, norm='batch', init_type='normal', gpu_ids=[], init_gain=0.02):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)
    netD = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=False)
    return init_net(netD, init_type, init_gain, gpu_ids)

def define_G(in_c=3, out_c=3, n_feat=40, scale_unetfeats=20, scale_orsnetfeats=16, num_cab=8, kernel_size=3, reduction=4,bias=False, norm='batch',init_type='normal',init_gain=0.02,gpu_ids=[]):

    norm_layer = get_norm_layer(norm_type=norm)
    MPR = MPRNet(in_c, out_c, n_feat, scale_unetfeats, scale_orsnetfeats, num_cab, kernel_size, reduction,bias)
    return init_net(MPR, init_type, init_gain, gpu_ids)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.relu(x1) * x2
        x = self.project_out(x)
        return x
###########################################################################
class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.mask = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.k_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, mask):
        b, c, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        m = mask
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        m = rearrange(m, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q+m) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out)
        return out
##########################################################################
class channel_transformer_encoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(channel_transformer_encoder, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, list):
        x = list[0]
        mask = list[1]
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return [x, mask]

class channel_transformer_decoder(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(channel_transformer_decoder, self).__init__()
        self.MAIN = MAIN(dim, eps=1e-5)
        self.attn = Attention(dim, num_heads, bias)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
    def forward(self, list):
        x = list[0]
        mask = list[1]
        x = x + self.attn(self.MAIN(x, mask), mask)
        x = x + self.ffn(self.MAIN(x, mask))
        return [x, mask]

#############################################################################
class Old_Photo_Net(nn.Module):
    def __init__(self, n_feat=48, scale_unetfeats=48, scale_orsnetfeats=16, kernel_size=3,reduction=4,num_heads = [1,2,4,8],bias=True, ffn_expansion_factor=1, LayerNorm_type='WithBias'):
        super(Old_Photo_Net, self).__init__()
        act=nn.PReLU()
        self.shallow_feat = shallow_feat(n_feat, kernel_size, reduction, bias=bias, act=act)
        self.scratch_net = scratch_net(n_feat, kernel_size, bias)
        self.encoder = Encoder(n_feat, scale_unetfeats, num_heads, ffn_expansion_factor, bias, kernel_size, LayerNorm_type)
        self.decoder = Decoder(n_feat, scale_unetfeats, num_heads, ffn_expansion_factor, bias, LayerNorm_type)
        self.conv1 = conv(48, 3, kernel_size, bias=bias)
        self.conv2 = conv(6, 3, kernel_size, bias=bias)
        self.concat  = conv(80, 40, kernel_size, bias=bias)
        
    def forward(self, img, mask, mask_patch, mask_scratch):

        x_scratch = torch.cat([img * (1 - mask_patch), mask_scratch], 1)
        x_scratch = self.conv2(x_scratch)
        x_scratch = self.shallow_feat(x_scratch)
        x = self.shallow_feat(img * (1 - mask))

        mask_list = []
        mask_1 = mask.squeeze(0).resize_(256, 256)
        mask_1 = mask_1.repeat(1, 48, 1, 1)
        mask_list.append(mask_1)

        mask_2 = mask.squeeze(0).resize_(128, 128)
        mask_2 = mask_2.repeat(1, 96, 1, 1)
        mask_list.append(mask_2)

        mask_3 = mask.squeeze(0).resize_(64, 64)
        mask_3 = mask_3.repeat(1, 192, 1, 1)
        mask_list.append(mask_3)

        mask_4 = mask.squeeze(0).resize_(32, 32)
        mask_4 = mask_4.repeat(1, 384, 1, 1)
        mask_list.append(mask_4)


        mask_patch_list = []
        mask_1 = mask_patch.squeeze(0).resize_(256, 256)
        mask_1 = mask_1.repeat(1, 48, 1, 1)
        mask_patch_list.append(mask_1)

        mask_2 = mask_patch.squeeze(0).resize_(128, 128)
        mask_2 = mask_2.repeat(1, 96, 1, 1)
        mask_patch_list.append(mask_2)

        mask_3 = mask_patch.squeeze(0).resize_(64, 64)
        mask_3 = mask_3.repeat(1, 192, 1, 1)
        mask_patch_list.append(mask_3)

        mask_4 = mask_patch.squeeze(0).resize_(32, 32)
        mask_4 = mask_4.repeat(1, 384, 1, 1)
        mask_patch_list.append(mask_4)

        mask_scratch_list = []
        mask_1 = mask_scratch.squeeze(0).resize_(256, 256)
        mask_1 = mask_1.repeat(1, 48, 1, 1)
        mask_scratch_list.append(mask_1)

        mask_2 = mask_scratch.squeeze(0).resize_(128, 128)
        mask_2 = mask_2.repeat(1, 96, 1, 1)
        mask_scratch_list.append(mask_2)

        mask_3 = mask_scratch.squeeze(0).resize_(64, 64)
        mask_3 = mask_3.repeat(1, 192, 1, 1)
        mask_scratch_list.append(mask_3)

        mask_4 = mask_scratch.squeeze(0).resize_(32, 32)
        mask_4 = mask_4.repeat(1, 384, 1, 1)
        mask_scratch_list.append(mask_4)

        x_scratch = self.scratch_net(x_scratch, mask_scratch_list)
        feat1 = self.encoder(x, x_scratch, mask_patch_list)
        res1 = self.decoder(feat1, mask_list)
        result = self.conv1(res1[0])
        result = result + img * (1 - mask)

        return [result]


