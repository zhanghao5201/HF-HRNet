import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import (ConvModule, DepthwiseSeparableConvModule,
                      build_conv_layer, build_norm_layer, constant_init,
                      normal_init)
from torch.nn.modules.batchnorm import _BatchNorm

from mmpose.utils import get_root_logger
from ..builder import BACKBONES
from .utils import load_checkpoint
import math

class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 num_kernels=1):
        super().__init__()

        self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, groups=1,
                 num_kernels=1):
        super().__init__()

        self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class CrossResolutionWeighting(nn.Module):
    """Cross-resolution channel weighting module.

    Args:
        channels (int): The channels of the module.
        ratio (int): channel reduction ratio.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        act_cfg (dict): Config dict for activation layer.
            Default: (dict(type='ReLU'), dict(type='Sigmoid')).
            The last ConvModule uses Sigmoid by default.
    """

    def __init__(self,
                 channels,
                 ratio=16,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=(dict(type='ReLU'), dict(type='Sigmoid'))):
        super().__init__()
        if isinstance(act_cfg, dict):
            act_cfg = (act_cfg, act_cfg)
        assert len(act_cfg) == 2
        assert mmcv.is_tuple_of(act_cfg, dict)
        self.channels = channels
        total_channel = sum(channels)
        #print("hojfdio",ratio)#8
        self.conv1 = ConvModule(
            in_channels=total_channel,
            out_channels=int(total_channel / ratio),
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[0])
        self.conv2 = ConvModule(
            in_channels=int(total_channel / ratio),
            out_channels=total_channel,
            kernel_size=1,
            stride=1,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg[1])

    def forward(self, x):
        mini_size = x[-1].size()[-2:]
        mini_size = [int(v) for v in mini_size]
        out = [F.adaptive_avg_pool2d(s, mini_size) for s in x[:-1]] + [x[-1]]
        out = torch.cat(out, dim=1)
        out = self.conv1(out)
        out = self.conv2(out)
        out = torch.split(out, self.channels, dim=1)
        out = [
            s * F.interpolate(a, size=s.size()[-2:], mode='nearest')
            for s, a in zip(x, out)
        ]
        return out


class MASBlock(nn.Module):
    """Conditional channel weighting block.

    Args:
        in_channels (int): The input channels of the block.
        stride (int): Stride of the 3x3 convolution layer.
        reduce_ratio (int): channel reduction ratio.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels,
                 stride,
                 reduce_ratio,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super().__init__()
        self.with_cp = with_cp
        self.stride = stride
        assert stride in [1, 2]

        branch_channels = [channel  for channel in in_channels]
               
        self.cross_resolution_weighting = CrossResolutionWeighting(
            branch_channels,
            ratio=reduce_ratio,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg)
            
                   
        self.depthwise_convsfirst = nn.ModuleList([ConvBNReLU(in_channels=channel, out_channels=channel, 
        kernel_size=3, stride=1, padding=1, dilation=1, groups=channel, num_kernels=channel )for channel in branch_channels
        ])
        self.global1=nn.ModuleList([
        nn.Sequential(
                nn.AvgPool2d(3, stride=(2,2),padding=1), 
                nn.Conv2d(channel, channel, kernel_size=1, bias=False) ,
            ) for channel in branch_channels])
        
              
        self.depthwise_convslast = nn.ModuleList([ConvBN(in_channels=channel, out_channels=channel, kernel_size=3, stride=self.stride, padding=1, 
        dilation=1, groups=channel, num_kernels=channel )for channel in branch_channels
        ])
        self.global2=nn.ModuleList([
        nn.Sequential(
                nn.AvgPool2d(3, stride=(2,2),padding=1), 
                nn.Conv2d(channel, channel, kernel_size=1, bias=False) ,
            ) for channel in branch_channels])
        
       
        self.up1=nn.ModuleList([nn.Upsample(scale_factor=(2,2), mode='nearest') for channel in branch_channels]) 
        self.up2=nn.ModuleList([nn.Upsample(scale_factor=(2,2), mode='nearest') for channel in branch_channels])  
        
        

    def forward(self, x):

        def _inner_forward(x):            
            
            x1 = self.cross_resolution_weighting(x) 
            #print("k0",x1[0].shape,len(x1))  
            x1 = [dw(s) for s,dw in zip(x1,self.depthwise_convsfirst)]            
            
            global1 = [ga(s) for s,ga in zip(x1,self.global1)]
            
            
            global1=[s(ga) for s,ga in zip(self.up1,global1)] 
            
            out2 = [s2+s3 for s2,s3 in zip(global1,x1)] 
                       
            x1 = [dw(s) for s, dw in zip(out2, self.depthwise_convslast)]
            
            global2 = [ga(s) for s,ga in zip(x1,self.global2)]
            
            
            global2=[s(ga) for s,ga in zip(self.up2,global2)] 
            
            out2 = [s2+s3 for s2,s3 in zip(global2,x1)] 
            
            out2 = [s1+s2 for s1, s2 in zip(out2,x)] 
            return out2


        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ASBlock(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, keep_3x3=False):
        super(ASBlock, self).__init__()
        assert stride in [1, 2]
        
        self.maps1=nn.Parameter(torch.ones(1,oup,1,1),requires_grad=True)
        
        self.shift_dim1=(oup//4,oup-(oup//4))
        self.shift_dim2=(oup//3,oup-(oup//3))        
        
        self.maps3=nn.Parameter(torch.ones(1,oup,1,1),requires_grad=True)        
        
        
        self.dwonsample = nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride=2, padding=1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.Conv2d(inp, oup, 1, bias=False),
                nn.BatchNorm2d(oup),
            )
        
        self.conv1=nn.Sequential(
                # dw
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))  
                
        self.conv2 = nn.Sequential(
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True))
        
        self.global1=nn.Sequential(
                nn.AvgPool2d(3, stride=(2,2),padding=1),   
                nn.Conv2d(oup, oup, kernel_size=1,groups=1, bias=False) ,
                nn.ReLU(inplace=True)) 
        
        self.global2=nn.Sequential(
                nn.AvgPool2d(3, stride=(2,2),padding=1),   
                nn.Conv2d(oup, oup, kernel_size=1,groups=1, bias=False) ,
                nn.ReLU(inplace=True)) 
        
        
        self.conv3 = nn.Sequential(
                # dw
                nn.Conv2d(oup, oup, 3, stride=2, padding=1, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
                )
        self.up1=nn.Upsample(scale_factor=(2,2), mode='nearest')   
        self.up2=nn.Upsample(scale_factor=(2,2), mode='nearest')             
    def forward(self, x):
        out = self.conv1(x)
        xdown=self.dwonsample(x)
        
        out=self.conv2(out) 
        #print(out.shape)
        global1=self.global1(out)  #torch.Size([1, 48, 128, 96])
               
        global1=self.up1(global1)
        out=global1+out
        #print("out",kd)
        
        out=self.conv3(out)
        global2=self.global2(out)  #torch.Size([1, 48, 128, 96])
       
        global2=self.up2(global2) 
        out=global2+out+ xdown
        return out
def _upsample( x, y):
        _,_,h,w = y.size()
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)     
        
class Stem(nn.Module):
    """Stem network block.

    Args:
        in_channels (int): The input channels of the block.
        stem_channels (int): Output channels of the stem layer.
        out_channels (int): The output channels of the block.
        expand_ratio (int): adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 in_channels,
                 stem_channels,
                 out_channels,
                 expand_ratio,#1stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.with_cp = with_cp

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=stem_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=dict(type='ReLU'))
        self.conv2=ASBlock(stem_channels,out_channels,2,2)
       
    def forward(self, x):

        def _inner_forward(x):
            x = self.conv1(x)
            out=self.conv2(x)
            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)
        #print("dossoifoi",out.shape)#torch.Size([1, 32, 64, 64])
        return out


class IterativeHead(nn.Module):
    """Extra iterative head for feature learning.

    Args:
        in_channels (int): The input channels of the block.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
    """

    def __init__(self, in_channels, norm_cfg=dict(type='BN')):
        super().__init__()
        projects = []
        num_branchs = len(in_channels)
        self.in_channels = in_channels[::-1]

        for i in range(num_branchs):
            if i != num_branchs - 1:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i + 1],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
            else:
                projects.append(
                    DepthwiseSeparableConvModule(
                        in_channels=self.in_channels[i],
                        out_channels=self.in_channels[i],
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        norm_cfg=norm_cfg,
                        act_cfg=dict(type='ReLU'),
                        dw_act_cfg=None,
                        pw_act_cfg=dict(type='ReLU')))
        self.projects = nn.ModuleList(projects)

    def forward(self, x):
        x = x[::-1]

        y = []
        last_x = None
        for i, s in enumerate(x):
            if last_x is not None:
                last_x = F.interpolate(
                    last_x,
                    size=s.size()[-2:],
                    mode='bilinear',
                    align_corners=True)
                s = s + last_x
            s = self.projects[i](s)
            y.append(s)
            last_x = s

        return y[::-1]

class HF_HRModule(nn.Module):
    """High-Resolution Module for LiteHRNet.

    It contains conditional channel weighting blocks and
    shuffle blocks.


    Args:
        num_branches (int): Number of branches in the module.
        num_blocks (int): Number of blocks in the module.
        in_channels (list(int)): Number of input image channels.
        reduce_ratio (int): Channel reduction ratio.
        module_type (str): 'LITE' or 'NAIVE'
        multiscale_output (bool): Whether to output multi-scale features.
        with_fuse (bool): Whether to use fuse layers.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.
    """

    def __init__(
            self,
            num_branches,
            num_blocks,
            in_channels,
            reduce_ratio,            
            multiscale_output=False,
            with_fuse=True,
            conv_cfg=None,
            norm_cfg=dict(type='BN'),
            with_cp=False,
    ):
        super().__init__()
        self._check_branches(num_branches, in_channels)

        self.in_channels = in_channels
        self.num_branches = num_branches
        
        self.multiscale_output = multiscale_output
        self.with_fuse = with_fuse
        self.norm_cfg = norm_cfg
        self.conv_cfg = conv_cfg
        self.with_cp = with_cp
       
        self.layers = self._make_weighting_blocks(num_blocks, reduce_ratio)
       
        if self.with_fuse:
            self.fuse_layers = self._make_fuse_layers()
            self.relu = nn.ReLU()

    def _check_branches(self, num_branches, in_channels):
        """Check input to avoid ValueError."""
        if num_branches != len(in_channels):
            error_msg = f'NUM_BRANCHES({num_branches}) ' \
                f'!= NUM_INCHANNELS({len(in_channels)})'
            raise ValueError(error_msg)

    def _make_weighting_blocks(self, num_blocks, reduce_ratio, stride=1):
        """Make channel weighting blocks."""
        layers = []
        for i in range(num_blocks):
            layers.append(
                MASBlock(
                    self.in_channels,
                    stride=stride,
                    reduce_ratio=reduce_ratio,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))

        return nn.Sequential(*layers)    

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.in_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1
        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels[j],
                                in_channels[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels[i])[1],
                            nn.Upsample(
                                scale_factor=2**(j - i), mode='nearest')))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[i],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[i])[1]))
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        groups=in_channels[j],
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    build_conv_layer(
                                        self.conv_cfg,
                                        in_channels[j],
                                        in_channels[j],
                                        kernel_size=1,
                                        stride=1,
                                        padding=0,
                                        bias=False),
                                    build_norm_layer(self.norm_cfg,
                                                     in_channels[j])[1],
                                    nn.ReLU(inplace=True)))
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.layers[0](x[0])]

        out = self.layers(x)
        

        if self.with_fuse:
            out_fuse = []
            for i in range(len(self.fuse_layers)):
                # `y = 0` will lead to decreased accuracy (0.5~1 mAP)
                y = out[0] if i == 0 else self.fuse_layers[i][0](out[0])
                for j in range(self.num_branches):
                    if i == j:
                        y += out[j]
                    else:
                        y += self.fuse_layers[i][j](out[j])
                out_fuse.append(self.relu(y))
            
            out = out_fuse
        #print("jsk",len(out),out[0].shape)
        if not self.multiscale_output:
            out = [out[0]]
        return out


@BACKBONES.register_module()
class HF_HRNet(nn.Module):
    """Lite-HRNet backbone.

    `Lite-HRNet: A Lightweight High-Resolution Network
    <https://arxiv.org/abs/2104.06403>`_.

    Code adapted from 'https://github.com/HRNet/Lite-HRNet'.

    Args:
        extra (dict): detailed configuration for each stage of HRNet.
        in_channels (int): Number of input image channels. Default: 3.
        conv_cfg (dict): dictionary to construct and config conv layer.
        norm_cfg (dict): dictionary to construct and config norm layer.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed.

    Example:
        >>> from mmpose.models import LiteHRNet
        >>> import torch
        >>> extra=dict(
        >>>    stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
        >>>    num_stages=3,
        >>>    stages_spec=dict(
        >>>        num_modules=(2, 4, 2),
        >>>        num_branches=(2, 3, 4),
        >>>        num_blocks=(2, 2, 2),
        >>>        module_type=('LITE', 'LITE', 'LITE'),
        >>>        with_fuse=(True, True, True),
        >>>        reduce_ratios=(8, 8, 8),
        >>>        num_channels=(
        >>>            (40, 80),
        >>>            (40, 80, 160),
        >>>            (40, 80, 160, 320),
        >>>        )),
        >>>    with_head=False)
        >>> self = LiteHRNet(extra, in_channels=1)
        >>> self.eval()
        >>> inputs = torch.rand(1, 1, 32, 32)
        >>> level_outputs = self.forward(inputs)
        >>> for level_out in level_outputs:
        ...     print(tuple(level_out.shape))
        (1, 40, 8, 8)
    """

    def __init__(self,
                 extra,
                 in_channels=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 norm_eval=False,
                 with_cp=False):
        super().__init__()
        self.extra = extra
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.stem = Stem(
            in_channels,
            stem_channels=self.extra['stem']['stem_channels'],
            out_channels=self.extra['stem']['out_channels'],
            expand_ratio=self.extra['stem']['expand_ratio'],
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg)

        self.num_stages = self.extra['num_stages']
        self.stages_spec = self.extra['stages_spec']

        num_channels_last = [
            self.stem.out_channels,
        ]
        for i in range(self.num_stages):
            num_channels = self.stages_spec['num_channels'][i]
            num_channels = [num_channels[i] for i in range(len(num_channels))]
            setattr(
                self, f'transition{i}',
                self._make_transition_layer(num_channels_last, num_channels))

            stage, num_channels_last = self._make_stage(
                self.stages_spec, i, num_channels, multiscale_output=True)
            setattr(self, f'stage{i}', stage)

        self.with_head = self.extra['with_head']
        if self.with_head:
            self.head_layer = IterativeHead(
                in_channels=num_channels_last,
                norm_cfg=self.norm_cfg,
            )

    def _make_transition_layer(self, num_channels_pre_layer,
                               num_channels_cur_layer):
        """Make transition layer."""
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_pre_layer[i],
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=num_channels_pre_layer[i],
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_pre_layer[i])[1],
                            build_conv_layer(
                                self.conv_cfg,
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg,
                                             num_channels_cur_layer[i])[1],
                            nn.ReLU()))
                else:
                    transition_layers.append(None)
            else:
                conv_downsamples = []
                for j in range(i + 1 - num_branches_pre):
                    in_channels = num_channels_pre_layer[-1]
                    out_channels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else in_channels
                    conv_downsamples.append(
                        nn.Sequential(
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                in_channels,
                                kernel_size=3,
                                stride=2,
                                padding=1,
                                groups=in_channels,
                                bias=False),
                            build_norm_layer(self.norm_cfg, in_channels)[1],
                            build_conv_layer(
                                self.conv_cfg,
                                in_channels,
                                out_channels,
                                kernel_size=1,
                                stride=1,
                                padding=0,
                                bias=False),
                            build_norm_layer(self.norm_cfg, out_channels)[1],
                            nn.ReLU()))
                transition_layers.append(nn.Sequential(*conv_downsamples))

        return nn.ModuleList(transition_layers)

    def _make_stage(self,
                    stages_spec,
                    stage_index,
                    in_channels,
                    multiscale_output=True):
        num_modules = stages_spec['num_modules'][stage_index]
        num_branches = stages_spec['num_branches'][stage_index]
        num_blocks = stages_spec['num_blocks'][stage_index]
        reduce_ratio = stages_spec['reduce_ratios'][stage_index]
        with_fuse = stages_spec['with_fuse'][stage_index]
        #module_type = stages_spec['module_type'][stage_index]

        modules = []
        for i in range(num_modules):
           
            # multi_scale_output is only used last module
            if not multiscale_output and i == num_modules - 1:
                reset_multiscale_output = False
            else:
                reset_multiscale_output = True
            #print(modules)
            modules.append(
                HF_HRModule(
                    num_branches,
                    num_blocks,
                    in_channels,
                    reduce_ratio,
                    #module_type,
                    multiscale_output=reset_multiscale_output,
                    with_fuse=with_fuse,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    with_cp=self.with_cp))
            in_channels = modules[-1].in_channels

        return nn.Sequential(*modules), in_channels

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.001)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        """Forward function."""
        #print("zhjh",x.shape)
        x = self.stem(x)

        y_list = [x]
        for i in range(self.num_stages):
            x_list = []
            transition = getattr(self, f'transition{i}')
            for j in range(self.stages_spec['num_branches'][i]):
                if transition[j]:
                    if j >= len(y_list):
                        x_list.append(transition[j](y_list[-1]))
                    else:
                        x_list.append(transition[j](y_list[j]))
                else:
                    x_list.append(y_list[j])
            y_list = getattr(self, f'stage{i}')(x_list)

        x = y_list
        if self.with_head:
            x = self.head_layer(x)
        #print("l",x[0].shape,x[1].shape,x[2].shape,x[3].shape)# torch.Size([1, 60, 64, 64]) torch.Size([1, 60, 32, 32]) torch.Size([1, 120, 16, 16]) torch.Size([1, 240, 8, 8])
        return [x[0]]

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()

