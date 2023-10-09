import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       nn.BatchNorm2d(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       nn.BatchNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class FeatExtractor(nn.Module):
    def __init__(self, base_chs, c_in=3, num_stage=3, use_dropout=False, use_bias=False):
        super(FeatExtractor, self).__init__()
        self.base_chs = base_chs
        self.num_stage = num_stage

        # cin->base_chs
        self.conv0 = nn.Sequential(
            nn.Conv2d(c_in, base_chs, kernel_size=7, padding=3, bias=use_bias),
            nn.BatchNorm2d(base_chs),
            nn.ReLU(True)
        )
        self.conv1 = ResnetBlock(base_chs, 'zero', use_dropout, use_bias)
        self.conv2 = ResnetBlock(base_chs, 'zero', use_dropout, use_bias)

        # base_chs->base_chs*2
        self.conv3 = nn.Sequential(
            nn.Conv2d(base_chs, base_chs*2, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(base_chs*2),
            nn.ReLU(True)
        )
        self.conv4 = ResnetBlock(base_chs*2, 'zero', use_dropout, use_bias)
        self.conv5 = ResnetBlock(base_chs*2, 'zero', use_dropout, use_bias)

        # base_chs*2->base_chs*4
        self.conv6 = nn.Sequential(
            nn.Conv2d(base_chs * 2, base_chs * 4, kernel_size=3, stride=2, padding=1, bias=use_bias),
            nn.BatchNorm2d(base_chs*4),
            nn.ReLU(True)
        )
        self.conv7 = ResnetBlock(base_chs*4, 'zero', use_dropout, use_bias)
        self.conv8 = ResnetBlock(base_chs*4, 'zero', use_dropout, use_bias)

        self.out1 = nn.Conv2d(base_chs*4, base_chs*4, 1, bias=False)
        self.out_channels = [base_chs*4]

        # base_chs*4->base_chs*2
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(base_chs*4, base_chs*2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(base_chs*2),
            nn.ReLU(True)
        )
        self.deconv2 = ResnetBlock(base_chs*2, 'zero', use_dropout, use_bias)
        self.deconv3 = ResnetBlock(base_chs*2, 'zero', use_dropout, use_bias)
        self.out2 = nn.Conv2d(base_chs*2, base_chs*2, 1, bias=False)
        self.out_channels.append(2 * base_chs)

        # base_chs*2->base_chs
        self.deconv4 = nn.Sequential(
            nn.ConvTranspose2d(base_chs*2, base_chs, kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
            nn.BatchNorm2d(base_chs),
            nn.ReLU(True)
        )
        self.deconv5 = ResnetBlock(base_chs, 'zero', use_dropout, use_bias)
        self.deconv6 = ResnetBlock(base_chs, 'zero', use_dropout, use_bias)
        self.out3 = nn.Conv2d(base_chs, base_chs, 1, bias=False)
        self.out_channels.append(base_chs)


    def forward(self, x):
        out = self.conv0(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)

        outputs = {}
        out1 = self.out1(out)
        outputs['stage1'] = out1

        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.deconv3(out)

        out2 = self.out2(out)
        outputs['stage2'] = out2

        out = self.deconv4(out)
        out = self.deconv5(out)
        out = self.deconv6(out)

        out3 = self.out3(out)
        outputs['stage3'] = out3
        return outputs, out



class NENet(nn.Module):
    def __init__(self, base_chs, fuse_type='max', c_in=3, use_dropout=False, use_bias=False):
        super(NENet, self).__init__()
        self.extractor = FeatExtractor(base_chs=base_chs, c_in=c_in, use_bias=use_bias)
        self.c_in = c_in
        self.fuse_type = fuse_type
        self.regressor = nn.Sequential(
            ResnetBlock(base_chs, 'zero', use_dropout, use_bias),
            ResnetBlock(base_chs, 'zero', use_dropout, use_bias),
            nn.Conv2d(base_chs, 3, kernel_size=7, padding=3)  # normal est
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def prepareInputs(self, x):
        imgs = x['imgs']
        b, v, l, c, h, w = imgs.shape
        single_view_imgs = torch.unbind(imgs, 1)  # (b, L, 3, h, w) along nviews dim
        single_view_dirs = torch.unbind(x['light_dirs'], 1)  # (b, l, 3) along nviews dim
        view_inputs = []
        for v in range(len(single_view_imgs)):  # for each view
            single_light_imgs = torch.unbind(single_view_imgs[v], 1)  # (b, 3, h, w) along l dim
            single_light_dirs = torch.unbind(single_view_dirs[v], 1)  # (b, 3) along l dim
            assert len(single_light_imgs) == len(single_light_dirs)  # l
            single_view_inputs = []
            for l in range(len(single_light_imgs)):  # for each light
                l_dir = single_light_dirs[l] if single_light_dirs[l].dim() == 4 else single_light_dirs[l].view(b, -1, 1, 1)  # (b, 3, 1, 1)
                img = single_light_imgs[l]  # (b, 3, h, w)
                img_light = torch.cat([img, l_dir.expand_as(img)], 1)  # (b,6,h,w)
                single_view_inputs.append(img_light)
            del single_light_imgs, single_light_dirs
            view_inputs.append(single_view_inputs)
        del single_view_imgs, single_view_dirs
        return view_inputs

    def forward(self, x):
        view_inputs = self.prepareInputs(x)
        view_feats_stage1 = []
        view_feats_stage2 = []
        view_feats_stage3 = []
        view_normals = []
        for v in range(len(view_inputs)):
            inputs = view_inputs[v]
            feats_stage1 = torch.Tensor()
            feats_stage2 = torch.Tensor()
            feats_stage3 = torch.Tensor()
            feats = torch.Tensor()
            for i in range(len(inputs)):
                outputs_dict, out = self.extractor(inputs[i])
                if i == 0:
                    feats_stage1 = outputs_dict['stage1']
                    feats_stage2 = outputs_dict['stage2']
                    feats_stage3 = outputs_dict['stage3']
                    feats = out
                else:
                    if self.fuse_type == 'mean':
                        feats_stage1 = torch.stack([feats_stage1, outputs_dict['stage1']], 1).sum(1)
                        feats_stage2 = torch.stack([feats_stage2, outputs_dict['stage2']], 1).sum(1)
                        feats_stage3 = torch.stack([feats_stage3, outputs_dict['stage3']], 1).sum(1)
                        feats = torch.stack([feats, out], 1).sum(1)
                    elif self.fuse_type == 'max':
                        feats_stage1, _ = torch.stack([feats_stage1, outputs_dict['stage1']], 1).max(1)
                        feats_stage2, _ = torch.stack([feats_stage2, outputs_dict['stage2']], 1).max(1)
                        feats_stage3, _ = torch.stack([feats_stage3, outputs_dict['stage3']], 1).max(1)
                        feats, _ = torch.stack([feats, out], 1).max(1)
            if self.fuse_type == 'mean':
                feats_stage1 = feats_stage1 / len(inputs)
                feats_stage2 = feats_stage2 / len(inputs)
                feats_stage3 = feats_stage3 / len(inputs)
                feats = feats / len(inputs)
            view_feats_stage1.append(feats_stage1)
            view_feats_stage2.append(feats_stage2)
            view_feats_stage3.append(feats_stage3)

            normal = self.regressor(feats)
            normal = torch.nn.functional.normalize(normal, 2, 1)
            view_normals.append(normal)
            del feats
        view_normals = torch.stack(view_normals, 1)  # (b, v, 3, h, w)
        view_feats = {
            'stage1': view_feats_stage1,
            'stage2': view_feats_stage2,
            'stage3': view_feats_stage3
        }
        # view_normals: (b, v, 3, h, w)
        return view_feats, view_normals  # dict and tensor
