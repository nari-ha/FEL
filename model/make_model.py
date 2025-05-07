import torch
import torch.nn as nn
import numpy as np

from model.fel import BiAttentionBlock
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from pdb import set_trace as bp

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num
        self.sie_coe = cfg.MODEL.SIE_COE
        
        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.classifier_proj = nn.Linear(self.in_planes_proj, self.num_classes, bias=False)
        self.classifier_proj.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        self.bottleneck_proj = nn.BatchNorm1d(self.in_planes_proj)
        self.bottleneck_proj.bias.requires_grad_(False)
        self.bottleneck_proj.apply(weights_init_kaiming)

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        self.clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        self.clip_model.to("cuda")

        self.image_encoder = self.clip_model.visual
        self.feature_enhancer_layer = BiAttentionBlock(
                v_dim=self.in_planes_proj,
                l_dim=self.in_planes_proj,
                embed_dim=self.in_planes_proj // 2,
                num_heads=8//2,
                dropout=0.1,
                drop_path=0.0,
        )
        
        self.dataset_name = cfg.DATASETS.NAMES
        self.eval_name = cfg.DATASETS.EVAL

        # if cfg.MODEL.SIE_CAMERA and cfg.MODEL.SIE_VIEW:
        #     self.cv_embed = nn.Parameter(torch.zeros(camera_num * view_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(camera_num))
        # elif cfg.MODEL.SIE_CAMERA:
        #     self.cv_embed = nn.Parameter(torch.zeros(camera_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(camera_num))
        # elif cfg.MODEL.SIE_VIEW:
        #     self.cv_embed = nn.Parameter(torch.zeros(view_num, self.in_planes))
        #     trunc_normal_(self.cv_embed, std=.02)
        #     print('camera number is : {}'.format(view_num))

    def forward(self, x, label=None, cam_label= None, view_label=None):
        if self.model_name == 'RN50':
            image_features_last, image_features, image_features_proj = self.image_encoder(x) #B,512  B,128,512
            img_feature_last = nn.functional.avg_pool2d(image_features_last, image_features_last.shape[2:4]).view(x.shape[0], -1) 
            img_feature = nn.functional.avg_pool2d(image_features, image_features.shape[2:4]).view(x.shape[0], -1) 
            img_feature_proj = image_features_proj[0]
            
        if self.eval_name == "veri":
            text = "A photo of a vehicle."
        else:
            text = "A photo of a person."
        tokens = _tokenizer.encode(text)
        padded_tokens = tokens + [0] * (77 - len(tokens))
        text = torch.tensor([padded_tokens]).cuda()
        text_features = self.clip_model.encode_text(text)
        text_features = text_features.repeat(img_feature_proj.size()[0], 1)
        img_feature_proj = img_feature_proj.unsqueeze(1) # [B, D]
        text_features = text_features.unsqueeze(1)
        img_feature_proj, text_features = self.feature_enhancer_layer(v=img_feature_proj, l=text_features, attention_mask_v=None, attention_mask_l=None)
        img_feature_proj = img_feature_proj.squeeze(1)  # [B, D]
        text_features = text_features.squeeze(1)
        bp()
        
        assert torch.all(text_features == text_features[0]), "모든 행이 동일하지 않습니다."
        text_features = text_features[0].unsqueeze(0)
        text_features = text_features.expand(self.num_classes, -1)
        
        bp()

        feat = self.bottleneck(img_feature) 
        feat_proj = self.bottleneck_proj(img_feature_proj) 

        if self.training:
            cls_score = self.classifier(feat)
            cls_score_proj = self.classifier_proj(feat_proj)
            return [cls_score, cls_score_proj], [img_feature_last, img_feature, img_feature_proj], img_feature_proj, text_features

        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return torch.cat([feat, feat_proj], dim=1)
            else:
                return torch.cat([img_feature, img_feature_proj], dim=1)


    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model


from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model