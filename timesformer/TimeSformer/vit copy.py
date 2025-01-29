# inspired  and immplimented bassed on https://github.com/facebookresearch/TimeSformer

import torch
import torch.nn as nn
from functools import partial
import math
import warnings
import torch.nn.functional as F
import numpy as np
from typing import List,Union

from torch import einsum
from einops import rearrange, reduce, repeat
from vit_utils import DropPath, trunc_normal_

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class MaskAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., with_qkv=True,mask=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.with_qkv = with_qkv
        if self.with_qkv:
           self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
           self.proj = nn.Linear(dim, dim)
           self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        B, N, C = x.shape
        if self.with_qkv:
           qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
           q, k, v = qkv[0], qkv[1], qkv[2]
        else:
           qkv = x.reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
           q, k, v  = qkv, qkv, qkv

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        if self.with_qkv:
           x = self.proj(x)
           x = self.proj_drop(x)
        return x

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = Attention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        
class BlockCross(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.1, act_layer=nn.GELU, norm_layer=nn.LayerNorm, attention_type='divided_space_time'):
        super().__init__()
        self.attention_type = attention_type
        assert(attention_type in ['divided_space_time', 'space_only','joint_space_time'])

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
           dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        ## Temporal Attention Parameters
        if self.attention_type == 'divided_space_time':
            self.temporal_norm1 = norm_layer(dim)
            self.temporal_attn = MaskAttention(
              dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
            self.temporal_fc = nn.Linear(dim, dim)

        ## drop path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self, x, B, T, W):
        num_spatial_tokens = (x.size(1) - 1) // T
        H = num_spatial_tokens // W

        if self.attention_type in ['space_only', 'joint_space_time']:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x
        elif self.attention_type == 'divided_space_time':
            ## Temporal
            xt = x[:,1:,:]
            xt = rearrange(xt, 'b (h w t) m -> (b h w) t m',b=B,h=H,w=W,t=T)
            res_temporal = self.drop_path(self.temporal_attn(self.temporal_norm1(xt)))
            res_temporal = rearrange(res_temporal, '(b h w) t m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res_temporal = self.temporal_fc(res_temporal)
            xt = x[:,1:,:] + res_temporal

            ## Spatial
            init_cls_token = x[:,0,:].unsqueeze(1)
            cls_token = init_cls_token.repeat(1, T, 1)
            cls_token = rearrange(cls_token, 'b t m -> (b t) m',b=B,t=T).unsqueeze(1)
            xs = xt
            xs = rearrange(xs, 'b (h w t) m -> (b t) (h w) m',b=B,h=H,w=W,t=T)
            xs = torch.cat((cls_token, xs), 1)
            res_spatial = self.drop_path(self.attn(self.norm1(xs)))

            ### Taking care of CLS token
            cls_token = res_spatial[:,0,:]
            cls_token = rearrange(cls_token, '(b t) m -> b t m',b=B,t=T)
            cls_token = torch.mean(cls_token,1,True) ## averaging for every frame
            res_spatial = res_spatial[:,1:,:]
            res_spatial = rearrange(res_spatial, '(b t) (h w) m -> b (h w t) m',b=B,h=H,w=W,t=T)
            res = res_spatial
            x = xt

            ## Mlp
            x = torch.cat((init_cls_token, x), 1) + torch.cat((cls_token, res), 1)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size =img_size #to_2tuple(img_size)
        patch_size =patch_size #to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, T, H, W = x.shape
        sizes = x.shape
        x = rearrange(x, 'b c t h w -> (b t) c h w')
        x = self.proj(x)
        W = x.size(-1)
        x = x.flatten(2).transpose(1, 2)
        return x, T, W,sizes


class VisionTransformer(nn.Module):
    """ Vision Transformere
    """
    def __init__(self,batch_size:int,output_size:List[int], img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.,
                 out_chans=1,in_periods=1,place='Arctic',emb_mult=1):
        super().__init__()
        self.emb_mult=emb_mult
        self.emb_dim = embed_dim
        self.in_period = in_periods
        self.attention_type = attention_type
        self.depth = depth
        self.channel_from_1dim  = output_size[0]//patch_size[0]*output_size[1]//patch_size[1]*in_periods//output_size[0]
        self.img_size = output_size
         
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=output_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        ###Conv layers
        self.padings = {'Arctic':[(3,1),(1,1),(1,0)],'kara':[(0,1),(0,1),(0,1)],'laptev':[(3,1),(1,0),(1,0)]}

        ###new conv
        # self.conv_pred = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(4,3),stride=(1,1),padding=(1,1))
        #                                 )
        
        self.in_channels = int(self.emb_mult*self.channel_from_1dim)
        print('in_channels',self.in_channels)
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels*4,kernel_size=3,stride=1,padding=(1,1)),
                                            nn.ReLU(),
                                            nn.BatchNorm2d(self.in_channels*4))
            
        self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=self.in_channels*4,out_channels=256,kernel_size=3,stride=(1,1),padding=(1,1)),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(256),
                                        nn.Conv2d(in_channels=256,out_channels=out_chans,kernel_size=3,stride=(1,1),padding=(1,1)),
                                        nn.ReLU(),
                                        nn.BatchNorm2d(out_chans))
        # if place=='laptev':
        #     self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=self.in_period,out_channels=64,kernel_size=3,stride=(2,2),padding=(3,1)),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(64))
            
        #     self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=(1,1),padding=(1,0)),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(256),
        #                                     nn.Conv2d(in_channels=256,out_channels=out_chans,kernel_size=3,stride=(1,1),padding=(1,0)),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(out_chans))
        # elif place=='Arctic':
        #     self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=self.in_period,out_channels=64,kernel_size=3,stride=(2,2),padding=(3,1)),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(64))
            
        #     self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=(1,1),padding=(1,1)),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(256),
        #                                     nn.Conv2d(in_channels=256,out_channels=out_chans,kernel_size=3,stride=(1,1),padding=(1,1)),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(out_chans))
        # if place=='kara':
        #     self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=self.in_period,out_channels=64,kernel_size=3,stride=(2,2),padding=(0,1)),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(64))
            
        #     self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=(1,1),padding=(0,1)),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(256),
        #                                     nn.Conv2d(in_channels=256,out_channels=out_chans,kernel_size=3,stride=(1,1),padding=(0,1)),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(out_chans))
        # self.conv3_3 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=out_chans,kernel_size=3,stride=1,padding=self.padings[place][0]),
        #                                     nn.ReLU(),
        #                            
        #          nn.BatchNorm2d(out_chans))
        # ## Positional Embeddings
        self.sigm = nn.Sigmoid()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W,sizes = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        # y1 = self.conv1(x.unsqueeze(1))
        # y2 = torch.reshape(y1,(self.batch_size,128,y1.shape[2]//2,self.embed_dim))
        # y3 = self.conv2(y2)
        # y4 = self.conv3(y3)
        y1 = torch.reshape(x,(x.shape[0],1,x.shape[1],x.shape[2]))
        img_scaling = x.shape[1]//(self.img_size[0])*self.img_size[0]
        dim2_scale = x.shape[1]//(self.img_size[0])
        dim3_scale = x.shape[2]//(self.img_size[1])
        y2 = F.interpolate(y1,(img_scaling,y1.shape[-1]))
        y3 = torch.reshape(y2,(y2.shape[0],dim2_scale*dim3_scale,y2.shape[2]//dim2_scale,y2.shape[3]//dim3_scale))
        
        # x1 = self.conv_pred(x.unsqueeze(1))
        # x1 = torch.reshape(x1,(x1.shape[0],self.in_period,x1.shape[2]//self.in_period,self.embed_dim))
        x2 = self.conv1_2(y3)
        x3 = self.conv2_2(x2)
        return self.sigm(x3)

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)
        return x
 

class VisionTransformer_3d(nn.Module):
    """ Vision Transformere
    """
    def __init__(self,batch_size:int,output_size:List[int], img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.,
                 out_chans=1,in_periods=1,place='Arctic'):
        super().__init__()
        self.emb_dim = embed_dim
        self.in_period = in_periods
        self.attention_type = attention_type
        self.depth = depth
        self.S1 = (img_size//patch_size)**2*in_periods
        self.att_size = [1,self.S1*batch_size+1,embed_dim]
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        ###Conv layers
        self.padings = {'Arctic':[(3,1),(1,1),(1,0)],'kara':[(0,1),(0,1),(0,1)],'laptev':[(3,1),(1,0),(1,0)]}

        ###new conv
        self.conv_pred = nn.Sequential(nn.Conv2d(in_channels=1,out_channels=1,kernel_size=(4,3),stride=(1,1),padding=(1,1))
                                        )
        

        # self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=self.in_period,out_channels=64,kernel_size=3,stride=(2,2),padding=self.padings[place][0]),
        #                                     nn.ReLU(),
        #                                     nn.BatchNorm2d(64))
            
        # self.conv2_2 = nn.Sequential(nn.Conv2d(in_channels=64,out_channels=256,kernel_size=3,stride=(1,1),padding=self.padings[place][1]),
        #                                 nn.ReLU(),
        #                                 nn.BatchNorm2d(256),
        #                                 nn.Conv2d(in_channels=256,out_channels=out_chans,kernel_size=3,stride=(1,1),padding=self.padings[place][2]),
        #                                 nn.ReLU(),
        #                                 nn.BatchNorm2d(out_chans))
        

        self.conv3d_1 = nn.Sequential(nn.Conv3d(in_channels=1,out_channels=out_chans*8,kernel_size=(3,3,3),stride=(1,2,2),padding=(1,0,1)),
                                            nn.ReLU(),
                                            nn.BatchNorm3d(out_chans*8))
        # self.conv3d_2 = nn.Sequential(nn.Conv3d(in_channels=self.in_period,out_channels=64,kernel_size=3,stride=(1,2,2),padding=(1,0,1)),
        #                                     nn.ReLU())
        #paddig = max((math.ceil(self.in_period / 1) - 1) * 1 + (self.in_period - 1) * 1 + 1 - i, 0)
        self.conv3d_t1 = nn.Sequential(nn.Conv3d(in_channels=out_chans*8,out_channels=out_chans*8,kernel_size=(3,3,3),stride=(1,1,1),padding=(1,0,1)),
                                        nn.ReLU(),
                                        nn.BatchNorm3d(out_chans*8),
                                        nn.Conv3d(in_channels=out_chans*8,out_channels=out_chans,kernel_size=(self.in_period,3,3),stride=(1,1,1),padding=(0,0,1),groups=out_chans),
                                        nn.ReLU(),
                                        nn.BatchNorm3d(out_chans))
        # self.conv3d_2 = nn.Sequential(nn.Conv3d(in_channels=out_chans,out_channels=out_chans,kernel_size=(self.in_period,3,3),stride=(1,1,1),padding=(0,0,1),groups=out_chans),
        #                                 nn.ReLU(),
        #                                 nn.BatchNorm3d(out_chans),
        #                                 nn.Conv3d(in_channels=out_chans,out_channels=out_chans,kernel_size=(self.in_period,3,3),stride=(1,1,1),padding=(1,0,1),groups=out_chans),
        #                                 nn.ReLU(),
        #                                 nn.BatchNorm3d(out_chans))

        # ## Positional Embeddings
        self.sigm = nn.Sigmoid()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W,sizes = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        # y1 = self.conv1(x.unsqueeze(1))
        # y2 = torch.reshape(y1,(self.batch_size,128,y1.shape[2]//2,self.embed_dim))
        # y3 = self.conv2(y2)
        # y4 = self.conv3(y3)

        x1 = self.conv_pred(x.unsqueeze(1))
        x1 = torch.reshape(x1,(self.batch_size,1,self.in_period,x1.shape[2]//self.in_period,self.embed_dim))
        x2 = self.conv3d_1(x1)
        x3 = self.conv3d_t1(x2).squeeze(2)
        return self.sigm(x3)

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)
        return x

def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            if v.shape[-1] != patch_size:
                patch_size = v.shape[-1]
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v
    return out_dict

 
class vit_base_patch16_224(nn.Module):
    def __init__(self, cfg, **kwargs):
        super(vit_base_patch16_224, self).__init__()
        self.pretrained=True
        patch_size = 16
        self.model = VisionTransformer(img_size=cfg.DATA.TRAIN_CROP_SIZE, num_classes=cfg.MODEL.NUM_CLASSES, patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=cfg.DATA.NUM_FRAMES, attention_type=cfg.TIMESFORMER.ATTENTION_TYPE, **kwargs)

        self.attention_type = cfg.TIMESFORMER.ATTENTION_TYPE
        #self.model.default_cfg = default_cfgs['vit_base_patch16_224']
        self.num_patches = (cfg.DATA.TRAIN_CROP_SIZE // patch_size) * (cfg.DATA.TRAIN_CROP_SIZE // patch_size)
        pretrained_model=cfg.TIMESFORMER.PRETRAINED_MODEL
    def forward(self, x):
        x = self.model(x)
        return x

 
class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(TimeSformer, self).__init__()
        self.pretrained=False
        #self.emb_dim=kwargs['embeded_dim']
        self.model = VisionTransformer(img_size=img_size, num_classes=num_classes, patch_size=patch_size, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        #self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (kwargs['output_size'][0] // patch_size[0]) * (kwargs['output_size'][1] // patch_size[1])
    def forward(self, x):
        x = self.model(x)
        return x
    
 
class TimeSformer_3d(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(TimeSformer_3d, self).__init__()
        self.pretrained=False
        #self.emb_dim=kwargs['embeded_dim']
        self.model = VisionTransformer_3d(img_size=img_size, num_classes=num_classes, patch_size=patch_size, depth=12, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        #self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (img_size // patch_size) * (img_size // patch_size)

    def forward(self, x):
        x = self.model(x)
        return x


 
class VisionTransformer_conv_aug(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=400, num_frames=8, attention_type='divided_space_time',  pretrained_model='', **kwargs):
        super(VisionTransformer_conv_aug, self).__init__()
        self.pretrained=False
        #self.emb_dim=kwargs['embeded_dim']
        self.model = VisionTransformer_conv_augment(img_size=img_size, num_classes=num_classes, patch_size=patch_size, mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), drop_path_rate=0.1, num_frames=num_frames, attention_type=attention_type, **kwargs)

        self.attention_type = attention_type
        #self.model.default_cfg = default_cfgs['vit_base_patch'+str(patch_size)+'_224']
        self.num_patches = (kwargs['output_size'][0] // patch_size[0]) * (kwargs['output_size'][1] // patch_size[1])
    def forward(self, x):
        x = self.model(x)
        return x

class VisionTransformer_conv_augment(nn.Module):
    """ Vision Transformere
    """
    def __init__(self,batch_size:int,output_size:List[int], img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1, hybrid_backbone=None, norm_layer=nn.LayerNorm, num_frames=8, attention_type='divided_space_time', dropout=0.,
                 out_chans=1,in_periods=1,place='Arctic',emb_mult=1):
        super().__init__()
        self.emb_mult=emb_mult
        self.emb_dim = embed_dim
        self.in_period = in_periods
        self.attention_type = attention_type
        self.depth = depth
        self.channel_from_1dim  = output_size[0]/patch_size[0]*output_size[1]/patch_size[1]*in_periods/output_size[0]
        # self.att_size = [1,self.S1*batch_size+1,embed_dim]
        self.batch_size = batch_size
        self.dropout = nn.Dropout(dropout)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_embed = PatchEmbed(
            img_size=output_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.img_size = output_size
        num_patches = self.patch_embed.num_patches
        self.out_chans = out_chans
        ###Conv layers
        self.padings = {'Arctic':[(3,1),(1,1),(1,0)],'kara':[(0,1),(0,1),(0,1)],'laptev':[(3,1),(1,0),(1,0)]}
        self.in_channels = int(self.emb_mult*self.channel_from_1dim)
        channel_scale = 3
        self.conv = nn.Sequential(nn.Conv2d(in_channels=self.in_channels,out_channels=self.in_channels*channel_scale,kernel_size=3,stride=(1,1),padding=(1,1)),
                                        nn.SELU(),
                                        nn.BatchNorm2d(self.in_channels*channel_scale),
                                        nn.Conv2d(in_channels=self.in_channels*channel_scale,out_channels=self.in_channels*channel_scale,kernel_size=3,stride=(1,1),padding=(1,1)),
                                        nn.SELU(),
                                        nn.BatchNorm2d(self.in_channels*channel_scale),
                                        nn.Conv2d(in_channels=self.in_channels*channel_scale,out_channels=out_chans,kernel_size=3,stride=(1,1),padding=(1,1)))

        # ## Positional Embeddings
        self.sigm = nn.Sigmoid()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        if self.attention_type != 'space_only':
            self.time_embed = nn.Parameter(torch.zeros(1, num_frames, embed_dim))
            self.time_drop = nn.Dropout(p=drop_rate)

        ## Attention Blocks
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, attention_type=self.attention_type)
            for i in range(self.depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

        ## initialization of temporal attention weights
        if self.attention_type == 'divided_space_time':
            i = 0
            for m in self.blocks.modules():
                m_str = str(m)
                if 'Block' in m_str:
                    if i > 0:
                      nn.init.constant_(m.temporal_fc.weight, 0)
                      nn.init.constant_(m.temporal_fc.bias, 0)
                    i += 1
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x, T, W,sizes = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        ## resizing the positional embeddings in case they don't match the input at inference
        if x.size(1) != self.pos_embed.size(1):
            pos_embed = self.pos_embed
            cls_pos_embed = pos_embed[0,0,:].unsqueeze(0).unsqueeze(1)
            other_pos_embed = pos_embed[0,1:,:].unsqueeze(0).transpose(1, 2)
            P = int(other_pos_embed.size(2) ** 0.5)
            H = x.size(1) // W
            other_pos_embed = other_pos_embed.reshape(1, x.size(2), P, P)
            new_pos_embed = F.interpolate(other_pos_embed, size=(H, W), mode='nearest')
            new_pos_embed = new_pos_embed.flatten(2)
            new_pos_embed = new_pos_embed.transpose(1, 2)
            new_pos_embed = torch.cat((cls_pos_embed, new_pos_embed), 1)
            x = x + new_pos_embed
        else:
            x = x + self.pos_embed
        x = self.pos_drop(x)


        ## Time Embeddings
        if self.attention_type != 'space_only':
            cls_tokens = x[:B, 0, :].unsqueeze(1)
            x = x[:,1:]
            x = rearrange(x, '(b t) n m -> (b n) t m',b=B,t=T)
            ## Resizing time embeddings in case they don't match
            if T != self.time_embed.size(1):
                time_embed = self.time_embed.transpose(1, 2)
                new_time_embed = F.interpolate(time_embed, size=(T), mode='nearest')
                new_time_embed = new_time_embed.transpose(1, 2)
                x = x + new_time_embed
            else:
                x = x + self.time_embed
            x = self.time_drop(x)
            x = rearrange(x, '(b n) t m -> b (n t) m',b=B,t=T)
            x = torch.cat((cls_tokens, x), dim=1)

        ## Attention blocks
        for blk in self.blocks:
            x = blk(x, B, T, W)

        ### Predictions for space-only baseline
        if self.attention_type == 'space_only':
            x = rearrange(x, '(b t) n m -> b t n m',b=B,t=T)
            x = torch.mean(x, 1) # averaging predictions for every frame

        x = self.norm(x)
        y1 = torch.reshape(x,(x.shape[0],1,x.shape[1],x.shape[2]))
        img_scaling = x.shape[1]//(self.img_size[0])*self.img_size[0]
        dim2_scale = x.shape[1]//(self.img_size[0])
        dim3_scale = x.shape[2]//(self.img_size[1])
        y2 = F.interpolate(y1,(img_scaling,y1.shape[-1]))
        y3 = torch.reshape(y2,(y2.shape[0],dim2_scale*dim3_scale,y2.shape[2]//dim2_scale,y2.shape[3]//dim3_scale))
        y = self.conv(y3)

        
        # y3 = torch.reshape(y2,(y2.shape[0],self.out_chans,y2.shape[2]//self.out_chans,y2.shape[3]))
        # y = F.interpolate(y3,(self.img_size[0],self.img_size[1]))

        # x1 = torch.reshape(x,(self.batch_size,x.shape[2],x.shape[1]))
        # x2 = nn.functional.interpolate(x1,img_scaling)
        # scale_factor_out = x2.shape[2]*x2.shape[1]//self.img_size[0]//self.img_size[1]//self.out_chans#self.out_chan
        # x3 = torch.reshape(x2,(self.batch_size,self.out_chans,self.img_size[0]*scale_factor_out,self.img_size[1]*2))
        # y = nn.functional.interpolate(x3,(self.img_size[0],self.img_size[1]),mode='bilinear')
        #x3 = torch.reshape(x2,(self.batch_size,x2.shape[2]//self.img_size[0]*2,self.img_size[0],self.img_size[1]))
        
        # y1 = self.conv1(x.unsqueeze(1))
        # y2 = torch.reshape(y1,(self.batch_size,128,y1.shape[2]//2,self.embed_dim))
        # y3 = self.conv2(y2)
        # y4 = self.conv3(y3)

        # x1 = self.conv_pred(x.unsqueeze(1))
        # x1 = torch.reshape(x1,(self.batch_size,self.in_period,x1.shape[2]//self.in_period,self.embed_dim))
        # x2 = self.conv1_2(x1)
        # x3 = self.conv2_2(x2)
        return self.sigm(y)

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)
        return x
    
def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output