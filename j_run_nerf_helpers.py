import jittor as jt
import jittor.nn as nn

import numpy as np

# Misc
img2mse = lambda x, y : jt.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * jt.log(x) / jt.log(jt.Tensor([10.]))
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

#位置编码
class Embedder:
    def __init__(self, **kwargs):
        # 变量1：self.kwargs → 接收位置编码的所有配置参数,应该包含：
        # input_dims 输入维度（三维）
        # include_input 布尔值 代表是否保留原本输入
        # max_freq_log2 最大频率
        # num_freqs 频率个数 
        # log_sampling 布尔值 决定是否对数采样
        # periodic_fns 选用的周期函数
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        #输出维度
        out_dim = 0
        #判断是否保留原有数据
        if self.kwargs['include_input']:
            embed_fns.append(lambda x : x)
            out_dim += d

        #设置频率
        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']
        #是否对数采样
        if self.kwargs['log_sampling']:
            freq_bands = 2.**jt.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = jt.linspace(2.**0., 2.**max_freq, steps=N_freqs)

        #构建编码函数
        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
                out_dim += d

        #绑定
        self.embed_fns = embed_fns
        self.out_dim = out_dim
    #拼接
    def embed(self, inputs):
        return jt.cat([fn(inputs) for fn in self.embed_fns], -1)
    
def get_embedder(multires, i=0):
    #特殊情况，恒等编码
    if i == -1:
        return nn.Identity(), 3#输出为模型加输出数据的维数
    #设置kwargs的参数
    embed_kwargs = {
                'include_input' : True,
                'input_dims' : 3,
                'max_freq_log2' : multires-1,
                'num_freqs' : multires,
                'log_sampling' : True,
                'periodic_fns' : [jt.sin, jt.cos],#使用三角函数作为编码函数
    }
    #生成编码函数
    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj : eo.embed(x)
    return embed, embedder_obj.out_dim

#核心MLP
import jittor as jt
import numpy as np

class NeRF(jt.Module):  # 替换为 jt.Module（关键！）
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, output_ch=4, skips=[4], use_viewdirs=False):
        super(NeRF, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # 替换为 jittor 的线性层（jt.nn.Linear）
        self.pts_linears = jt.nn.ModuleList(
            [jt.nn.Linear(input_ch, W)] + 
            [jt.nn.Linear(W, W) if i not in self.skips else jt.nn.Linear(W + input_ch, W) 
             for i in range(D-1)]
        )

        self.views_linears = jt.nn.ModuleList([jt.nn.Linear(input_ch_views + W, W//2)])

        if use_viewdirs:
            self.feature_linear = jt.nn.Linear(W, W)
            self.alpha_linear = jt.nn.Linear(W, 1)
            self.rgb_linear = jt.nn.Linear(W//2, 3)
        else:
            self.output_linear = jt.nn.Linear(W, output_ch)

    # 替换 forward 为 execute（Jittor 要求的前向传播方法）
    def execute(self, x):
        # 拆分输入（jt.split 保持不变）
        input_pts, input_views = jt.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        
        # 遍历 pts_linears 网络层
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = jt.nn.relu(h)  # 替换为 jittor 的 relu
            if i in self.skips:
                h = jt.cat([input_pts, h], -1)

        # 视角分支处理
        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = jt.cat([feature, input_views], -1)
        
            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = jt.nn.relu(h)  # 替换为 jittor 的 relu

            rgb = self.rgb_linear(h)
            outputs = jt.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)
        
        return outputs

    # 修正权重加载逻辑（Jittor 张量无 .data 属性）
    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"
        
        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            # Jittor 直接赋值权重，无需 .data
            self.pts_linears[i].weight = jt.array(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias = jt.array(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight = jt.array(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias = jt.array(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight = jt.array(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias = jt.array(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight = jt.array(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias = jt.array(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight = jt.array(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias = jt.array(np.transpose(weights[idx_alpha_linear+1]))

#光线处理
def get_rays(H, W, K, c2w):
    i, j = jt.meshgrid(jt.linspace(0, W-1, W), jt.linspace(0, H-1, H))  # pytorch's meshgrid has indexing='ij'
    i = i.t()
    j = j.t()
    dirs = jt.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -jt.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = jt.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = c2w[:3,-1].expand(rays_d.shape)
    return rays_o, rays_d

#numpy版的光线处理
def get_rays_np(H, W, K, c2w):
    i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-K[0][2])/K[0][0], -(j-K[1][2])/K[1][1], -np.ones_like(i)], -1)
    # Rotate ray directions from camera frame to the world frame
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    # Translate camera frame's origin to the world frame. It is the origin of all rays.
    rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
    return rays_o, rays_d

#归一化设备坐标（NDC）的光线转换
def ndc_rays(H, W, focal, near, rays_o, rays_d):
    # 平移光线原点到近平面
    t = -(near + rays_o[...,2]) / rays_d[...,2]
    rays_o = rays_o + t[...,None] * rays_d

    # NDC投影
    o0 = -1./(W/(2.*focal)) * rays_o[...,0] / rays_o[...,2]
    o1 = -1./(H/(2.*focal)) * rays_o[...,1] / rays_o[...,2]
    o2 = 1. + 2. * near / rays_o[...,2]

    d0 = -1./(W/(2.*focal)) * (rays_d[...,0]/rays_d[...,2] - rays_o[...,0]/rays_o[...,2])
    d1 = -1./(H/(2.*focal)) * (rays_d[...,1]/rays_d[...,2] - rays_o[...,1]/rays_o[...,2])
    d2 = -2. * near / rays_o[...,2]

    # 拼接结果
    rays_o = jt.stack([o0, o1, o2], -1)
    rays_d = jt.stack([d0, d1, d2], -1)

    return rays_o, rays_d

#分层采样
def sample_pdf(bins, weights, N_samples, det=False, pytest=False):
    #获取权重
    weights = weights + 1e-5 # prevent nans
    pdf = weights / jt.sum(weights, -1, keepdim=True)
    cdf = jt.cumsum(pdf, -1)
    cdf = jt.cat([jt.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    #取样方式(随机取样、均匀取样)
    if det:
        u = jt.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = jt.rand(list(cdf.shape[:-1]) + [N_samples])

    #测试模式
    if pytest:
        np.random.seed(0)
        new_shape = list(cdf.shape[:-1]) + [N_samples]
        if det:
            u = np.linspace(0., 1., N_samples)
            u = np.broadcast_to(u, new_shape)
        else:
            u = np.random.rand(*new_shape)
        u = jt.Tensor(u)

    #
    u = u.contiguous()
    inds = jt.searchsorted(cdf, u, right=True)
    below = jt.maximum(jt.zeros_like(inds-1), inds-1)
    above = jt.minimum((jt.ones_like(inds-1) * (bins.shape[-1]-1)), inds)
    inds_g = jt.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = jt.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = jt.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = jt.where(denom<1e-5, jt.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples