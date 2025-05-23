import torch
import math
from torch import nn
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
import torch.nn.functional as F
from torch import einsum
from models.CRAPCN_utils import PointNet_SA_Module_KNN, MLP_Res, MLP_CONV, fps_subsample, query_knn, grouping_operation, \
    get_nearest_index, indexing_neighbor
from pointnet2_ops import pointnet2_utils
from extensions.chamfer_dist import ChamferDistanceL1, ChamferDistanceL1_PM
from models.build import MODELS
from MPGLNet_utils import SA_Layer, MSMHA, EDformer
from knn_cuda import KNN
from extensions.infocd.Chamfer3D.dist_chamfer_3D import chamfer_3DDist

def fps(pc, num):
    fps_idx = pointnet2_utils.furthest_point_sample(pc, num)
    sub_pc = pointnet2_utils.gather_operation(pc.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return sub_pc

chamfer_dist = chamfer_3DDist()
def calc_cd_like_InfoV2(p1, p2):
    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    dist1 = torch.clamp(dist1, min=1e-9)
    dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)

    distances1 = - torch.log(
        torch.exp(-0.5 * d1) / (torch.sum(torch.exp(-0.5 * d1) + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7)
    distances2 = - torch.log(
        torch.exp(-0.5 * d2) / (torch.sum(torch.exp(-0.5 * d2) + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7)

    return (torch.sum(distances1) + torch.sum(distances2)) / 2


def calc_cd_one_side_like_InfoV2(p1, p2):
    dist1, dist2, idx1, idx2 = chamfer_dist(p1, p2)
    dist1 = torch.clamp(dist1, min=1e-9)
    dist2 = torch.clamp(dist2, min=1e-9)
    d1 = torch.sqrt(dist1)
    d2 = torch.sqrt(dist2)

    distances1 = - torch.log(
        torch.exp(-0.5 * d1) / (torch.sum(torch.exp(-0.5 * d1) + 1e-7, dim=-1).unsqueeze(-1)) ** 1e-7)

    return torch.sum(distances1)

class VectorAttention(nn.Module):
    def __init__(self, in_channel=128, dim=64, n_knn=16, attn_hidden_multiplier=4):
        super().__init__()
        self.n_knn = n_knn
        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)
        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, dim, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 1)
        )
        self.attn_mlp = nn.Sequential(
            nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, dim, 1)
        )
        self.conv_end = nn.Conv1d(dim, in_channel, 1)

    def forward(self, query, support):
        pq, fq = query
        ps, fs = support

        identity = fq
        query, key, value = self.conv_query(fq), self.conv_key(fs), self.conv_value(fs)

        B, D, N = query.shape

        pos_flipped_1 = ps.permute(0, 2, 1).contiguous()
        pos_flipped_2 = pq.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos_flipped_1, pos_flipped_2)

        key = grouping_operation(key, idx_knn)
        qk_rel = query.reshape((B, -1, N, 1)) - key

        pos_rel = pq.reshape((B, -1, N, 1)) - grouping_operation(ps, idx_knn)
        pos_embedding = self.pos_mlp(pos_rel)

        attention = self.attn_mlp(qk_rel + pos_embedding)
        attention = torch.softmax(attention, -1)

        value = grouping_operation(value, idx_knn) + pos_embedding
        agg = einsum('b c i j, b c i j -> b c i', attention, value)
        output = self.conv_end(agg) + identity

        return output


def hierarchical_fps(pts, rates):
    pts_flipped = pts.permute(0, 2, 1).contiguous()
    B, _, N = pts.shape
    now = N
    fps_idxs = []
    for i in range(len(rates)):
        now = now // rates[i]
        if now == N:
            fps_idxs.append(None)
        else:
            fps_idxs.append(furthest_point_sample(pts_flipped, now))
    return fps_idxs


# project f from p1 onto p2
def three_inter(f, p1, p2):
    # print(f.shape, p1.shape, p2.shape)
    # p1_flipped = p1.permute(0, 2, 1).contiguous()
    # p2_flipped = p2.permute(0, 2, 1).contiguous()
    idx, dis = get_nearest_index(p2, p1, k=3, return_dis=True)
    dist_recip = 1.0 / (dis + 1e-8)
    norm = torch.sum(dist_recip, dim=2, keepdim=True)
    weight = dist_recip / norm
    proj_f = torch.sum(indexing_neighbor(f, idx) * weight.unsqueeze(1), dim=-1)
    return proj_f


# Cross-Resolution Transformer
class CRT(nn.Module):
    def __init__(self, dim_in=128, is_inter=True, down_rates=[1, 4, 2], knns=[16, 12, 8]):
        super().__init__()
        self.down_rates = down_rates
        self.is_inter = is_inter
        self.num_scale = len(down_rates)

        self.attn_lists = nn.ModuleList()
        self.q_mlp_lists = nn.ModuleList()
        self.s_mlp_lists = nn.ModuleList()
        for i in range(self.num_scale):
            self.attn_lists.append(VectorAttention(in_channel=dim_in, dim=64, n_knn=knns[i]))

        for i in range(self.num_scale - 1):
            self.q_mlp_lists.append(MLP_Res(in_dim=128 * 2, hidden_dim=128, out_dim=128))
            self.s_mlp_lists.append(MLP_Res(in_dim=128 * 2, hidden_dim=128, out_dim=128))

    def forward(self, query, support, fps_idxs_q=None, fps_idxs_s=None):
        pq, fq = query
        ps, fs = support
        # prepare fps_idxs_q and fps_idxs_s
        if fps_idxs_q == None:
            fps_idxs_q = hierarchical_fps(pq, self.down_rates)

        if fps_idxs_s == None:
            if self.is_inter:
                fps_idxs_s = hierarchical_fps(ps, self.down_rates)  # inter-level
            else:
                fps_idxs_s = fps_idxs_q  # intra-level

        # top-down aggregation
        pre_f = None
        pre_pos = None

        for i in range(self.num_scale - 1, -1, -1):
            if fps_idxs_q[i] == None:
                _pos1 = pq
            else:
                _pos1 = gather_operation(pq, fps_idxs_q[i])

            if fps_idxs_s[i] == None:
                _pos2 = ps
            else:
                _pos2 = gather_operation(ps, fps_idxs_s[i])

            if i == self.num_scale - 1:
                if fps_idxs_q[i] == None:
                    _f1 = fq
                else:
                    _f1 = gather_operation(fq, fps_idxs_q[i])
                if fps_idxs_s[i] == None:
                    _f2 = fs
                else:
                    _f2 = gather_operation(fs, fps_idxs_s[i])

            else:
                proj_f1 = three_inter(pre_f, pre_pos, _pos1)
                proj_f2 = three_inter(pre_f, pre_pos, _pos2)
                if fps_idxs_q[i] == None:
                    _f1 = fq
                else:
                    _f1 = gather_operation(fq, fps_idxs_q[i])
                if fps_idxs_s[i] == None:
                    _f2 = fs
                else:
                    _f2 = gather_operation(fs, fps_idxs_s[i])

                _f1 = self.q_mlp_lists[i](torch.cat([_f1, proj_f1], dim=1))
                _f2 = self.s_mlp_lists[i](torch.cat([_f2, proj_f2], dim=1))

            f = self.attn_lists[i]([_pos1, _f1], [_pos2, _f2])

            pre_f = f
            pre_pos = _pos1

        agg_f = pre_f
        return agg_f, fps_idxs_q, fps_idxs_s

class CrossTransformer(nn.Module):
    def __init__(self, in_channel, dim=256, n_knn=16, pos_hidden_dim=64, attn_hidden_multiplier=4):
        super(CrossTransformer, self).__init__()
        self.n_knn = n_knn

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, in_channel, 1)
        )

        self.attn_mlp = nn.Sequential(
            nn.Conv2d(in_channel, dim * attn_hidden_multiplier, 1),
            nn.BatchNorm2d(dim * attn_hidden_multiplier),
            nn.ReLU(),
            nn.Conv2d(dim * attn_hidden_multiplier, in_channel, 1)
        )

    def forward(self, pcd, feat, pcd_feadb, feat_feadb):
        b, _, num_point = pcd.shape

        fusion_pcd = torch.cat((pcd, pcd_feadb), dim=2)
        fusion_feat = torch.cat((feat, feat_feadb), dim=2)

        key_point = pcd
        key_feat = feat

        key_point_idx = query_knn(self.n_knn, fusion_pcd.transpose(2, 1).contiguous(),
                                  key_point.transpose(2, 1).contiguous(), include_self=True)

        group_point = grouping_operation(fusion_pcd, key_point_idx)
        group_feat = grouping_operation(fusion_feat, key_point_idx)

        qk_rel = key_feat.reshape((b, -1, num_point, 1)) - group_feat
        pos_rel = key_point.reshape((b, -1, num_point, 1)) - group_point

        pos_embedding = self.pos_mlp(pos_rel)
        sample_weight = self.attn_mlp(qk_rel + pos_embedding)  # b, in_channel + 3, n, n_knn
        sample_weight = torch.softmax(sample_weight, -1)  # b, in_channel + 3, n, n_knn

        group_feat = group_feat + pos_embedding  #
        refined_feat = einsum('b c i j, b c i j -> b c i', sample_weight, group_feat)

        return refined_feat

#encoder
class LGGP(nn.Module):
    def __init__(self, out_dim=512, n_knn=16):
        super().__init__()
        self.OA = SA_Layer(512)
        self.CrossTransformer = CrossTransformer(in_channel=512)
        self.offset = SA_Layer(512)

        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)

        self.conv1 = nn.Conv1d(3, 512, 1)
        self.conv2 = nn.Conv1d(256, 512, 1)

    def forward(self, partial_cloud):

        # B, N, C = partial_cloud.shape
        l2048_xyz = partial_cloud #[B, 2048, 3]
        l2048_points = partial_cloud

        l1_xyz, l1_points, _ = self.sa_module_1(l2048_xyz.permute(0, 2, 1).contiguous(), l2048_points.permute(0, 2, 1).contiguous())
        l2_xyz, l2_points, _ = self.sa_module_2(l1_xyz, l1_points)

        l2_points_ = self.conv2(l2_points)
        l1024_xyz = fps(l2048_xyz, 1024)

        l1024_f = self.conv1(l1024_xyz.permute(0, 2, 1))

        x_oa2 = self.OA(l1024_f)
        x_ct2 = self.CrossTransformer(l1024_xyz.permute(0, 2, 1).contiguous(), x_oa2, l2_xyz, l2_points_)
        x_att_1024 = self.offset(x_ct2)
        x_1024 = F.adaptive_max_pool1d(x_att_1024, 1)

        return l2_xyz, l2_points, x_1024

class UpTransformer(nn.Module):
    def __init__(self, in_channel=128, out_channel=128, dim=64, n_knn=20, up_factor=2,
                 pos_hidden_dim=64, attn_hidden_multiplier=4, scale_layer=nn.Softmax, attn_channel=True):
        super(UpTransformer, self).__init__()
        self.n_knn = n_knn
        self.up_factor = up_factor
        attn_out_channel = dim if attn_channel else 1

        self.conv_key = nn.Conv1d(in_channel, dim, 1)
        self.conv_query = nn.Conv1d(in_channel, dim, 1)
        self.conv_value = nn.Conv1d(in_channel, dim, 1)

        self.scale = scale_layer(dim=-1) if scale_layer is not None else nn.Identity()

        self.pos_mlp = nn.Sequential(
            nn.Conv2d(3, pos_hidden_dim, 1),
            nn.BatchNorm2d(pos_hidden_dim),
            nn.ReLU(),
            nn.Conv2d(pos_hidden_dim, dim, 1)
        )

        # attention layers
        self.attn_mlp = [nn.Conv2d(dim, dim * attn_hidden_multiplier, 1),
                         nn.BatchNorm2d(dim * attn_hidden_multiplier),
                         nn.ReLU()]
        if up_factor:
            self.attn_mlp.append(
                nn.ConvTranspose2d(dim * attn_hidden_multiplier, attn_out_channel, (up_factor, 1), (up_factor, 1)))
        else:
            self.attn_mlp.append(nn.Conv2d(dim * attn_hidden_multiplier, attn_out_channel, 1))
        self.attn_mlp = nn.Sequential(*self.attn_mlp)

        # upsample previous feature
        self.upsample1 = nn.Upsample(scale_factor=(up_factor, 1)) if up_factor else nn.Identity()
        self.upsample2 = nn.Upsample(scale_factor=up_factor) if up_factor else nn.Identity()

        # residual connection
        self.conv_end = nn.Conv1d(dim, out_channel, 1)
        if in_channel != out_channel:
            self.residual_layer = nn.Conv1d(in_channel, out_channel, 1)
        else:
            self.residual_layer = nn.Identity()

    def forward(self, pos1, query, pos2, key):
        """
        Inputs:
            pos: (B, 3, N)
            key: (B, in_channel, N)
            query: (B, in_channel, N)
        """

        value = key  # (B, dim, N)
        identity = query
        key = self.conv_key(key)  # (B, dim, N)
        query = self.conv_query(query)
        value = self.conv_value(value)
        b, dim, n = query.shape

        pos1_flipped = pos1.permute(0, 2, 1).contiguous()
        pos2_flipped = pos2.permute(0, 2, 1).contiguous()
        idx_knn = query_knn(self.n_knn, pos2_flipped, pos1_flipped)  # b, N1, k

        key = grouping_operation(key, idx_knn)  # (B, dim, N1, k)
        qk_rel = query.reshape((b, -1, n, 1)) - key

        pos_rel = pos1.reshape((b, -1, n, 1)) - grouping_operation(pos2, idx_knn)  # (B, 3, N, k)
        pos_embedding = self.pos_mlp(pos_rel)  # (B, dim, N, k)

        # attention
        attention = self.attn_mlp(qk_rel + pos_embedding)  # (B, dim, N*up_factor, k)

        # softmax function
        attention = self.scale(attention)

        # knn value is correct
        value = grouping_operation(value, idx_knn) + pos_embedding  # (B, dim, N, k)
        value = self.upsample1(value)  # (B, dim, N*up_factor, k)

        agg = einsum('b c i j, b c i j -> b c i', attention, value)  # (B, dim, N*up_factor)
        y = self.conv_end(agg)  # (B, out_dim, N*up_factor)

        # shortcut
        identity = self.residual_layer(identity)  # (B, out_dim, N)
        identity = self.upsample2(identity)  # (B, out_dim, N*up_factor)

        return y + identity


# seed generator
class SeedGenerator(nn.Module):
    def __init__(self, feat_dim=512, seed_dim=128, n_knn=16, factor=2, attn_channel=True):
        super().__init__()
        self.uptrans = UpTransformer(in_channel=256, out_channel=128, dim=64, n_knn=n_knn, attn_channel=attn_channel,
                                     up_factor=factor, scale_layer=None)
        self.mlp_1 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=seed_dim)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(seed_dim, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, feat, patch_xyz, patch_feat, partial):

        x1 = self.uptrans(patch_xyz, patch_feat, patch_xyz, patch_feat)  # (B, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (B, 128, 256)
        seed = self.mlp_4(x3)  # (B, 3, 256)
        x = fps_subsample(torch.cat([seed.permute(0, 2, 1).contiguous(), partial], dim=1), 512).permute(0, 2,
                                                                                                        1).contiguous()  # b, 3, 512
        return seed, x3, x

class PN(nn.Module):
    def __init__(self, scale, feat_dim=512, dims=128):
        super().__init__()
        self.scale = scale
        self.mlp_1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.EDformer = EDformer(dims)
        self.mlp_2 = MLP_CONV(in_channel=128 * 2 + feat_dim, layer_dims=[512, 256, 128])

    def forward(self, xyz, global_feat):
        b, _, n = xyz.shape
        feat = self.mlp_1(xyz)
        feat = self.EDformer(feat).repeat(1, 1, self.scale)
        feat4cat = [feat, torch.max(feat, 2, keepdim=True)[0].repeat(1, 1, n), global_feat.repeat(1, 1, n)]
        point_feat = self.mlp_2(torch.cat(feat4cat, dim=1))
        return point_feat

class DeConv(nn.Module):
    def __init__(self, up_factor=4):
        super().__init__()
        self.decrease_dim = MLP_CONV(in_channel=128, layer_dims=[64, 32], bn=True)
        self.ps = nn.ConvTranspose1d(32, 128, up_factor, up_factor, bias=False)
        self.mlp_res = MLP_Res(in_dim=128 * 2, hidden_dim=128, out_dim=128)
        self.upper = nn.Upsample(scale_factor=up_factor)
        self.xyz_mlp = MLP_CONV(in_channel=128, layer_dims=[64, 3])

    def forward(self, xyz, feat):
        feat_child = self.ps(self.decrease_dim(feat))
        feat_child = self.mlp_res(torch.cat([feat_child, self.upper(feat)], dim=1))  # b, 128, n*r
        delta = self.xyz_mlp(torch.relu(feat_child))
        new_xyz = self.upper(xyz) + torch.tanh(delta)
        return new_xyz


class FFN(nn.Module):
    def __init__(self, in_dim=128, dim=512, out_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, f):
        return f + self.conv(f)

class PMA_UpBlock(nn.Module):
    def __init__(self, scale, feat_dim=512, dims=128, down_rates=[1, 4, 2], knns=[16, 12, 8], up_factor=4):
        super().__init__()
        self.pn = PN(scale)
        self.inter_crt = CRT(dim_in=128, is_inter=True, down_rates=down_rates, knns=knns)
        # self.ffn = FFN(128, 512, 128)
        self.intra_crt = CRT(dim_in=128, is_inter=False, down_rates=down_rates, knns=knns)
        self.MS = MSMHA(input_dim=128, scales=[1, 2, 4])
        self.deconv = DeConv(up_factor=up_factor)

    def forward(self, p, gf, pre, fps_idxs_1, fps_idxs_2):
        h = self.pn(p, gf)
        g, fps_idxs_q1, fps_idxs_s1 = self.inter_crt([p, h], pre, None, fps_idxs_1)
        # g = self.ffn(g)
        f, _, _ = self.intra_crt([p, g], [p, g], fps_idxs_q1, fps_idxs_q1)
        f_ = self.MS(f.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        f = f + f_
        new_xyz = self.deconv(p, f)
        return new_xyz, f, fps_idxs_q1, fps_idxs_s1

# decoder
class PMA(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.ub0 = PMA_UpBlock(scale=4, feat_dim=512, down_rates=[1, 2, 2], knns=[16, 12, 8], up_factor=1)
        self.ub1 = PMA_UpBlock(scale=4, feat_dim=512, down_rates=[1, 2, 2], knns=[16, 12, 8], up_factor=4)
        self.ub2 = PMA_UpBlock(scale=16, feat_dim=512, down_rates=[1, 4, 2], knns=[16, 12, 8], up_factor=8)

    def forward(self, global_f, p0, p_sd, f_sd):
        p1, f0, p0_fps_idxs_122, _ = self.ub0(p0, global_f, [p_sd, f_sd], None, None)
        p2, f1, p1_fps_idxs_122, _ = self.ub1(p1, global_f, [p0, f0], None, p0_fps_idxs_122)
        p3, _, _______________, _ = self.ub2(p2, global_f, [p1, f1], None, None)

        all_pc = [p_sd.permute(0, 2, 1).contiguous(), p1.permute(0, 2, 1).contiguous(), \
                  p2.permute(0, 2, 1).contiguous(), p3.permute(0, 2, 1).contiguous()]
        return all_pc

@MODELS.register_module()
class MPGLNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.encoder = LGGP()
        self.seed_generator = SeedGenerator()
        self.decoder = PMA()
        self.build_loss_func()
        self.knn_uniform = KNN(k=2, transpose_mode=True)
        self.knn_repulsion = KNN(k=20, transpose_mode=True)
    def forward(self, xyz):
        pp, fp, global_f = self.encoder(xyz.contiguous())
        p_sd, f_sd, p0 = self.seed_generator(global_f, pp, fp, xyz)
        all_pc = []
        all_pc = self.decoder(global_f, p0, p_sd, f_sd)
        all_pc = (xyz, *all_pc)
        return all_pc

    def build_loss_func(self):
        self.loss_func_CD = ChamferDistanceL1()
        self.loss_func_PM = ChamferDistanceL1_PM()


    def get_loss(self, pcds_pred, complete_gt, epoch=1, **kwargs):
        """loss function
        Args
            pcds_pred: List of predicted point clouds, order in [Pc, P1, P2, P3...]
        """
        partial_input = pcds_pred[0]

        partial_input, Pc, P1, P2, P3 = pcds_pred

        gt_2 = fps(complete_gt, P2.shape[1])
        gt_1 = fps(gt_2, P1.shape[1])
        gt_c = fps(gt_1, Pc.shape[1])

        cdc = calc_cd_like_InfoV2(Pc, gt_c)
        cd1 = calc_cd_like_InfoV2(P1, gt_1)
        cd2 = calc_cd_like_InfoV2(P2, gt_2)
        cd3 = calc_cd_like_InfoV2(P3, complete_gt)

        partial_matching = calc_cd_one_side_like_InfoV2(partial_input, P3)

        loss_coarse = (cdc + cd1 + cd2 + partial_matching) * 1e-3
        loss_fine = cd3 * 1e-3

        return loss_coarse, loss_fine


# testing
if __name__ == '__main__':
    model = MPGLNet().cuda()
    # print(model)
    pc = torch.rand(3, 2048, 3).cuda()
    re = model(pc)
    for i in re:
        print(i.shape)