import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG
import ipdb


# os.environ["CUDA_VISIBLE_DEVICES"]="1"

def KL(mu, logvar):
    mu = mu.view(mu.shape[0], -1)
    logvar = logvar.view(logvar.shape[0], -1)
    loss = 0.5 * torch.sum(mu * mu + torch.exp(logvar) - 1 - logvar, 1)
    # high star implementation
    # torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
    loss = torch.mean(loss)
    return loss


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,  # !1024
                radius=0.1,
                nsample=32,
                mlp=[self.hparams["input_feat"], 32, 32, 64],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=32,
                mlp=[64, 64, 64, 128],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=64,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
            )
        )
        self.SA_modules.append(
            PointnetSAModule(
                npoint=16,
                radius=0.8,
                nsample=32,
                mlp=[256, 256, 256, 512],
                use_xyz=True,
            )
        )

        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[128 + self.hparams["input_feat"], 128, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 64, 256, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 128, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + 256, 256, 256]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, self.hparams['feat_dim'], kernel_size=1, bias=False),
            nn.BatchNorm1d(self.hparams['feat_dim']),
            nn.ReLU(True),
        )

    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        xyz -= xyz.mean(axis=1, keepdim=True)
        xyz = xyz / (xyz.var(axis=1, keepdim=True) + 1e-8)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_directions=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_directions = num_directions
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.mlp = nn.Linear(self.num_directions * self.hidden_size, self.output_size)

    def forward(self, input_seq):
        batch_size, seq_len = input_seq.shape[0], input_seq.shape[1]
        h_0 = torch.Tensor(torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)).cuda()
        c_0 = torch.Tensor(torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)).cuda()
        # output(batch_size, seq_len, num_directions * hidden_size)
        output, _ = self.lstm(input_seq, (h_0, c_0))
        out = self.mlp(output[:, -1, :])
        return out


class ActionEncoder(nn.Module):
    def __init__(self, action_feat_dim, action_dim):
        super(ActionEncoder, self).__init__()
        self.action_feat_dim = action_feat_dim
        self.action_dim = action_dim

        self.mlp = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.Linear(128, 128),
            nn.Linear(128, action_feat_dim)
        )

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x.view(batch_size, self.action_dim))
        return x


class ActorEncoder(nn.Module):
    def __init__(self, input_feat_dim, feat_dim=128):
        super(ActorEncoder, self).__init__()

        self.mlp1 = nn.Linear(input_feat_dim, feat_dim)
        self.mlp2 = nn.Linear(feat_dim, feat_dim)
        self.get_mu = nn.Linear(feat_dim, feat_dim)
        self.get_logvar = nn.Linear(feat_dim, feat_dim)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

        self.z_dim = feat_dim

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, feat):
        net = F.leaky_relu(self.mlp1(feat))
        net = self.mlp2(net)
        mu = self.get_mu(net)
        logvar = self.get_logvar(net)
        noise = torch.Tensor(torch.randn(*mu.shape)).cuda()
        z = mu + torch.exp(logvar / 2) * noise
        return z, mu, logvar


class FrankaStateEncoder(nn.Module):
    def __init__(self, state_dim, state_feat_dim) -> None:
        super(FrankaStateEncoder, self).__init__()
        self.state_dim = state_dim
        self.state_feat_dim = state_feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Linear(128, 128),
            nn.Linear(128, state_feat_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x.view(batch_size, self.state_dim))
        return x


class ActorDecoder(nn.Module):
    def __init__(self, input_feat_dim, action_dim=9):
        super(ActorDecoder, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(input_feat_dim, 512),
            nn.Linear(512, 256),
            nn.Linear(256, action_dim)
        )
        self.action_dim = action_dim

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, z_all, feat):
        batch_size = z_all.shape[0]
        x = torch.cat([z_all, feat], dim=-1)
        x = self.mlp(x)
        x = x.view(batch_size, self.action_dim)
        return x


class ActorNetwork(nn.Module):
    def __init__(self, pc_input_feat=3, pc_feat_dim=128, state_feat_dim=128, action_feat_dim=128, action_dim=9,
                 state_dim=18, lbd_xyz=1.0, lbd_rpy=1.0, lbd_kl=1.0):
        super(ActorNetwork, self).__init__()

        self.pointnet2 = PointNet2SemSegSSG({'input_feat': pc_input_feat, 'feat_dim': pc_feat_dim})
        self.actionencoder = ActionEncoder(action_feat_dim=action_feat_dim, action_dim=action_dim)
        self.stateencoder = FrankaStateEncoder(state_feat_dim=state_feat_dim, state_dim=state_dim)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')
        self.L2Loss = nn.MSELoss(reduction="none")

        self.action_dim = action_dim
        self.state_feat_dim = state_feat_dim
        self.state_dim = state_dim
        self.z_dim = 128

        self.lbd_xyz = lbd_xyz
        self.lbd_rpy = lbd_rpy
        self.lbd_kl = lbd_kl

        out_feature_dim = pc_feat_dim + state_feat_dim + action_feat_dim

        self.encoder = ActorEncoder(input_feat_dim=out_feature_dim)
        self.decoder = ActorDecoder(input_feat_dim=pc_feat_dim + state_feat_dim + self.z_dim, action_dim=action_dim)

        self.mlp = nn.Sequential(
            nn.Linear(out_feature_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(32, action_dim),
        )

    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, whole_feats: B x F x N
    def forward(self, pcs, state, gt_action):
        B = pcs.shape[0]
        # print("shape", pcs.shape, contact_point.shape)

        # pc = pcs[:, :, :3]
        # input_pcs = torch.cat([pc, pcs], dim=-1)
        # whole_feats = self.pointnet2(input_pcs)
        # pcs_feat = torch.max(whole_feats, dim=2)[0]
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        pcs_feat = whole_feats[:, :, 0]
        action_feat = self.actionencoder(gt_action)
        state_feat = self.stateencoder(state)

        encode_feat = torch.cat((action_feat, pcs_feat, state_feat), dim=1)
        decode_feat = torch.cat((pcs_feat, state_feat), dim=-1)

        z_all, mu, logvar = self.encoder(encode_feat)
        action = self.decoder(z_all, decode_feat)

        return action, mu, logvar

    def get_loss(self, pcs, state, gt_action):
        B = state.shape[0]
        pred_action, mu, logvar = self.forward(pcs, state, gt_action)

        xyz_loss = self.L1Loss(pred_action[:, :3], gt_action[:, :3])
        rpy_loss = self.get_6d_rot_loss(pred_action[:, 3:], gt_action[:, 3:])

        kl_loss = KL(mu, logvar)

        # ipdb.set_trace()
        losses = {}
        losses['xyz'] = xyz_loss.mean()
        losses['rpy'] = rpy_loss.mean()
        losses['kl'] = kl_loss
        losses['tot'] = losses['xyz'] * self.lbd_xyz + losses['rpy'] * self.lbd_rpy + losses['kl'] * self.lbd_kl

        return losses

    # input sz bszx3x2
    def bgs(self, d6s):
        bsz = d6s.shape[0]
        b1 = F.normalize(d6s[:, :, 0], p=2, dim=1)
        a2 = d6s[:, :, 1]
        # b2 = F.normalize(a2 - torch.bmm(b1.view(bsz, 1, -1), a2.view(bsz, -1, 1)).view(bsz, 1) * b1, p=2, dim=1)
        b2 = F.normalize(a2 - (b1 * a2).sum(-1, keepdim=True) * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        # return torch.stack([b1, b2, b3], dim=1).permute(0, 2, 1)
        return torch.stack([b1, b2, b3], dim=1)

    # batch geodesic loss for rotation matrices
    def bgdR(self, Rgts, Rps):
        Rds = torch.bmm(Rgts.permute(0, 2, 1), Rps)
        Rt = torch.sum(Rds[:, torch.eye(3).bool()], 1)  # batch trace
        # necessary or it might lead to nans and the likes
        theta = torch.clamp(0.5 * (Rt - 1), -1 + 1e-6, 1 - 1e-6)
        return torch.acos(theta)

    # 6D-Rot loss
    # input sz bszx6
    def get_6d_rot_loss(self, pred_6d, gt_6d):
        # ipdb.set_trace()
        pred_Rs = self.bgs(pred_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        gt_Rs = self.bgs(gt_6d.reshape(-1, 2, 3).permute(0, 2, 1))
        # pred_Rs = self.bgs(pred_6d.reshape(-1, 3, 2))
        # gt_Rs = self.bgs(gt_6d.reshape(-1, 3, 2))
        theta = self.bgdR(gt_Rs, pred_Rs)
        return theta

    def sample(self, pcs, state):
        batch_size = pcs.shape[0]
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        pcs_feat = whole_feats[:, :, 0]

        state_feat = self.stateencoder(state)
        decode_feat = torch.cat((pcs_feat, state_feat), dim=-1)

        z_all = torch.Tensor(torch.randn(batch_size, self.z_dim)).cuda()
        pred_action = self.decoder(z_all, decode_feat)

        pred_pos = pred_action[:, :3]
        pred_rot = self.bgs(pred_action[:, 3:].view(-1, 2, 3).permute(0, 2, 1))

        return pred_pos.detach(), pred_rot.detach()

    def sample_n(self, pcs, state, rvs=100):
        batch_size = pcs.shape[0]
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)
        pcs_feat = whole_feats[:, :, 0]
        state_feat = self.stateencoder(state)

        pcs_feat = pcs_feat.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
        state_feat = state_feat.unsqueeze(dim=1).repeat(1, rvs, 1).view(batch_size * rvs, -1)
        decode_feat = torch.cat((pcs_feat, state_feat), dim=-1)

        z_all = torch.Tensor(torch.randn(batch_size * rvs, self.z_dim)).cuda()
        pred_action = self.decoder(z_all, decode_feat)

        pred_pos = pred_action[:, :3]
        pred_rot = self.bgs(pred_action.view(-1, 3, 2))

        return pred_pos, pred_rot