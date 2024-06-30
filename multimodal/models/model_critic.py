import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
# https://github.com/erikwijmans/Pointnet2_PyTorch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from pointnet2.models.pointnet2_ssg_cls import PointNet2ClassificationSSG


class PointNet2SemSegSSG(PointNet2ClassificationSSG):
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=1024,    #!1024
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
        xyz = xyz/(xyz.var(axis=1, keepdim=True)+1e-8)

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

class FrankaStateEncoder(nn.Module):
    def __init__(self, state_dim, state_feat_dim) -> None:
        super(FrankaStateEncoder, self).__init__()
        self.state_dim = state_dim
        self.state_feat_dim = state_feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_feat_dim)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x.view(batch_size, self.state_dim))
        return x

class Hand_Root_Encoder(nn.Module):
    def __init__(self):
        super(Hand_Root_Encoder, self).__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(3,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(3,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64)
        )
    
    def forward(self, x1, x2):
        batch_size = x1.shape[0]
        x1 = self.mlp1(x1.view(batch_size, 3))
        x2 = self.mlp2(x2.view(batch_size, 3))
        return torch.cat([x1,x2], dim=-1)

class FrankaIKEncoder(nn.Module):
    def __init__(self, ik_dim, ik_feat_dim) -> None:
        super(FrankaIKEncoder, self).__init__()
        self.ik_dim = ik_dim
        self.ik_feat_dim = ik_feat_dim

        self.mlp = nn.Sequential(
            nn.Linear(ik_dim, 128),
            nn.Linear(128, 128),
            nn.Linear(128, ik_feat_dim)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x.view(batch_size, self.ik_dim))
        return x

class ActionEncoder(nn.Module):
    def __init__(self, action_feat_dim, action_dim):
        super(ActionEncoder, self).__init__()
        self.action_feat_dim = action_feat_dim
        self.action_dim = action_dim

        self.mlp = nn.Sequential(
            nn.Linear(action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_feat_dim)
        )

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.mlp(x.view(batch_size, self.action_dim))
        return x

class Critic(nn.Module):
    def __init__(self, pc_feat_dim, state_dim, state_feat_dim, action_dim, action_feat_dim):
        super(Critic, self).__init__()

        self.state_encoder = FrankaStateEncoder(state_dim, state_feat_dim)
        # self.ik_encoder = FrankaIKEncoder(ik_dim, ik_feat_dim)
        self.action_encoder = ActionEncoder(action_feat_dim, action_dim)

        self.mlp1 = nn.Linear(pc_feat_dim + state_feat_dim + action_feat_dim, 128)
        self.mlp2 = nn.Linear(128, 1)

        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, pc_feat, state, action):
        state_feat = self.state_encoder(state)
        # ik_feat = self.ik_encoder(ik)
        action_feat = self.action_encoder(action)
        net = torch.cat([pc_feat, state_feat, action_feat], dim=-1)
        net = F.leaky_relu(self.mlp1(net))
        pred_score = self.mlp2(net).squeeze(-1)
        return pred_score

    # cross entropy loss
    def get_ce_loss(self, pred_logits, gt_labels):
        loss = self.BCELoss(pred_logits, gt_labels.float())
        return loss

    # cross entropy loss
    def get_l1_loss(self, pred_logits, gt_labels):
        loss = self.L1Loss(pred_logits, gt_labels)
        return loss
'''
包含hand_pos 以及 root pos
'''
class Critic_more(nn.Module):
    def __init__(self, pc_feat_dim, state_dim, state_feat_dim, action_dim, action_feat_dim):
        super(Critic_more, self).__init__()

        self.state_encoder = FrankaStateEncoder(state_dim, state_feat_dim)
        # self.ik_encoder = FrankaIKEncoder(ik_dim, ik_feat_dim)
        self.action_encoder = ActionEncoder(action_feat_dim, action_dim)

        self.hand_root_encoder = Hand_Root_Encoder()

        self.score = nn.Sequential(
            nn.Linear(pc_feat_dim + state_feat_dim + action_feat_dim + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.BCELoss = nn.BCEWithLogitsLoss(reduction='none')
        self.L1Loss = nn.L1Loss(reduction='none')

    # pixel_feats B x F, query_fats: B x 6
    # output: B
    def forward(self, pc_feat, state, action, hand_pos, root_pos):
        state_feat = self.state_encoder(state)
        # ik_feat = self.ik_encoder(ik)
        action_feat = self.action_encoder(action)
        pos_feat = self.hand_root_encoder(hand_pos, root_pos)
        net = torch.cat([pc_feat, state_feat, action_feat, pos_feat], dim=-1)
        net = self.score(net).squeeze(-1)
        return net

    # cross entropy loss
    def get_ce_loss(self, pred_logits, gt_labels):
        loss = self.BCELoss(pred_logits, gt_labels.float())
        return loss

    # cross entropy loss
    def get_l1_loss(self, pred_logits, gt_labels):
        loss = self.L1Loss(pred_logits, gt_labels)
        return loss


class CriticNetwork(nn.Module):
    def __init__(self, pc_input_feat, pc_feat_dim, state_dim, state_feat_dim, action_dim, action_feat_dim, more = False):
        super(CriticNetwork, self).__init__()

        self.pointnet2 = PointNet2SemSegSSG({'input_feat': pc_input_feat, 'feat_dim': pc_feat_dim})
        if more:
            self.critic = Critic_more(pc_feat_dim, state_dim, state_feat_dim, action_dim, action_feat_dim)
        else:
            self.critic = Critic(pc_feat_dim, state_dim, state_feat_dim, action_dim, action_feat_dim)

        
    # pcs: B x N x 3 (float), with the 0th point to be the query point
    # pred_result_logits: B, whole_feats: B x F x N
    def forward(self, pcs, state, action):
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]

        # input_queries = torch.cat([dirs1, dirs2], dim=1)

        pred_score = self.critic(net, state, action)

        return pred_score
    
    def forward_more(self, pcs, state, action, hand_pos, root_pos):
        pcs = pcs.repeat(1, 1, 2)
        whole_feats = self.pointnet2(pcs)

        net = whole_feats[:, :, 0]

        pred_score = self.critic(net, state, action, hand_pos, root_pos)
        return pred_score

    def get_loss(self, pcs, state, action, gt_score, hand_pos=None, root_pos=None, more=False):
        if more:
            pred_score = self.forward_more(pcs, state, action, hand_pos, root_pos)
        else:
            pred_score = self.forward(pcs, state, action)
        # ipdb.set_trace()
        loss = self.critic.get_l1_loss(pred_score, gt_score)
        losses = {}
        losses["tot"] = loss.mean()
        
        return losses

    def inference(self, pcs, state, action, hand_pos=None, root_pos=None, more=False):
        if more:
            score = self.forward_more(pcs, state, action, hand_pos, root_pos)
            return score.detach()
        else:
            score = self.forward(pcs, state, action)
            return score.detach()
