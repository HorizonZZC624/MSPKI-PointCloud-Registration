
import torch
import torch.nn as nn



def solve_local_rotations(Am, Bm, weights=None, weight_threshold=0):
    pass








    bs = Am.shape[0]
    if weights is None:
        weights = torch.ones_like(Am[:, :, 0])
    weights[weights < weight_threshold] = 0


    Weight = torch.diag_embed(weights)
    H = Am.permute(0, 2, 1) @ Weight @ Bm


    U, S, Vt = torch.svd(H.cpu())
    U, S, Vt = U.to(weights.device), S.to(weights.device), Vt.to(weights.device)
    delta_UV = torch.det(Vt @ U.permute(0, 2, 1))
    eye = torch.eye(3)[None, :, :].repeat(bs, 1, 1).to(Am.device)
    eye[:, -1, -1] = delta_UV
    R = Vt @ eye @ U.permute(0, 2, 1)
    return R

def weighted_procrustes(
    src_points,
    ref_points,
    weights=None,
    weight_thresh=0.0,
    eps=1e-5,
    return_transform=False,
):
    pass
















    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        if weights is not None:
            weights = weights.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    if weights is None:
        weights = torch.ones_like(src_points[:, :, 0])
    weights = torch.where(torch.lt(weights, weight_thresh), torch.zeros_like(weights), weights)
    weights = weights / (torch.sum(weights, dim=1, keepdim=True) + eps)
    weights = weights.unsqueeze(2)

    src_centroid = torch.sum(src_points * weights, dim=1, keepdim=True)
    ref_centroid = torch.sum(ref_points * weights, dim=1, keepdim=True)
    src_points_centered = src_points - src_centroid
    ref_points_centered = ref_points - ref_centroid

    H = src_points_centered.permute(0, 2, 1) @ (weights * ref_points_centered)
    U, _, V = torch.svd(H.cpu())
    Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t
def cal_leading_eigenvector( M, method='power'):
    pass







    if method == 'power':

        leading_eig = torch.ones_like(M[:, :, 0:1])
        leading_eig_last = leading_eig
        for i in range(10):
            leading_eig = torch.bmm(M, leading_eig)
            leading_eig = leading_eig / (torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6)
            if torch.allclose(leading_eig, leading_eig_last):
                break
            leading_eig_last = leading_eig
        leading_eig = leading_eig.squeeze(-1)
        return leading_eig
    elif method == 'eig':
        e, v = torch.symeig(M, eigenvectors=True)
        leading_eig = v[:, :, -1]
        return leading_eig
    else:
        exit(-1)
def soft_weight(src_points, ref_points, valid=None):
    knn_M = torch.norm(src_points[:, :, None, :] - src_points[:, None, :, :], 2, -1) - torch.norm(ref_points[:, :, None, :] - ref_points[:, None, :, :], 2, -1)
    knn_M = torch.clamp(1 - knn_M ** 2 / 0.3 ** 2, min=0)
    if valid is not None:
        knn_M.masked_fill_(~(valid * valid.permute(0, 2, 1)), 0.)
    knn_M[:, torch.arange(knn_M.shape[1]), torch.arange(knn_M.shape[1])] = 0
    weights = cal_leading_eigenvector(knn_M)
    return weights


def procrustes(
    src_points,
    ref_points,
    valid_points=None,
    return_transform=False,
    src_feats=None,
    ref_feats=None
):
    pass
















    if src_points.ndim == 2:
        src_points = src_points.unsqueeze(0)
        ref_points = ref_points.unsqueeze(0)
        valid_points = valid_points.unsqueeze(0)
        squeeze_first = True
    else:
        squeeze_first = False

    batch_size = src_points.shape[0]
    valid_points = valid_points.unsqueeze(2)



    src_centroid = torch.sum(src_points * valid_points, dim=1, keepdim=True) / valid_points.sum(dim=1, keepdim=True)
    ref_centroid = torch.sum(ref_points * valid_points, dim=1, keepdim=True) / valid_points.sum(dim=1, keepdim=True)
    src_points_centered = src_points - src_centroid
    ref_points_centered = ref_points - ref_centroid

    if src_feats is not None:




        src_points_centered = src_feats
        ref_points_centered = ref_feats
        valid_points = soft_weight(src_feats, ref_feats).unsqueeze(2)


    H = src_points_centered.permute(0, 2, 1) @ (valid_points * ref_points_centered)
    U, _, V = torch.svd(H.cpu())
    Ut, V = U.transpose(1, 2).cuda(), V.cuda()
    eye = torch.eye(3).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
    eye[:, -1, -1] = torch.sign(torch.det(V @ Ut))
    R = V @ eye @ Ut

    t = ref_centroid.permute(0, 2, 1) - R @ src_centroid.permute(0, 2, 1)
    t = t.squeeze(2)

    if return_transform:
        transform = torch.eye(4).unsqueeze(0).repeat(batch_size, 1, 1).cuda()
        transform[:, :3, :3] = R
        transform[:, :3, 3] = t
        if squeeze_first:
            transform = transform.squeeze(0)
        return transform
    else:
        if squeeze_first:
            R = R.squeeze(0)
            t = t.squeeze(0)
        return R, t


class WeightedProcrustes(nn.Module):
    def __init__(self, weight_thresh=0.0, eps=1e-5, return_transform=False):
        super(WeightedProcrustes, self).__init__()
        self.weight_thresh = weight_thresh
        self.eps = eps
        self.return_transform = return_transform

    def forward(self, src_points, tgt_points, weights=None):
        return weighted_procrustes(
            src_points,
            tgt_points,
            weights=weights,
            weight_thresh=self.weight_thresh,
            eps=self.eps,
            return_transform=self.return_transform,
        )

