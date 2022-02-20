import torch
import ransac_voting_3d


def ransac_voting_layer(cloud, pred_t, round_hyp_num=128, inlier_thresh=0.99, confidence=0.99, max_iter=20, min_num=5, max_num=30000):
    """
    Args:
        cloud:  [b, pn, 3] - x, y, z
        pred_t: [b, pn, 3] - dx, dy, dz
        round_hyp_num: number of hypothesis per round
        inlier_thresh: voting threshold, cosine angle
    Returns:
        batch_win_pts: [b, 3] - x, y, z
        batch_inliers: [b, pn]

    """
    b, pn, _ = pred_t.shape
    vn = 1  # only voting for center
    batch_win_pts = []
    batch_inliers = []
    for bi in range(b):
        hyp_num = 0
        foreground_num = torch.tensor(pn, device=pred_t.device)
        cur_mask = torch.ones([pn, 1], dtype=torch.uint8, device=pred_t.device)
        # if too few points, just skip it
        if foreground_num < min_num:
            win_pts = torch.zeros([1, 3], dtype=torch.float32, device=pred_t.device)
            inliers = torch.zeros([1, pn], dtype=torch.uint8, device=pred_t.device)
            batch_win_pts.append(win_pts)
            batch_inliers.append(inliers)
            continue
        # if too many inliers, randomly downsample
        if foreground_num > max_num:
            selection = torch.zeros(cur_mask.shape, dtype=torch.float32, device=pred_t.device).uniform_(0, 1)
            selected_mask = (selection < (max_num / foreground_num.float()))
            cur_mask *= selected_mask

        tn = torch.sum(cur_mask)
        coords = cloud[bi, :, :].masked_select(cur_mask).view([tn, 3])    # [tn, 3]
        direct = pred_t[bi, :, :].masked_select(cur_mask).view([tn, vn, 3])   # [tn, vn, 3]

        # RANSAC
        idxs = torch.zeros([round_hyp_num, vn, 2], dtype=torch.int32, device=pred_t.device).random_(0, direct.shape[0])
        all_win_ratio = torch.zeros([vn], dtype=torch.float32, device=pred_t.device)
        all_win_pts = torch.zeros([vn, 3], dtype=torch.float32, device=pred_t.device)
        cur_iter = 0
        while True:
            # generate hypothesis
            cur_hyp_pts = ransac_voting_3d.generate_hypothesis(direct, coords, idxs)  # [hn, vn, 3]
            # voting for hypothesis
            cur_inlier = torch.zeros([round_hyp_num, vn, tn], dtype=torch.uint8, device=pred_t.device)  # [hn, vn, tn]
            ransac_voting_3d.voting_for_hypothesis(direct, coords, cur_hyp_pts, cur_inlier, inlier_thresh)
            # find max
            cur_inlier_counts = torch.sum(cur_inlier, 2)                   # [hn, vn]
            cur_win_counts, cur_win_idx = torch.max(cur_inlier_counts, 0)  # [vn]
            cur_win_pts = cur_hyp_pts[cur_win_idx, torch.arange(vn)]
            cur_win_ratio = cur_win_counts.float() / tn
            # update best point
            larger_mask = all_win_ratio < cur_win_ratio
            all_win_pts[larger_mask, :] = cur_win_pts[larger_mask, :]
            all_win_ratio[larger_mask] = cur_win_ratio[larger_mask]
            # check confidence
            hyp_num += round_hyp_num
            cur_iter += 1
            cur_min_ratio = torch.min(all_win_ratio)
            if (1 - (1 - cur_min_ratio ** 2) ** hyp_num) > confidence or cur_iter > max_iter:
                break
        # compute mean intersection
        all_inlier = torch.zeros([1, vn, tn], dtype=torch.uint8, device=pred_t.device)
        all_win_pts = torch.unsqueeze(all_win_pts, 0)   # [1, vn, 3]
        ransac_voting_3d.voting_for_hypothesis(direct, coords, all_win_pts, all_inlier, inlier_thresh)
        all_inlier = all_inlier.view([tn, 1])   # because of vn = 1
        all_inlier_count = torch.sum(all_inlier)
        inlier_coords = coords.masked_select(all_inlier).view([all_inlier_count, 3, 1])
        # normalize directions
        inlier_direct = torch.squeeze(direct, 1)    # [tn, 3]
        inlier_direct = inlier_direct / torch.norm(inlier_direct, dim=1, keepdim=True)
        inlier_direct = inlier_direct.masked_select(all_inlier).view([all_inlier_count, 3, 1])
        S = torch.bmm(inlier_direct, inlier_direct.permute(0, 2, 1)) - \
                torch.unsqueeze(torch.eye(3, device=pred_t.device), 0).repeat(all_inlier_count, 1, 1)
        A = torch.sum(S, 0)    # [3, 3]
        b = torch.sum(torch.bmm(S, inlier_coords), 0)   # [3, 1]
        # voting result
        win_pts = torch.matmul(torch.inverse(A), b).permute(1, 0)     # [1, 3]
        batch_win_pts.append(win_pts)
        # mask
        inliers = torch.squeeze(cur_mask, 1).repeat(vn, 1)   # [vn, pn]
        index = torch.squeeze(cur_mask, 1).nonzero().view([tn]).repeat(vn, 1)   # [vn, tn]
        inliers.scatter_(1, index, all_inlier.permute(1, 0))
        batch_inliers.append(inliers)
    batch_win_pts = torch.cat(batch_win_pts)
    batch_inliers = torch.squeeze(torch.cat(batch_inliers), 1)

    return batch_win_pts, batch_inliers
