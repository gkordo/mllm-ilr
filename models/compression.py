import torch
import torch.nn.functional as F
from kmeans_pytorch import kmeans

START_IMAGE = 151652
IMAGE_TOKEN = 151655
END_IMAGE = 151653


def sampling_and_pooling_2D_segment(
    start: int,
    end: int,
    grid_thw: torch.Tensor,
    embeds: torch.Tensor,
    embed_dim: int,
    mode: str,  # "sampling" | "pooling"
    replacement: str,  # "skip" | "zero" | "replicate"
):
    t, H, W = grid_thw.tolist()
    assert (
        t == 1
        and mode in {"sampling", "pooling"}
        and replacement in {"skip", "zero", "replicate"}
    )

    h, w = H // 2, W // 2
    x = embeds[:, start + 1 : end].reshape(1, h, w, embed_dim).permute(0, 3, 1, 2)
    hc, wc = (h if h % 2 == 0 else h - 1), (w if w % 2 == 0 else w - 1)
    xc = x[:, :, :hc, :wc]
    B, C = 1, embed_dim
    device = embeds.device

    if mode == "sampling":
        base = xc[:, :, ::2, ::2]
    else:  # pooling
        base = F.avg_pool2d(xc, kernel_size=2, stride=2)

    if replacement == "skip":
        out = base
        rows = torch.arange(0, hc, 2, device=device)
        cols = torch.arange(0, wc, 2, device=device)
        rr, cc = torch.meshgrid(rows, cols, indexing="ij")
        lin = (rr.reshape(-1) * w + cc.reshape(-1)) + (start + 1)
        mask = torch.zeros(embeds.size(1), dtype=torch.bool, device=device)
        mask[lin] = True
        return mask, out.permute(0, 2, 3, 1).reshape(B, -1, C)

    if replacement == "zero":
        out = torch.zeros_like(x)
        out[:, :, ::2, ::2] = base
    else:  # replicate
        out = x.clone()
        out[:, :, :hc, :wc] = base.repeat_interleave(2, 2).repeat_interleave(2, 3)[:, :, :hc, :wc]
    
    mask = torch.zeros(embeds.size(1), dtype=torch.bool)
    mask[start + 1 : end] = True
    return mask, out.permute(0, 2, 3, 1).reshape(B, -1, C)


def sampling_and_pooling_2D(
    segments, grids, embeds, ids, attn, pos, mode, replacement, asym
):
    new_embeds, new_ids, new_attn, new_pos = [], [], [], []
    last_end = 0
    if asym:
        segments = [segments[-1]]
        grids = grids[-1:]
    for (start, end), grid in zip(segments, grids):
        # Keep tokens before the image segment
        if start > last_end:
            new_embeds.append(embeds[:, last_end : start + 1])
            new_ids.append(ids[:, last_end : start + 1])
            new_attn.append(attn[:, last_end : start + 1])
            new_pos.append(pos[:, :, last_end : start + 1])

        # Process image segment
        mask, pooled_seq = sampling_and_pooling_2D_segment(
            start, end, grid, embeds, embeds.size(-1), mode, replacement
        )
        new_embeds.append(pooled_seq)
        new_ids.append(ids[:, mask])
        new_attn.append(attn[:, mask])
        new_pos.append(pos[:, :, mask])

        last_end = end

    # Keep tokens after the last image segment
    if last_end < embeds.size(1):
        new_embeds.append(embeds[:, last_end:])
        new_ids.append(ids[:, last_end:])
        new_attn.append(attn[:, last_end:])
        new_pos.append(pos[:, :, last_end:])

    return (
        torch.cat(new_embeds, 1),
        torch.cat(new_ids, 1),
        torch.cat(new_attn, 1),
        torch.cat(new_pos, 2),
    )


def greedy_grouping(seg_embeds, group_size=4):
    n = seg_embeds.shape[1]
    embeds = seg_embeds[0]  # (n, D)
    norm_embeds = F.normalize(embeds, p=2, dim=1)
    cos_sim = norm_embeds @ norm_embeds.T  # (n, n)

    # Mask diagonal (self-similarity)
    cos_sim.fill_diagonal_(0)

    assigned = torch.zeros(n, dtype=torch.bool)
    groups = []

    while assigned.sum() < n:
        unassigned_idx = (~assigned).nonzero(as_tuple=False).flatten()
        sub_sim = cos_sim[unassigned_idx][:, unassigned_idx]

        flat_index = torch.argmax(sub_sim).item()  # convert tensor to int
        i, j = divmod(flat_index, sub_sim.shape[1])
        i, j = unassigned_idx[i], unassigned_idx[j]

        # Form group
        group = [i, j]
        # Add two more most similar unassigned embeddings
        unassigned_pool = [idx.item() for idx in unassigned_idx if idx not in group]
        if len(unassigned_pool) > 0:
            sims_to_group = cos_sim[unassigned_pool][:, group].sum(dim=1)
            top_k = min(group_size - 2, len(unassigned_pool))
            best_indices = torch.topk(sims_to_group, top_k).indices
            group += [unassigned_pool[idx] for idx in best_indices]

        groups.append(group)
        assigned[torch.tensor(group, dtype=torch.long)] = True

    return groups


def group_segment(inputs_embeds, start, end, group_size, reduce="mean"):
    assert reduce in {
        "mean",
        "nearest",
        "median",
    }, "reduce must be 'mean', 'nearest', or 'median'"
    seg_embeds = inputs_embeds[:, start + 1 : end, :]
    B, L, D = seg_embeds.shape
    assert B == 1, "Batch size 1 only"

    groups = greedy_grouping(seg_embeds, group_size=group_size)
    for group in groups:
        group_tensor = seg_embeds[0, group, :]

        if reduce == "mean":
            rep = group_tensor.mean(dim=0, keepdim=True)  # (1, D)

        elif reduce == "nearest":
            mean_vec = group_tensor.mean(dim=0, keepdim=True)  # (1, D)
            distances = torch.norm(group_tensor - mean_vec, dim=1)  # (g,)
            closest_idx = torch.argmin(distances).item()
            rep = group_tensor[closest_idx : closest_idx + 1]  # (1, D)

        else:  # "median"
            rep = group_tensor.median(dim=0, keepdim=True).values  # (1, D)

        seg_embeds[0, group, :] = rep

    inputs_embeds[:, start + 1 : end, :] = seg_embeds
    return inputs_embeds


def cluster_segment(inputs_embeds, start, end, n_clusters, reduce="mean"):
    """
    Cluster embeddings in the range [start:end) and replace each
    embedding in a cluster with the cluster's mean embedding.
    """
    seg_embeds = inputs_embeds[:, start + 1 : end, :]  # (1, L, D)
    B, L, D = seg_embeds.shape
    assert B == 1, "Batch size 1 only"

    if n_clusters < L:
        # Run k-means on the segment
        labels, _ = kmeans(
            X=seg_embeds[0].detach(),
            num_clusters=n_clusters,
            # num_clusters=L//4,
            distance="euclidean",
            device=seg_embeds.device,
            tqdm_flag=False,
        )  # labels: (L,)

        # Compute means per cluster
        s = 0
        for cluster_id in range(n_clusters):
            mask = labels == cluster_id
            s += mask.sum().item()
            if mask.any():
                if reduce == "mean":
                    cluster_mean = seg_embeds[0, mask].mean(dim=0, keepdim=True)  # (1, D)
                    seg_embeds[0, mask] = cluster_mean
                else:
                    cluster_median = (
                        seg_embeds[0, mask].median(dim=0, keepdim=True).values
                    )  # (1, D)
                    seg_embeds[0, mask] = cluster_median
        # print(start, end, end-start, len(torch.unique(labels)), s)

    # Put back into original tensor
    inputs_embeds[:, start + 1 : end, :] = seg_embeds
    return inputs_embeds


def grouping_and_clustering(
    inputs_embeds,
    segments,
    mode="clustering",  # "grouping" | "clustering"
    group_size=4,  # used when mode="grouping"
    n_clusters=70,  # used when mode="clustering"
    reduce="mean",  # "mean" | "nearest" | "median" (passed through)
    asym=False,
):
    """
    Apply either grouping or clustering to one or two disjoint segments of inputs_embeds.
    """

    first_start, first_end = segments[0]
    second_start, second_end = segments[1]
    # print(inputs_embeds.shape)

    if mode not in {"grouping", "clustering"}:
        raise ValueError("mode must be 'grouping' or 'clustering'")

    if mode == "grouping":
        # Validate args for grouping
        if group_size is None or group_size <= 0:
            raise ValueError(
                "group_size must be a positive integer for mode='grouping'"
            )

        # Optional first segment
        if not asym:
            inputs_embeds = group_segment(
                inputs_embeds,
                first_start,
                first_end,
                group_size=group_size,
                reduce=reduce,
            )

        # Second segment
        inputs_embeds = group_segment(
            inputs_embeds,
            second_start,
            second_end,
            group_size=group_size,
            reduce=reduce,
        )

    elif mode == "clustering":
        # Validate args for clustering
        if n_clusters is None or n_clusters <= 0:
            raise ValueError(
                "n_clusters must be a positive integer for mode='clustering'"
            )

        # Optional first segment
        if not asym:
            inputs_embeds = cluster_segment(
                inputs_embeds, first_start, first_end, n_clusters, reduce
            )

        # Second segment
        inputs_embeds = cluster_segment(
            inputs_embeds, second_start, second_end, n_clusters, reduce
        )

    return inputs_embeds


def pq_segment(x, start, end, pq_obj):
    if start == end:
        return x
    seg = x[:, start + 1 : end, :][0]  # [M, D]
    seg_np = seg.detach().cpu().float().numpy()
    codes = pq_obj.sa_encode(seg_np)
    rec_np = pq_obj.sa_decode(codes)
    rec = torch.from_numpy(rec_np).to(device=x.device, dtype=x.dtype).unsqueeze(0)
    out = x.clone()
    out[:, start + 1 : end, :] = rec
    return out


def product_quantization(inputs_embeds, segments, pq_obj, asym=False):
    """
    Apply Product Quantization to specific segments of [1, N, D] inputs_embeds.

    Args:
        inputs_embeds (torch.Tensor): [1, N, D]
        segments: [(first_start, first_end), (second_start, second_end)]
        pq_obj: object with sa_encode/sa_decode
        asym (bool): if True, skip the first segment
    """
    (first_start, first_end), (second_start, second_end) = segments
    # Always quantize the second segment
    out = pq_segment(inputs_embeds, second_start, second_end, pq_obj)

    # Optionally quantize the first segment
    if not asym:
        out = pq_segment(out, first_start, first_end, pq_obj)
    return out
