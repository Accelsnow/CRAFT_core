from typing import Tuple

import torch

def balanced_packing_gpu(weight: torch.Tensor, num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly n/m objects and the weights of all packs
    are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_gpus: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_gpus == 0
    pexp_per_gpu = num_groups // num_gpus

    if pexp_per_gpu == 1:
        gpu_assign = torch.arange(weight.size(-1), dtype=torch.int64, device=weight.device).expand(weight.shape)
        rank_in_gpu = torch.zeros_like(weight, dtype=torch.int64)
        return gpu_assign, rank_in_gpu

    indices = weight.float().sort(-1, descending=True).indices.cpu()

    gpu_assign = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device='cpu')
    rank_in_gpu = torch.full_like(gpu_assign, fill_value=-1)
    for i in range(num_layers):
        gpu_weights = [0] * num_gpus
        gpu_pexp_ct = [0] * num_gpus
        for pexp in indices[i]:
            # pick lightest load GPU
            gpu_idx = min((g for g in range(num_gpus) if gpu_pexp_ct[g] < pexp_per_gpu), key=gpu_weights.__getitem__)

            assert gpu_pexp_ct[gpu_idx] < pexp_per_gpu
            gpu_assign[i, pexp] = gpu_idx
            rank_in_gpu[i, pexp] = gpu_pexp_ct[gpu_idx]
            gpu_weights[gpu_idx] += weight[i, pexp]
            gpu_pexp_ct[gpu_idx] += 1

    return gpu_assign, rank_in_gpu

def balanced_packing_gpu_unique_gpunode(weight: torch.Tensor,
                                gpu_pexp_cts: torch.Tensor,
                                phy2log: torch.Tensor,
                                num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Like balanced_packing_gpu, but ENFORCES: replicas of the same logical expert
    must land on DIFFERENT GPUs (devices).
    Parameters:
        weight: [X, num_phy], weight of each physical expert
        gpu_pexp_cts: [X, num_gpus] capacity per device
        phy2log: [X, num_phy], logical expert id of each physical expert (for each layer)
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
    Returns:
        gpu_assign: [X, num_phy], device index for each physical expert
        rank_in_gpu: [X, num_phy], intra-device slot index
    """
    assert gpu_pexp_cts.shape[0] == weight.shape[0] == phy2log.shape[0], f"{gpu_pexp_cts.shape}, {weight.shape}, {phy2log.shape}"
    num_gpus = gpu_pexp_cts.shape[1]
    num_layers, num_phy = weight.shape
    device = gpu_pexp_cts.device
    gpus_per_node = num_gpus // num_nodes
    indices = weight.float().sort(-1, descending=True).indices.cpu()

    gpu_assign = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device=device)
    rank_in_gpu = torch.full_like(gpu_assign, fill_value=-1, device=device)

    for i in range(num_layers):
        gpu_weights = [0.0] * num_gpus
        gpu_counts  = [0]   * num_gpus
        gpu_existing_lexp = [set() for _ in range(num_gpus)]
        node_exiting_lexp = [set() for _ in range(num_nodes)]
        gpu_pexp_ct = gpu_pexp_cts[i].tolist()

        for pexp in indices[i]:
            log_id = int(phy2log[i, pexp])

            eligible_candidates = [g for g in range(num_gpus)
                          if gpu_counts[g] < gpu_pexp_ct[g]]

            # avoid placing replicas of the same logical expert on the same node
            candidates = [g for g in eligible_candidates if log_id not in node_exiting_lexp[g // gpus_per_node]]

            if len(candidates) == 0:
                # avoid placing replicas of the same logical expert on the same GPU
                candidates = [g for g in eligible_candidates if log_id not in gpu_existing_lexp[g]]

            if len(candidates) == 0:
                candidates = eligible_candidates

            if len(candidates) == 0:
                candidates = [0]

            # prioritize load balance and use available physical expert slots as tie-breaker
            target_gpu = min(candidates, key=lambda g: gpu_weights[g] * 1000 + gpu_pexp_ct[g])

            gpu_assign[i, pexp] = target_gpu
            rank_in_gpu[i, pexp] = gpu_counts[target_gpu]
            gpu_weights[target_gpu] += float(weight[i, pexp])
            gpu_counts[target_gpu] += 1
            gpu_existing_lexp[target_gpu].add(log_id)
            # Update seen logical experts on GPUs / nodes
            if all(log_id in s for s in gpu_existing_lexp):
                for s in gpu_existing_lexp:
                    s.remove(log_id)
            node_exiting_lexp[target_gpu // gpus_per_node].add(log_id)
            if all(log_id in s for s in node_exiting_lexp):
                for s in node_exiting_lexp:
                    s.remove(log_id)

    return gpu_assign, rank_in_gpu

def balanced_packing_gpu_unique_gpu_only(weight: torch.Tensor,
                                gpu_pexp_cts: torch.Tensor,
                                phy2log: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Like balanced_packing_gpu, but ENFORCES: replicas of the same logical expert
    must land on DIFFERENT GPUs (devices).
    Parameters:
        weight: [X, num_phy], weight of each physical expert
        gpu_pexp_cts: [X, num_gpus] capacity per device
        phy2log: [X, num_phy], logical expert id of each physical expert (for each layer)
    Returns:
        gpu_assign: [X, num_phy], device index for each physical expert
        rank_in_gpu: [X, num_phy], intra-device slot index
    """
    assert gpu_pexp_cts.shape[0] == weight.shape[0] == phy2log.shape[0], f"{gpu_pexp_cts.shape}, {weight.shape}, {phy2log.shape}"
    num_gpus = gpu_pexp_cts.shape[1]
    num_layers, num_phy = weight.shape
    indices = weight.float().sort(-1, descending=True).indices.cpu()

    gpu_assign = torch.full_like(weight, fill_value=-1, dtype=torch.int64, device='cpu')
    rank_in_gpu = torch.full_like(gpu_assign, fill_value=-1)

    for i in range(num_layers):
        gpu_weights = [0.0] * num_gpus
        gpu_counts  = [0]   * num_gpus
        gpu_existing_lexp = [set() for _ in range(num_gpus)]
        gpu_pexp_ct = gpu_pexp_cts[i].tolist()

        for pexp in indices[i]:
            log_id = int(phy2log[i, pexp])

            eligible_candidates = [g for g in range(num_gpus) if gpu_counts[g] < gpu_pexp_ct[g]]

            # avoid placing replicas of the same logical expert on the same GPU
            candidates = [g for g in eligible_candidates if log_id not in gpu_existing_lexp[g]]

            if len(candidates) == 0:
                print("WARNING AN MOE LAYER FORCED TO DUPLICATE PEXP ON SAME GPU")
                candidates = eligible_candidates

            # prioritize load balance and use available physical expert slots as tie-breaker
            target_gpu = min(candidates, key=lambda g: gpu_weights[g] * 1000 + gpu_pexp_ct[g])

            gpu_assign[i, pexp] = target_gpu
            rank_in_gpu[i, pexp] = gpu_counts[target_gpu]
            gpu_weights[target_gpu] += float(weight[i, pexp])
            gpu_counts[target_gpu] += 1
            gpu_existing_lexp[target_gpu].add(log_id)
            # Update seen logical experts on GPUs
            if all(log_id in s for s in gpu_existing_lexp):
                for s in gpu_existing_lexp:
                    s.remove(log_id)

    return gpu_assign, rank_in_gpu


def replicate_experts(weight: torch.Tensor, num_phy: int, num_gpus: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication
        num_gpus: number of GPUs

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0, f"{num_phy} < {num_log}"
    device = weight.device
    
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / torch.where(logcnt >= num_gpus, torch.inf, logcnt)).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(weight: torch.Tensor, gpu_pexp_cts: torch.Tensor, num_physical_experts: int,
                                   num_nodes: int, replicate_mode: int):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        gpu_pexp_cts: number of physical experts after replication per GPU
        num_physical_experts: number of physical experts
        num_nodes: number of server nodes
        replicate_mode: replica placement mode. 2 - spread across GPUs & nodes, 1 - spread across GPUs, 0 - default

    Returns: 
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert gpu_pexp_cts.shape[0] == num_layers, f"{gpu_pexp_cts.shape}, {weight.shape}"
    num_gpus = gpu_pexp_cts.shape[1]
    assert num_gpus % num_nodes == 0

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(perm.shape))
        return inv

    phy2log, phyrank, logcnt = replicate_experts(weight, num_physical_experts, num_gpus)

    tokens_per_phy = (weight / logcnt).gather(-1, phy2log)
    
    if replicate_mode == 0:
        pack_index, rank_in_pack = balanced_packing_gpu(tokens_per_phy, num_gpus)
    elif replicate_mode == 1:
        pack_index, rank_in_pack = balanced_packing_gpu_unique_gpu_only(
            tokens_per_phy,
            gpu_pexp_cts,
            phy2log,
        )
    elif replicate_mode == 2:
        pack_index, rank_in_pack = balanced_packing_gpu_unique_gpunode(
            tokens_per_phy,
            gpu_pexp_cts,
            phy2log,
            num_nodes,
        )
    else:
        raise NotImplementedError(f"Unique mode {replicate_mode} not supported")

    gpu_pexp_offsets = torch.cat([torch.zeros(gpu_pexp_cts.size(0), 1, dtype=gpu_pexp_cts.dtype, device=gpu_pexp_cts.device),
                                  torch.cumsum(gpu_pexp_cts, dim=1)], dim=1)

    phy2pphy = torch.gather(gpu_pexp_offsets, 1, pack_index) + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2log = phy2log.gather(-1, pphy2phy)
    pphy2log = (pphy2log.view(num_layers, 1, -1) +
                 torch.arange(0, num_logical_experts, num_logical_experts,
                              device=weight.device).view(1, -1, 1)).flatten(-2)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(weight: torch.Tensor, gpu_pexp_cts: torch.Tensor, num_nodes: int, lay_start: int, replicate_mode: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all logical experts
        gpu_pexp_cts: number of physical experts on each GPU at each layer (shape LxD)
        num_nodes: number of server nodes, where the intra-node network (e.g, NVLink) is faster
        lay_start: index of first MoE layer
        replicate_mode: replica placement mode. 2 - spread across GPUs & nodes, 1 - spread across GPUs, 0 - default

    Returns: 
        physical_to_logical_map: [layers, num_replicas], the expert index of each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica indices for each expert
        expert_count: [layers, num_logical_experts], number of physical replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape

    pexp_sums = gpu_pexp_cts.sum(dim=1)[lay_start:]
    # this function expects all layers to have the same number of physical experts
    assert torch.all(pexp_sums == pexp_sums[0]), f"{pexp_sums} not consistent, consider per-layer calls"
    num_physical_experts = pexp_sums[0].item()

    weight = weight.float()
    phy2log, phyrank, logcnt = rebalance_experts_hierarchical(weight, gpu_pexp_cts, num_physical_experts, num_nodes, replicate_mode)
    maxlogcnt = logcnt.max().item()
    log2phy: torch.Tensor = torch.full((num_layers, num_logical_experts, maxlogcnt),
                                       -1, dtype=torch.int64, device=logcnt.device)
    log2phy.view(num_layers, -1).scatter_(-1, phy2log * maxlogcnt + phyrank,
                                          torch.arange(num_physical_experts, dtype=torch.int64, device=log2phy.device).expand(
                                              num_layers, -1))
    return phy2log, log2phy, logcnt


__all__ = ['rebalance_experts']
