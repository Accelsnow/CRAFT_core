import torch
import pickle
import os
import argparse
import eplb_craft
import torch.nn.utils.rnn as rnn_utils
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class InputArgs:
    nodes: int
    experts: int
    layers: int
    first_moe_layer: int
    dist_file: str
    output_dir: str
    overwrite: bool


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got '{value}'.") from exc

    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got '{value}'.")
    return parsed


def _non_negative_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected an integer, got '{value}'.") from exc

    if parsed < 0:
        raise argparse.ArgumentTypeError(f"Expected a non-negative integer, got '{value}'.")
    return parsed


def _existing_file(value: str) -> str:
    if not os.path.isfile(value):
        raise argparse.ArgumentTypeError(f"Input load distribution file does not exist: '{value}'.")
    return value


def _output_dir_str(value: str) -> str:
    parsed = value.strip()
    if not parsed:
        raise argparse.ArgumentTypeError("output_dir cannot be empty.")
    return parsed


def parse_args() -> InputArgs:
    parser = argparse.ArgumentParser(
        description="Compute a CRAFT layer-wise expert replica allocation plan."
    )
    parser.add_argument(
        "--nodes",
        required=True,
        type=_positive_int,
        help="Number of nodes in the cluster (positive integer; assumes 8 GPUs per node).",
    )
    parser.add_argument(
        "--experts",
        required=True,
        type=_positive_int,
        help="Number of logical experts per layer (positive integer).",
    )
    parser.add_argument(
        "--layers",
        required=True,
        type=_positive_int,
        help="Total number of layers in the model (positive integer).",
    )
    parser.add_argument(
        "--first-moe-layer",
        required=True,
        dest="first_moe_layer",
        type=_non_negative_int,
        help="Index of first MoE layer (non-negative integer; must be < layers).",
    )
    parser.add_argument(
        "--dist-file",
        required=True,
        dest="dist_file",
        type=_existing_file,
        help="Path to input load distribution pickle file.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        dest="output_dir",
        type=_output_dir_str,
        help="Optional output directory for .opt files (default: results).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )

    parsed = parser.parse_args()
    if parsed.first_moe_layer >= parsed.layers:
        parser.error(f"first_moe_layer must be in [0, {parsed.layers - 1}], got {parsed.first_moe_layer}.")

    return InputArgs(
        experts=parsed.experts,
        nodes=parsed.nodes,
        layers=parsed.layers,
        first_moe_layer=parsed.first_moe_layer,
        dist_file=parsed.dist_file,
        output_dir=parsed.output_dir,
        overwrite=parsed.overwrite,
    )


def create_balanced_tensor_interleaved(num_layers: int, num_gpus: int, pexp_per_layer: List[int]) -> torch.Tensor:
    """Distribute per-layer replica slots across GPUs with interleaved tie-breaking.

    Args:
        num_layers: Number of layers.
        num_gpus: Number of GPUs.
        pexp_per_layer: Replica slots at each layer.

    Returns:
        Per-layer GPU replica counts with shape ``(num_layers, num_gpus)``.
    """
    if len(pexp_per_layer) != num_layers:
        raise ValueError(f"Length of C ({len(pexp_per_layer)}) must match L ({num_layers}).")

    pexp_per_layer = torch.tensor(pexp_per_layer, dtype=torch.int32)

    base = pexp_per_layer // num_gpus
    remainder = pexp_per_layer % num_gpus

    # each GPU starts with the base load. Only need to assign remainders.
    assignment = base.view(num_layers, 1).expand(num_layers, num_gpus).clone()

    gpu_load_sum = assignment.sum(dim=0).clone()

    for i in range(num_layers):
        r = remainder[i].item()

        if r > 0:
            sorted_indices = torch.argsort(gpu_load_sum)
            cutoff_load = gpu_load_sum[sorted_indices[r - 1]]

            # GPU with load less than cutoff are guaranteed to be selected
            must_choose_mask = gpu_load_sum < cutoff_load
            must_choose_indices = torch.where(must_choose_mask)[0]

            # GPUs with the same cutoff load get interleaved tiebreaking
            candidate_mask = gpu_load_sum == cutoff_load
            candidate_indices = torch.where(candidate_mask)[0]

            num_ties = r - len(must_choose_indices)

            if num_ties > 0:
                # select evenly spaced indices from the candidate pool to interleave
                selection_indices = torch.linspace(
                    0, len(candidate_indices) - 1, num_ties
                ).long()

                interleaved_from_candidates = candidate_indices[selection_indices]

                indices_to_increment = torch.cat([must_choose_indices, interleaved_from_candidates])
            else:
                indices_to_increment = must_choose_indices

            assignment[i, indices_to_increment] += 1
            gpu_load_sum[indices_to_increment] += 1

    return assignment


def balancedness(loads: torch.Tensor) -> torch.Tensor:
    """Compute mean-to-max load balancedness.

    Args:
        loads: Load tensor where the last dimension is the load value.

    Returns:
        Balancedness ratio of the input load tensor.
    """
    mean_load = loads.float().mean(dim=-1) + 1e-5
    max_load = loads.float().max(dim=-1)[0] + 1e-5
    return mean_load / max_load


def pexp_list_to_gpu(pexps: torch.Tensor, gpu_pexp_offsets: torch.Tensor) -> torch.Tensor:
    """Map physical expert indices to GPU indices using prefix offsets.

    Args:
        pexps: Physical expert indices.
        gpu_pexp_offsets: Start and end physical expert indices on each GPU

    Returns:
        GPU index for each physical expert.
    """
    return torch.searchsorted(gpu_pexp_offsets, pexps, right=True) - 1


def run_bal_trials(
        num_nodes: int,
        num_gpus: int,
        num_layers: int,
        num_experts: int,
        num_replicas: int,
        input_dist: torch.Tensor,
        start_layer: int,
        no_eplb: bool,
) -> torch.Tensor:
    """Run one balancing trial and return average GPU balancedness per MoE layer.

    Args:
        num_nodes: Number of nodes in the cluster.
        num_gpus: Total number of GPUs across nodes.
        num_layers: Total number of layers in the model.
        num_experts: Number of logical experts per layer.
        num_replicas: Number of additional replicas for each layer.
        input_dist: Input distribution tensor.
        start_layer: index of the first MoE layer.
        no_eplb: when enabled, no load balancing is performed.

    Returns:
        Average GPU balancedness for each MoE layer.
    """
    device = input_dist.device

    iters, lay, lexps = input_dist.shape
    if lay != num_layers:
        raise ValueError(f"Input trace has {lay} layers but --layers is {num_layers}.")

    weight = input_dist.sum(dim=0)

    assert num_gpus % 2 == 0
    assert lexps % num_gpus == 0

    gpu_per_node = num_gpus // num_nodes
    per_lay_pexp_ct = [num_experts + num_replicas for _ in range(lay)]
    gpu_pexp_cts = create_balanced_tensor_interleaved(lay, num_gpus, per_lay_pexp_ct)
    gpu_pexp_cts = gpu_pexp_cts.to(device)

    gpu_pexp_offsets = torch.cat(
        [torch.zeros(gpu_pexp_cts.size(0), 1, dtype=gpu_pexp_cts.dtype, device=device),
         torch.cumsum(gpu_pexp_cts, dim=1)], dim=1)

    pexp_to_gpu = rnn_utils.pad_sequence(
        [pexp_list_to_gpu(torch.arange(per_lay_pexp_ct[l], device=device), gpu_pexp_offsets[l]) for l in range(lay)],
        batch_first=True,
        padding_value=-1
    ).to(device)

    if no_eplb:
        phy2log = torch.arange(0, num_experts, device=device).unsqueeze(0).repeat(lay, 1)
        log2phy = torch.arange(num_experts, device=device).unsqueeze(1).unsqueeze(0).expand(num_layers, -1, -1)
        logcnt = torch.ones_like(phy2log)
    else:
        phy2log, log2phy, logcnt = eplb_craft.rebalance_experts(weight, gpu_pexp_cts, num_nodes, start_layer, 2)

    max_reps = log2phy.shape[2]
    lay_slice = slice(start_layer, None)
    valid_mask = (log2phy != -1)
    phy_exp_gpu_map = torch.gather(
        pexp_to_gpu.unsqueeze(1).expand(-1, lexps, -1),
        2,
        log2phy.clamp(min=0)
    ) * valid_mask

    # Replay input trace and calculate the per-iteration load on each GPU
    # The token load of a logical expert is evenly spread across its replicas (including original)
    idist_slice = input_dist[:, lay_slice, :]
    logcnt_slice = logcnt[lay_slice, :]
    gpu_map_slice = phy_exp_gpu_map[lay_slice, :, :]
    valid_mask_slice = valid_mask[lay_slice, :, :]
    num_valid_layers = gpu_map_slice.shape[0]

    loc_load = idist_slice // logcnt_slice.unsqueeze(0)

    loads_to_scatter = loc_load.unsqueeze(-1).expand(-1, -1, -1, max_reps)
    indices_to_scatter = gpu_map_slice.unsqueeze(0).expand(iters, -1, -1, -1)
    mask_expanded = valid_mask_slice.unsqueeze(0).expand(iters, -1, -1, -1)

    flat_loads = loads_to_scatter[mask_expanded]
    flat_gpu_indices = indices_to_scatter[mask_expanded]

    row_indices = torch.arange(iters, device=device).view(iters, 1, 1, 1)
    layer_indices = torch.arange(num_valid_layers, device=device).view(1, num_valid_layers, 1, 1)
    flat_row_layer_indices = (row_indices * num_valid_layers + layer_indices).expand_as(mask_expanded)[mask_expanded]

    linear_indices = flat_row_layer_indices * num_gpus + flat_gpu_indices
    gpu_loads = torch.zeros(iters * num_valid_layers, num_gpus, dtype=torch.int64, device=device)
    gpu_loads.put_(linear_indices, flat_loads, accumulate=True)
    gpu_loads = gpu_loads.view(iters, num_valid_layers, num_gpus)

    gpu_bal = balancedness(gpu_loads.view(-1, num_gpus))
    avg_gpu_bal_per_lay = gpu_bal.view(iters, num_valid_layers).mean(dim=0)

    return avg_gpu_bal_per_lay


def compute_replicate_benefits(
        replicas_to_layer_bal: Dict[int, torch.Tensor],
) -> Tuple[Dict[int, List[float]], float]:
    """Compute replication benefit relative to the EPLB placement-only baseline.

    Args:
        replicas_to_layer_bal: Map of replica size to per-layer balancedness tensor.

    Returns:
        Benefit map by replica size and global average benefit.
    """
    assert 0 in replicas_to_layer_bal, "Baseline replica size 0 must be present in the input map."

    sorted_replicas = sorted(replicas_to_layer_bal.keys())
    base_avg_bal = replicas_to_layer_bal[0]
    benefit_map = {}
    tot_sum = 0
    for i in range(1, len(sorted_replicas)):
        curr_replica_size = sorted_replicas[i]
        curr_avg_bal = replicas_to_layer_bal[curr_replica_size]
        bal_benefit = curr_avg_bal - base_avg_bal
        per_replica_benefit = bal_benefit / curr_replica_size
        tot_sum += per_replica_benefit.sum().item()
        benefit_map[curr_replica_size] = per_replica_benefit.tolist()

    tot_avg = tot_sum / ((len(sorted_replicas) - 1) * len(base_avg_bal))
    return benefit_map, tot_avg


def round_final_alloc(layer_alloc: List[int], num_gpus: int) -> Tuple[List[int], int]:
    """Pad benefit-driven replication plan such that the number of post-replication experts is divisible by GPU count.

    Args:
        layer_alloc: Replica allocation per layer.
        num_gpus: Total number of GPUs across nodes.

    Returns:
        Rounded per-layer replication plan.
    """
    current_sum = sum(layer_alloc)
    if current_sum % num_gpus == 0:
        return layer_alloc, current_sum // num_gpus

    target_ratio = current_sum // num_gpus + 1
    target_sum = target_ratio * num_gpus

    while current_sum < target_sum:
        min_val = min(layer_alloc)
        min_idx = layer_alloc.index(min_val)
        increase = max(min_val, 1)

        if current_sum + increase > target_sum:
            needed_addition = target_sum - current_sum
            layer_alloc[min_idx] += needed_addition
            current_sum += needed_addition
            break

        if min_val == 0:
            layer_alloc[min_idx] = 1
        else:
            layer_alloc[min_idx] *= 2
        current_sum += increase

    return layer_alloc, target_ratio


def compute_craft_bal(
        craft_plan: List[int],
        num_nodes: int,
        num_gpus: int,
        num_experts: int,
        input_dist: torch.Tensor,
        start_layer: int,
) -> float:
    """Run per-layer balancedness evaluation to compute the average balancedness of a craft replication plan.

    Args:
        craft_plan: Number of physical experts allocated by CRAFT for each layer.
        num_nodes: Number of nodes in the cluster.
        num_gpus: Total number of GPUs across nodes.
        num_experts: Number of logical experts per layer.
        input_dist: Input distribution tensor.
        start_layer: index of the first MoE layer.

    Returns:
        Total average balancedness of the craft replication plan.
    """

    layer_bal = []
    # skip non-MoE layers, then evaluate each layer separately as they have different number of replicas
    for layer, pexps in enumerate(craft_plan):
        if layer < start_layer:
            continue
        layer_bal.append(
            run_bal_trials(num_nodes, num_gpus, 1, num_experts, pexps - num_experts, input_dist[:, layer:layer + 1, :],
                           0, False).mean().item())

    return sum(layer_bal) / len(layer_bal)


def print_write(s: str, f) -> None:
    print(s)
    f.write(s + "\n")


def craft_expert_replica_allocation(
        num_nodes: int,
        num_gpus: int,
        num_layers: int,
        num_experts: int,
        logct_file: str,
        start_layer: int,
        output_dir: str = "results",
        overwrite: bool = False,
) -> None:
    """Compute a layer-wise expert replica allocation plan using CRAFT.

    Args:
        num_nodes: Number of nodes in the cluster.
        num_gpus: Total number of GPUs across nodes.
        num_layers: Total number of layers in the model.
        num_experts: Number of logical experts per layer.
        logct_file: Path to the expert load tensor.
        start_layer: index of the first MoE layer.
        output_dir: Output directory for writing allocation files.
        overwrite: Whether to overwrite existing output files.
    """
    if start_layer >= num_layers:
        raise ValueError(f"start_layer must be in [0, {num_layers - 1}], got {start_layer}.")

    os.makedirs(output_dir, exist_ok=True)
    output_file_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(logct_file))[0]}.opt")
    if os.path.exists(output_file_path) and not overwrite:
        print(
            f"Output file already exists: {output_file_path}. "
            "Re-run with --overwrite to replace it."
        )
        return

    replica_sizes = [0, 4, 8, 16, 32]
    replicas_to_layer_bal = {}
    device = torch.device("cpu")

    with open(logct_file, "rb") as f:
        input_dist = pickle.load(f)

    sums = input_dist.sum(dim=(1, 2))
    input_dist = input_dist[sums >= sums.to(torch.float).mean() / 10].to(device)

    with open(output_file_path, "w", encoding="utf-8") as output_file:
        for num_replicas in replica_sizes:
            print_write(
                f"Replaying inference batches from input load distribution for {num_replicas} {'replica' if num_replicas == 1 else 'replicas'}",
                output_file)
            replicas_to_layer_bal[num_replicas] = run_bal_trials(
                num_nodes, num_gpus, num_layers, num_experts, num_replicas, input_dist, start_layer, False
            )

        print("Evaluating per-layer replication benefits")
        replicate_benefits, benefit_tot_avg = compute_replicate_benefits(replicas_to_layer_bal)

        layer_allocations = []
        sorted_replica_sizes = sorted(replicate_benefits.keys())
        moe_layers = num_layers - start_layer

        for layer in range(moe_layers):
            pick_replica_size = 0
            max_benefit = 0
            is_high_benefit_layer = False
            for replica_size in sorted_replica_sizes:
                assert len(replicate_benefits[
                               replica_size]) == moe_layers, f"replicate_benefits {replicate_benefits} must have moe_layers {moe_layers} entries"

                benefit = replicate_benefits[replica_size][layer]

                if not is_high_benefit_layer and replica_size <= 16:
                    if benefit >= (benefit_tot_avg / 2.0):
                        pick_replica_size = replica_size
                    elif benefit > max_benefit:
                        max_benefit = benefit
                        pick_replica_size = replica_size

                if benefit >= benefit_tot_avg:
                    pick_replica_size = replica_size
                    is_high_benefit_layer = True
            layer_allocations.append(pick_replica_size)

        layer_allocations, replication_ratio = round_final_alloc(layer_allocations, num_gpus)
        print_write(
            f"CRAFT replication factor R={replication_ratio} (EPLB R={moe_layers}, CRAFT allocates {moe_layers / replication_ratio:.2f}x less memory)",
            output_file)

        layerwise_allocation_plan = [num_experts for _ in range(start_layer)] + [r + num_experts for r in
                                                                                 layer_allocations]

        print_write("CRAFT Replication Plan:", output_file)

        for i in range(0, start_layer):
            print_write(f"Layer {i} (non-MoE) - 0 experts", output_file)

        for i in range(start_layer, num_layers):
            print_write(f"Layer {i}{' ' if i >= 10 else '  '}(MoE) - {layerwise_allocation_plan[i]} experts",
                        output_file)

        print("CRAFT replication plan completed! Now evaluating EPLB and CRAFT balancedness via replay...")

        craft_bal = compute_craft_bal(layerwise_allocation_plan, num_nodes, num_gpus, num_experts, input_dist,
                                      start_layer)

        # EPLB allocates num_gpu replicas per layer
        eplb_bal = run_bal_trials(
            num_nodes, num_gpus, num_layers, num_experts, num_gpus, input_dist, start_layer, False
        ).mean().item()

        # replica_size=0 implies placement-only
        place_only_bal = replicas_to_layer_bal[0].mean().item()

        print_write(
            f"CRAFT balancedness: {craft_bal:.2f}, EPLB balancedness: {eplb_bal:.2f}, placement-only balancedness: {place_only_bal:.2f}",
            output_file
        )


def main():
    args = parse_args()
    num_gpus = args.nodes * 8
    craft_expert_replica_allocation(
        num_nodes=args.nodes,
        num_gpus=num_gpus,
        num_layers=args.layers,
        num_experts=args.experts,
        logct_file=args.dist_file,
        start_layer=args.first_moe_layer,
        output_dir=args.output_dir,
        overwrite=args.overwrite,
    )


if __name__ == '__main__':
    main()
