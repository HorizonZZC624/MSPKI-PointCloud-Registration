from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

from common import ROOT, SEEDS, correspondence_metrics, digest_file, fixed_indices, metadata_keys, paired_summary, pose_metrics, write_csv, write_json
from density_diagnostics import read_archive

EXPERIMENT = "geotransformer.kitti.stage5.gse.k3.max.oacl.stage2.sinkhorn"


def load_module(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description="Official GeoTransformer KITTI checkpoint with the existing fixed source masks.")
    p.add_argument("--geo-root", type=Path, required=True)
    p.add_argument("--snapshot", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=ROOT / "datasets/KITTI")
    p.add_argument("--metadata-root", type=Path, default=ROOT / "data/KITTI/metadata")
    p.add_argument("--output", type=Path, default=ROOT / "output/KITTI_additional/geotransformer")
    p.add_argument("--ratios", type=float, nargs="+", default=[1.0, .5])
    p.add_argument("--mask-seed", type=int, default=9017)
    p.add_argument("--inference-seed", type=int, default=7351)
    p.add_argument("--neighbor-limits", type=int, nargs=5)
    p.add_argument("--pare-archive", type=Path, default=ROOT / "results/additional_evaluation/kitti_density_complete.tar.gz")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--max-pairs", type=int, default=0)
    args = p.parse_args()
    if args.max_pairs < 0 or any(not 0 < x <= 1 for x in args.ratios):
        raise ValueError("Invalid density ratios or maximum pairs.")
    args.geo_root = args.geo_root.resolve()
    experiment = args.geo_root / "experiments" / EXPERIMENT
    metadata, expected = metadata_keys(args.metadata_root)
    required = [experiment / "model.py", experiment / "config.py", args.snapshot]
    if args.neighbor_limits is None:
        required.append(args.metadata_root / "train.pkl")
    required.extend(args.dataset_root / item[key] for item in metadata for key in ("pcd0", "pcd1"))
    missing = sorted({str(x) for x in required if not x.is_file()})
    plan = {"model": "GeoTransformer", "checkpoint_status": "one released checkpoint, not three trained seeds",
            "geo_root": str(args.geo_root), "snapshot": str(args.snapshot), "ratios": args.ratios,
            "mask_seed": args.mask_seed, "inference_seed": args.inference_seed,
            "mask_rule": "identical SHA256(mask_seed:original_sequence_id:source_frame) permutation prefix before hierarchy construction",
            "IR_radius_m": 1.0, "TR_RRE_strict_deg": 5.0, "TR_RTE_strict_m": 2.0,
            "pose_estimator": "official GeoTransformer LGR; PARE-Net/MSPKI archive uses FHP",
            "comparison_scope": "system-level common-input evaluation; not a same-training-protocol module ablation",
            "correspondence_IR": "native full set plus top1000 scores; always report correspondence counts",
            "complete_test_set": args.max_pairs == 0, "missing_inputs": missing}
    print(json.dumps(plan, indent=2), flush=True)
    if args.dry_run:
        return
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} required files; first: {missing[0]}")
    archive = read_archive(args.pare_archive, args.ratios, SEEDS, args.mask_seed, expected) if not args.max_pairs else None
    if any(args.output.glob("density_src_*/features/*.npz")):
        raise FileExistsError("GeoTransformer outputs already exist. Use a new --output directory.")
    sys.path.insert(0, str(args.geo_root))
    sys.path.insert(0, str(experiment))
    import torch
    from geotransformer.datasets.registration.kitti.dataset import OdometryKittiPairDataset
    from geotransformer.utils.data import registration_collate_fn_stack_mode, calibrate_neighbors_stack_mode
    from geotransformer.utils.torch import initialize, to_cuda
    if not torch.cuda.is_available():
        raise RuntimeError("GeoTransformer inference requires CUDA and its compiled official operators.")
    cfg = load_module("geo_kitti_config", experiment / "config.py").make_cfg()
    model_module = load_module("geo_kitti_model", experiment / "model.py")
    initialize(seed=args.inference_seed)

    class SharedMetadataDataset(OdometryKittiPairDataset):
        def __init__(self, subset):
            self.dataset_root = str(args.dataset_root)
            self.subset = subset
            self.point_limit = cfg.train.point_limit if subset == "train" else None
            self.use_augmentation = cfg.train.use_augmentation if subset == "train" else False
            for attribute in ("augmentation_noise", "augmentation_min_scale", "augmentation_max_scale", "augmentation_shift", "augmentation_rotation"):
                setattr(self, attribute, getattr(cfg.train, attribute))
            self.return_corr_indices = False
            self.matching_radius = None
            with (args.metadata_root / f"{subset}.pkl").open("rb") as stream:
                self.metadata = pickle.load(stream)

    neighbor_limits = args.neighbor_limits
    if neighbor_limits is None:
        neighbor_limits = calibrate_neighbors_stack_mode(SharedMetadataDataset("train"), registration_collate_fn_stack_mode,
                                                       cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius).tolist()
    model = model_module.create_model(cfg).cuda().eval()
    try:
        state = torch.load(args.snapshot, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(args.snapshot, map_location="cpu")
    model.load_state_dict(state["model"], strict=True)
    plan.update(snapshot_sha256=digest_file(args.snapshot), metadata_sha256=digest_file(args.metadata_root / "test.pkl"),
                neighbor_limits=[int(x) for x in neighbor_limits], torch_version=torch.__version__,
                numpy_version=np.__version__, entry_script_sha256=digest_file(__file__), completed=False)
    try:
        plan["geo_commit"] = subprocess.check_output(["git", "-C", str(args.geo_root), "rev-parse", "HEAD"], text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        plan["geo_commit"] = "unavailable"
    write_json(args.output / "protocol.json", plan)
    dataset = SharedMetadataDataset("test")
    results = {}
    for ratio in args.ratios:
        folder = args.output / f"density_src_r{round(100*ratio):03d}_m{args.mask_seed}"
        (folder / "features").mkdir(parents=True, exist_ok=True)
        rows, masks = [], []
        limit = args.max_pairs or len(dataset)
        for index in range(min(limit, len(dataset))):
            sample = dataset[index]
            full_src = sample["src_points"]
            indices = fixed_indices(len(full_src), ratio, args.mask_seed, sample["seq_id"], sample["src_frame"])
            scan_hash = hashlib.sha256(np.ascontiguousarray(full_src).tobytes()).hexdigest()
            index_hash = hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()
            sample["src_points"] = full_src[indices].copy()
            sample["src_feats"] = sample["src_feats"][indices].copy()
            data = registration_collate_fn_stack_mode([sample], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
                                                      cfg.backbone.init_radius, neighbor_limits)
            data = to_cuda(data)
            with torch.no_grad():
                output = model(data)
            arrays = {key: output[key].detach().cpu().numpy() for key in ("ref_corr_points", "src_corr_points", "corr_scores", "estimated_transform")}
            arrays["transform"] = sample["transform"]
            key = (int(sample["seq_id"]), int(sample["src_frame"]), int(sample["ref_frame"]))
            np.savez_compressed(folder / "features" / ("_".join(map(str, key)) + ".npz"), **arrays)
            ref, src, inlier = correspondence_metrics(arrays["ref_corr_points"], arrays["src_corr_points"], arrays["transform"])
            _, _, top_inlier = correspondence_metrics(ref, src, arrays["transform"], arrays["corr_scores"], 1000)
            rre, rte, tr = pose_metrics(arrays["transform"], arrays["estimated_transform"])
            rows.append({"seq": key[0], "src_frame": key[1], "ref_frame": key[2], "IR": float(inlier.mean()) if len(inlier) else 0.0,
                         "correspondences": len(inlier), "IR_top1000": float(top_inlier.mean()) if len(top_inlier) else 0.0,
                         "top1000_actual_count": len(top_inlier), "RRE": rre, "RTE_m": rte, "TR": tr})
            masks.append({"seq": key[0], "src_frame": key[1], "ref_frame": key[2], "original_sequence_repr": str(sample["seq_id"]),
                          "source_input_points": len(full_src), "source_retained_points": len(indices), "raw_source_sha256": scan_hash,
                          "source_indices_sha256": index_hash})
            if (index+1) % 25 == 0 or index+1 == limit:
                print(f"GeoTransformer ratio={ratio}: {index+1}/{limit}", flush=True)
        write_csv(folder / "pairs_lgr.csv", rows)
        write_csv(folder / "masks.csv", masks)
        results[ratio] = rows
        manifest = dict(plan, keep_ratio=ratio, saved_pairs=len(rows), completed=len(rows) == 555)
        write_json(folder / "features/manifest.json", manifest)
    if not args.max_pairs:
        comparisons = []
        for ratio, geo in results.items():
            for seed in SEEDS:
                for method in ("baseline", "mspki"):
                    other = archive[(ratio, seed, method)][0]
                    comparison = paired_summary(geo, [other[k] for k in expected])
                    comparison = {k.replace("baseline_", "geotransformer_").replace("mspki_", "candidate_"): v for k, v in comparison.items()}
                    comparison.update(keep_ratio=ratio, candidate=method, candidate_training_seed=seed,
                                      geotransformer_checkpoint_count=1, geotransformer_pose="LGR", candidate_pose="FHP")
                    comparisons.append(comparison)
        write_csv(args.output / "native_system_comparison.csv", comparisons)
    plan["completed"] = not args.max_pairs
    write_json(args.output / "protocol.json", plan)


if __name__ == "__main__":
    main()
