
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
for path in (str(HERE), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import make_cfg
from model import create_model


class Args:
    def __init__(self, variant, seed):
        self.variant = variant
        self.seed = seed
        self.dataset_root = None
        self.metadata_root = None
        self.output_root = None
        self.run_tag = ""
        self.num_workers = None
        self.re_feature_source = "official"
        self.mspki_scales = [8, 16, 32]
        self.mspki_branch_channels = 16
        self.mspki_ffn_ratio = 2.0
        self.layer_scale_init = 0.10
        self.knn_query_chunk_size = 1024
        self.knn_support_chunk_size = 4096


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensor_hash(state, keys):
    h = hashlib.sha256()
    for key in sorted(keys):
        h.update(key.encode("utf-8"))
        t = state[key].detach().cpu().contiguous()
        h.update(str(tuple(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def main():
    seed = 7351
    bcfg = make_cfg(Args("baseline", seed))
    mcfg = make_cfg(Args("mspki", seed))

    set_seed(seed)
    baseline = create_model(bcfg).cpu()
    rng_after_baseline = torch.get_rng_state().clone()

    set_seed(seed)
    mspki = create_model(mcfg).cpu()
    rng_after_mspki = torch.get_rng_state().clone()

    bs = baseline.state_dict()
    ms = mspki.state_dict()
    common = sorted(k for k in bs if k in ms and tuple(bs[k].shape) == tuple(ms[k].shape))
    differing = [k for k in common if not torch.equal(bs[k], ms[k])]
    extra_mspki = sorted(k for k in ms if k not in bs)

    result = {
        "seed": seed,
        "baseline_parameters": sum(p.numel() for p in baseline.parameters()),
        "mspki_parameters": sum(p.numel() for p in mspki.parameters()),
        "common_state_tensors": len(common),
        "differing_common_state_tensors": len(differing),
        "extra_mspki_state_tensors": len(extra_mspki),
        "global_rng_state_equal_after_model_creation": bool(
            torch.equal(rng_after_baseline, rng_after_mspki)
        ),
        "shared_state_hash_baseline": tensor_hash(bs, common),
        "shared_state_hash_mspki": tensor_hash(ms, common),
        "shared_initialization_exact_match": len(differing) == 0,
        "allowed_architecture_difference": "fine_context: none vs MSPKI(8,16,32); extra MSPKI parameters only",
    }


    official_backbone = PROJECT_ROOT / "experiments" / "KITTI" / "backbone.py"
    result["official_backbone_found"] = official_backbone.is_file()
    if official_backbone.is_file():
        spec = importlib.util.spec_from_file_location("_official_kitti_backbone_audit", official_backbone)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        from backbone import PAREConvFPN as NewBackbone

        set_seed(seed)
        old = mod.PAREConvFPN(96, 64, 4, False, "edge_conv", True, True)
        set_seed(seed)
        new = NewBackbone(
            96, 64, 4, False, "edge_conv", True,
            re_feature_source="official",
            coarse_context="none",
            fine_context="none",
            spsa_partial_ratio=0.5,
            spsa_qk_channels=32,
            spsa_ffn_ratio=2.0,
            mspki_neighbor_scales=(8, 16, 32),
            mspki_branch_channels=16,
            mspki_ffn_ratio=2.0,
            layer_scale_init=0.10,
        )
        osd, nsd = old.state_dict(), new.state_dict()
        same_keys = list(osd.keys()) == list(nsd.keys())
        same_shapes = same_keys and all(osd[k].shape == nsd[k].shape for k in osd)
        same_values = same_shapes and all(torch.equal(osd[k], nsd[k]) for k in osd)
        result["official_baseline_backbone_state_keys_exact_match"] = bool(same_keys)
        result["official_baseline_backbone_shapes_exact_match"] = bool(same_shapes)
        result["official_baseline_backbone_initial_values_exact_match"] = bool(same_values)
        result["official_baseline_backbone_parameters"] = sum(p.numel() for p in old.parameters())
        result["new_baseline_backbone_parameters"] = sum(p.numel() for p in new.parameters())

    out = HERE / "fairness_audit.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))

    required = [
        result["shared_initialization_exact_match"],
        result["global_rng_state_equal_after_model_creation"],
    ]
    if result.get("official_backbone_found"):
        required.extend([
            result.get("official_baseline_backbone_state_keys_exact_match", False),
            result.get("official_baseline_backbone_shapes_exact_match", False),
            result.get("official_baseline_backbone_initial_values_exact_match", False),
        ])
    if not all(required):
        raise SystemExit("FAIRNESS AUDIT: FAIL")
    print("FAIRNESS AUDIT: PASS")


if __name__ == "__main__":
    main()
