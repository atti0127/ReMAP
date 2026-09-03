"""Propagate re-observed evidence with exact dense evidence anchoring."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

from metrics import cal_pro_score_fast
from remap.runtime import gaussian_filter2d


def positive_feature_support(scores, features, steps=1):
    """One parameter-free, feature-bounded neighborhood support step."""

    side = int(scores.shape[1] ** 0.5)
    feature_grid = features.transpose(1, 2).reshape(
        features.shape[0], features.shape[2], side, side
    )
    neighbors = F.unfold(
        F.pad(feature_grid, (1, 1, 1, 1), mode="replicate"), kernel_size=3
    ).reshape(features.shape[0], features.shape[2], 9, -1)
    center = features.transpose(1, 2).unsqueeze(2)
    affinity = (neighbors * center).sum(dim=1).clamp_min(0.0)
    affinity = affinity / affinity.sum(dim=1, keepdim=True).clamp_min(1e-6)
    supported = scores
    for _ in range(steps):
        score_grid = supported.reshape(scores.shape[0], 1, side, side)
        score_neighbors = F.unfold(
            F.pad(score_grid, (1, 1, 1, 1), mode="replicate"), kernel_size=3
        ).reshape(scores.shape[0], 9, -1)
        neighbor_support = (affinity * score_neighbors).sum(dim=1)
        supported = torch.maximum(supported, neighbor_support)
    return supported


def inverse_crop_grid(geometry, image_size, device):
    axis = torch.linspace(-1.0, 1.0, image_size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    if geometry.shape[1] == 3:
        center_x, center_y, extent_x = geometry.unbind(dim=1)
        extent_y = extent_x
    elif geometry.shape[1] == 4:
        center_x, center_y, extent_x, extent_y = geometry.unbind(dim=1)
    else:
        raise ValueError("geometry must contain center x/y and one or two extents")
    grid_x = (xx - center_x[:, None, None]) / extent_x[:, None, None]
    grid_y = (yy - center_y[:, None, None]) / extent_y[:, None, None]
    grid = torch.stack((grid_x, grid_y), dim=-1)
    valid = (grid_x.abs() <= 1.0) & (grid_y.abs() <= 1.0)
    return grid, valid[:, None]


def dense_histogram_transport(anchor, ordering):
    flat_anchor = anchor.flatten(1)
    flat_ordering = ordering.flatten(1)
    sorted_values = flat_anchor.sort(dim=1).values
    order = flat_ordering.argsort(dim=1, stable=True)
    transported = torch.empty_like(flat_anchor)
    transported.scatter_(1, order, sorted_values)
    return transported.reshape_as(anchor)


def reconstruct_maps(metadata, fovea_meta, branches, crop_scores, crop_features,
                     geometry, rate_scores,
                     batch_size, workers, feature_steps=1, minimal=False,
                     work_dir=None, work_prefix="maps",
                     direct_intermediate=False, store_rate=True,
                     component_ablation=False, gaussian_backend="scipy"):
    count = metadata["count"]
    image_size = metadata["image_size"]
    base_side = int(metadata["tokens"] ** 0.5)
    crop_side = int(fovea_meta["crop_tokens"] ** 0.5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    shape = (count, image_size, image_size)

    def allocate(name):
        if work_dir is None:
            return np.empty(shape, dtype=np.float32)
        destination = Path(work_dir)
        destination.mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(
            destination / f"{work_prefix}_{name}.npy", mode="w+",
            dtype=np.float32, shape=shape,
        )

    baseline = allocate("baseline")
    foveal = None if minimal or component_ablation else allocate("foveal")
    positive_foveal = (
        None if minimal or component_ablation else allocate("positive_foveal")
    )
    rate_maps = (
        allocate("rate") if rate_scores is not None and store_rate else None
    )
    joint_maps = (
        None if minimal or rate_scores is None else allocate("joint")
    )
    direct_joint_maps = (
        None if minimal or component_ablation or rate_scores is None
        else allocate("direct_joint")
    )
    feature_joint_maps = (
        allocate("feature_joint")
        if rate_scores is not None and crop_features is not None else None
    )
    direct_feature_joint_maps = (
        allocate("direct_feature_joint")
        if (not minimal and not component_ablation and rate_scores is not None
            and crop_features is not None)
        else None
    )
    histogram_error = 0.0
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        native = torch.from_numpy(
            np.array(branches[start:stop, -1], dtype=np.float32, copy=True)
        ).to(device).reshape(-1, 1, base_side, base_side)
        crop = torch.from_numpy(
            np.array(crop_scores[start:stop], dtype=np.float32, copy=True)
        ).to(device).reshape(-1, 1, crop_side, crop_side)
        if crop_features is not None:
            crop_feature_t = torch.from_numpy(
                np.array(crop_features[start:stop], dtype=np.float32, copy=True)
            ).to(device)
        geometry_t = torch.from_numpy(
            np.array(geometry[start:stop], dtype=np.float32, copy=True)
        ).to(device)
        base_dense = F.interpolate(
            native, (image_size, image_size), mode="bilinear", align_corners=False
        )
        grid, valid = inverse_crop_grid(geometry_t, image_size, device)
        crop_dense = F.grid_sample(
            crop, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        base_logit = torch.logit(base_dense.clamp(1e-4, 1.0 - 1e-4))
        crop_logit = torch.logit(crop_dense.clamp(1e-4, 1.0 - 1e-4))
        base_footprint = image_size / base_side
        area_extent = (
            geometry_t[:, 2]
            if geometry_t.shape[1] == 3 else
            torch.sqrt(geometry_t[:, 2] * geometry_t[:, 3])
        )
        crop_footprint = area_extent * image_size / crop_side
        crop_precision = crop_footprint.square().reciprocal()
        base_precision = base_footprint ** -2
        crop_weight = crop_precision / (crop_precision + base_precision)
        fused = base_logit + valid * crop_weight[:, None, None, None] * (
            crop_logit - base_logit
        )
        anchored = dense_histogram_transport(base_dense, fused)
        positive_fused = base_logit + valid * crop_weight[:, None, None, None] * F.relu(
            crop_logit - base_logit
        )
        positive_anchored = dense_histogram_transport(base_dense, positive_fused)
        if rate_scores is not None:
            rate_native = torch.from_numpy(
                np.array(rate_scores[start:stop], dtype=np.float32, copy=True)
            ).to(device).reshape(-1, 1, base_side, base_side)
            rate_dense = F.interpolate(
                rate_native, (image_size, image_size), mode="bilinear",
                align_corners=False,
            )
            # RATE determines global ordering.  The crop may only add local
            # evidence, weighted by the measurement footprints; there is no
            # learned or selected fusion coefficient.
            positive_delta = valid * crop_weight[:, None, None, None] * F.relu(
                crop_logit - base_logit
            )
            joint_ordering = torch.logit(
                rate_dense.clamp(1e-4, 1.0 - 1e-4)
            ) + positive_delta
            joint_anchored = dense_histogram_transport(base_dense, joint_ordering)
            direct_joint = torch.sigmoid(joint_ordering)
            if crop_features is not None:
                supported_crop = positive_feature_support(
                    crop.flatten(1), crop_feature_t, steps=feature_steps
                ).reshape_as(crop)
                supported_dense = F.grid_sample(
                    supported_crop, grid, mode="bilinear", padding_mode="border",
                    align_corners=True,
                )
                supported_logit = torch.logit(
                    supported_dense.clamp(1e-4, 1.0 - 1e-4)
                )
                supported_delta = (
                    valid * crop_weight[:, None, None, None]
                    * F.relu(supported_logit - base_logit)
                )
                feature_joint = dense_histogram_transport(
                    base_dense,
                    torch.logit(rate_dense.clamp(1e-4, 1.0 - 1e-4))
                    + supported_delta,
                )
                direct_feature_joint = torch.sigmoid(
                    torch.logit(rate_dense.clamp(1e-4, 1.0 - 1e-4))
                    + supported_delta
                )
        error = (
            anchored.flatten(1).sort(dim=1).values
            - base_dense.flatten(1).sort(dim=1).values
        ).abs().max()
        histogram_error = max(histogram_error, float(error.item()))
        if gaussian_backend == "torch":
            def smooth_tensor(value):
                return gaussian_filter2d(value.float(), sigma=4.0)[:, 0].cpu().numpy()

            base_smoothed = smooth_tensor(base_dense)
            if foveal is not None:
                foveal_smoothed = smooth_tensor(anchored)
                positive_smoothed = smooth_tensor(positive_anchored)
            if rate_scores is not None:
                rate_smoothed = smooth_tensor(rate_dense)
                if joint_maps is not None:
                    joint_smoothed = smooth_tensor(joint_anchored)
                if direct_joint_maps is not None:
                    direct_joint_smoothed = smooth_tensor(direct_joint)
                if crop_features is not None:
                    if (minimal or component_ablation) and direct_intermediate:
                        direct_feature_joint_smoothed = smooth_tensor(
                            direct_feature_joint
                        )
                    else:
                        feature_joint_smoothed = smooth_tensor(feature_joint)
                        if direct_feature_joint_maps is not None:
                            direct_feature_joint_smoothed = smooth_tensor(
                                direct_feature_joint
                            )
        elif gaussian_backend == "scipy":
            base_np = base_dense[:, 0].cpu().numpy()
            foveal_np = anchored[:, 0].cpu().numpy()
            positive_np = positive_anchored[:, 0].cpu().numpy()
            if rate_scores is not None:
                rate_np = rate_dense[:, 0].cpu().numpy()
                joint_np = joint_anchored[:, 0].cpu().numpy()
                direct_joint_np = direct_joint[:, 0].cpu().numpy()
                if crop_features is not None:
                    feature_joint_np = feature_joint[:, 0].cpu().numpy()
                    direct_feature_joint_np = direct_feature_joint[:, 0].cpu().numpy()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                base_smoothed = list(executor.map(lambda image: gaussian_filter(image, sigma=4), base_np))
                if foveal is not None:
                    foveal_smoothed = list(executor.map(lambda image: gaussian_filter(image, sigma=4), foveal_np))
                    positive_smoothed = list(executor.map(lambda image: gaussian_filter(image, sigma=4), positive_np))
                if rate_scores is not None:
                    rate_smoothed = list(executor.map(
                        lambda image: gaussian_filter(image, sigma=4), rate_np
                    ))
                    if joint_maps is not None:
                        joint_smoothed = list(executor.map(
                            lambda image: gaussian_filter(image, sigma=4), joint_np
                        ))
                    if direct_joint_maps is not None:
                        direct_joint_smoothed = list(executor.map(
                            lambda image: gaussian_filter(image, sigma=4), direct_joint_np
                        ))
                    if crop_features is not None:
                        if (minimal or component_ablation) and direct_intermediate:
                            direct_feature_joint_smoothed = list(executor.map(
                                lambda image: gaussian_filter(image, sigma=4),
                                direct_feature_joint_np,
                            ))
                        else:
                            feature_joint_smoothed = list(executor.map(
                                lambda image: gaussian_filter(image, sigma=4),
                                feature_joint_np,
                            ))
                            if direct_feature_joint_maps is not None:
                                direct_feature_joint_smoothed = list(executor.map(
                                    lambda image: gaussian_filter(image, sigma=4),
                                    direct_feature_joint_np,
                                ))
        else:
            raise ValueError(f"unknown Gaussian backend: {gaussian_backend}")
        baseline[start:stop] = np.stack(base_smoothed)
        if foveal is not None:
            foveal[start:stop] = np.stack(foveal_smoothed)
            positive_foveal[start:stop] = np.stack(positive_smoothed)
        if rate_scores is not None:
            if rate_maps is not None:
                rate_maps[start:stop] = np.stack(rate_smoothed)
            if joint_maps is not None:
                joint_maps[start:stop] = np.stack(joint_smoothed)
            if direct_joint_maps is not None:
                direct_joint_maps[start:stop] = np.stack(direct_joint_smoothed)
            if crop_features is not None:
                selected_feature = (
                    direct_feature_joint_smoothed
                    if (minimal or component_ablation) and direct_intermediate else
                    feature_joint_smoothed
                )
                feature_joint_maps[start:stop] = np.stack(selected_feature)
                if direct_feature_joint_maps is not None:
                    direct_feature_joint_maps[start:stop] = np.stack(
                        direct_feature_joint_smoothed
                    )
    for array in (
        baseline, foveal, positive_foveal, rate_maps, joint_maps,
        direct_joint_maps, feature_joint_maps, direct_feature_joint_maps,
    ):
        if isinstance(array, np.memmap):
            array.flush()
    return (baseline, foveal, positive_foveal, rate_maps, joint_maps,
            direct_joint_maps, feature_joint_maps, direct_feature_joint_maps,
            histogram_error)


def metrics(maps, masks, records, workers):
    classes = np.asarray([record["class"] for record in records])

    def one(class_name):
        indices = np.flatnonzero(classes == class_name)
        target = masks[indices]
        prediction = maps[indices]
        return class_name, {
            "pixel_auroc": float(roc_auc_score(target.ravel(), prediction.ravel())),
            "pixel_aupro": float(cal_pro_score_fast(target, prediction)),
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        categories = dict(executor.map(one, sorted(set(classes))))
    return {
        "categories": categories,
        "mean": {
            key: float(np.mean([value[key] for value in categories.values()]))
            for key in ("pixel_auroc", "pixel_aupro")
        },
    }
