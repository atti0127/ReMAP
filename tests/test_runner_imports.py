import importlib


def test_remap_runner_entry_points_import():
    for module in (
        "rate_prompt.cache_dataset",
        "rate_prompt.cache_pyramid_rate",
        "rate_fovea.cache",
        "rate_fovea.cache_crop_semantic",
        "rate_fovea.evaluate_prompt_cache",
        "remap.infer",
        "remap.inference",
        "remap.report_results",
    ):
        importlib.import_module(module)
