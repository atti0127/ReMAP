import unittest

import torch

from rate import adapt_rate, align_auxiliary_patch_features, make_pyramid_auxiliary_features


class RATETests(unittest.TestCase):
    def test_public_wrapper_preserves_scores_and_inputs(self):
        torch.manual_seed(17)
        text = torch.randn(2, 8)
        high = torch.randn(1, 16, 8)
        auxiliary = torch.randn(1, 9, 8)
        scores = torch.linspace(0.05, 0.95, 16).reshape(1, 16)
        originals = [value.clone() for value in (text, high, auxiliary, scores)]

        result = adapt_rate(text, high, auxiliary, scores)

        transported = result.transported_patch_scores
        self.assertIsNotNone(transported)
        self.assertTrue(
            torch.equal(
                torch.sort(transported, dim=-1).values,
                torch.sort(scores, dim=-1).values,
            )
        )
        for value, original in zip((text, high, auxiliary, scores), originals):
            self.assertTrue(torch.equal(value, original))

    def test_leave_one_voter_out_preserves_original_score_values(self):
        torch.manual_seed(23)
        text = torch.randn(2, 8)
        high = torch.randn(1, 16, 8)
        auxiliary = torch.randn(1, 9, 8)
        scores = torch.rand(1, 16)
        configurations = (
            ("rarity_corrected_rank", "extrapolated_fusion_rank"),
            ("cross_scale_affinity_rank", "extrapolated_fusion_rank"),
            ("cross_scale_affinity_rank", "rarity_corrected_rank"),
        )
        for voters in configurations:
            with self.subTest(voters=voters):
                result = adapt_rate(
                    text, high, auxiliary, scores, voters=voters
                )
                transported = result.transported_patch_scores
                self.assertTrue(torch.equal(
                    torch.sort(transported, dim=-1).values,
                    torch.sort(scores, dim=-1).values,
                ))
                self.assertEqual(
                    result.diagnostics["aggregation_branches"], 2.0
                )

    def test_rate_rejects_invalid_voter_sets(self):
        text = torch.randn(2, 8)
        high = torch.randn(1, 16, 8)
        auxiliary = torch.randn(1, 9, 8)
        scores = torch.rand(1, 16)
        with self.assertRaises(ValueError):
            adapt_rate(text, high, auxiliary, scores, voters=())
        with self.assertRaises(ValueError):
            adapt_rate(
                text, high, auxiliary, scores,
                voters=("cross_scale_affinity_rank", "unknown"),
            )

    def test_sequential_and_batched_search_preserve_the_same_histogram(self):
        torch.manual_seed(31)
        text = torch.randn(2, 8)
        high = torch.randn(1, 25, 8)
        auxiliary = torch.randn(1, 9, 8)
        scores = torch.rand(1, 25)
        outputs = {}
        for evaluation in ("sequential", "batched"):
            with self.subTest(evaluation=evaluation):
                result = adapt_rate(
                    text, high, auxiliary, scores,
                    candidate_evaluation=evaluation,
                )
                self.assertTrue(torch.equal(
                    torch.sort(result.transported_patch_scores, dim=-1).values,
                    torch.sort(scores, dim=-1).values,
                ))
                self.assertEqual(
                    result.diagnostics["candidate_evaluation_batched"],
                    float(evaluation == "batched"),
                )
                outputs[evaluation] = result.transported_patch_scores
        self.assertTrue(torch.equal(outputs["sequential"], outputs["batched"]))
        with self.assertRaises(ValueError):
            adapt_rate(
                text, high, auxiliary, scores,
                candidate_evaluation="unknown",
            )

    def test_pyramid_auxiliary_is_aligned_and_parameter_free(self):
        torch.manual_seed(29)
        identity = torch.randn(2, 25, 8)
        auxiliary = make_pyramid_auxiliary_features(identity, auxiliary_side=3)
        aligned = align_auxiliary_patch_features(auxiliary, 25)

        self.assertEqual(auxiliary.shape, (2, 9, 8))
        self.assertEqual(aligned.shape, identity.shape)
        self.assertTrue(torch.isfinite(aligned).all())
        self.assertTrue(torch.allclose(aligned.norm(dim=-1), torch.ones(2, 25)))
        bicubic = make_pyramid_auxiliary_features(
            identity, auxiliary_side=3, interpolation="bicubic"
        )
        self.assertEqual(bicubic.shape, auxiliary.shape)

    def test_inference_only_rate_is_score_identical(self):
        for seed in range(5):
            torch.manual_seed(seed)
            text = torch.randn(2, 16)
            high = torch.randn(1, 36, 16)
            auxiliary = torch.randn(1, 16, 16)
            scores = torch.rand(1, 36)
            diagnostic = adapt_rate(text, high, auxiliary, scores)
            inference = adapt_rate(
                text, high, auxiliary, scores, inference_only=True
            )
            self.assertTrue(torch.equal(
                diagnostic.transported_patch_scores,
                inference.transported_patch_scores,
            ))
            self.assertEqual(inference.diagnostics, {})


if __name__ == "__main__":
    unittest.main()
