import unittest
from unittest import mock

import torch

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.utils.attr_dict import AttrDict


class LearnerValidBatchingTest(unittest.TestCase):
    def _make_learner(self, *, batch_size: int = 1024, num_batches_per_epoch: int = 1, target_valid: int = 0):
        learner = Learner.__new__(Learner)
        learner.cfg = AttrDict(
            batch_size=batch_size,
            num_batches_per_epoch=num_batches_per_epoch,
            target_valid_samples_per_update=target_valid,
        )
        learner._pending_valid_batch = None
        learner._pending_valid_samples = 0
        return learner

    def test_filter_invalid_samples_keeps_only_valid_rows(self):
        learner = self._make_learner()
        batch = TensorDict(
            {
                "actions": torch.tensor([0, 1, 1, 0, 1, 0], dtype=torch.int64),
                "valids": torch.tensor([True, False, True, False, False, True], dtype=torch.bool),
            }
        )

        filtered, valid_count = learner._filter_invalid_samples_from_prepared_batch(batch)

        self.assertEqual(valid_count, 3)
        self.assertEqual(filtered["actions"].tolist(), [0, 1, 0])
        self.assertTrue(torch.all(filtered["valids"]))

    def test_filter_invalid_samples_does_not_use_torch_nonzero(self):
        learner = self._make_learner()
        batch = TensorDict(
            {
                "actions": torch.tensor([0, 1, 1, 0], dtype=torch.int64),
                "valids": torch.tensor([True, False, True, False], dtype=torch.bool),
            }
        )

        with mock.patch("torch.Tensor.nonzero", side_effect=RuntimeError("nonzero must not be used")):
            filtered, valid_count = learner._filter_invalid_samples_from_prepared_batch(batch)

        self.assertEqual(valid_count, 2)
        self.assertEqual(filtered["actions"].tolist(), [0, 1])
        self.assertTrue(torch.all(filtered["valids"]))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for mixed device regression test")
    def test_filter_invalid_samples_supports_mixed_cpu_gpu_tensors(self):
        learner = self._make_learner()
        batch = TensorDict(
            {
                "actions": torch.tensor([0, 1, 2, 3], dtype=torch.int64, device="cuda"),
                "valids": torch.tensor([True, False, True, False], dtype=torch.bool, device="cuda"),
                "dones_cpu": torch.tensor([0.0, 1.0, 0.0, 1.0], dtype=torch.float32, device="cpu"),
            }
        )

        filtered, valid_count = learner._filter_invalid_samples_from_prepared_batch(batch)

        self.assertEqual(valid_count, 2)
        self.assertEqual(filtered["actions"].cpu().tolist(), [0, 2])
        self.assertEqual(filtered["dones_cpu"].tolist(), [0.0, 0.0])
        self.assertEqual(filtered["actions"].device.type, "cuda")
        self.assertEqual(filtered["dones_cpu"].device.type, "cpu")
        self.assertTrue(torch.all(filtered["valids"]))

    def test_accumulate_valid_samples_until_target_then_train(self):
        learner = self._make_learner(target_valid=4)

        batch_a = TensorDict(
            {
                "actions": torch.tensor([0, 1, 0], dtype=torch.int64),
                "valids": torch.ones(3, dtype=torch.bool),
            }
        )
        batch_b = TensorDict(
            {
                "actions": torch.tensor([1, 1, 0], dtype=torch.int64),
                "valids": torch.ones(3, dtype=torch.bool),
            }
        )

        should_train, train_batch, train_size = learner._accumulate_valid_samples_for_training(batch_a, 3)
        self.assertFalse(should_train)
        self.assertIsNone(train_batch)
        self.assertEqual(train_size, 0)

        should_train, train_batch, train_size = learner._accumulate_valid_samples_for_training(batch_b, 3)
        self.assertTrue(should_train)
        self.assertEqual(train_size, 4)
        self.assertEqual(train_batch["actions"].tolist(), [0, 1, 0, 1])
        self.assertEqual(learner._pending_valid_samples, 2)
        self.assertEqual(learner._pending_valid_batch["actions"].tolist(), [1, 0])

    def test_resolve_train_batch_size_prefers_cfg_batch_size_when_divisible(self):
        learner = self._make_learner(batch_size=1024, num_batches_per_epoch=1)
        resolved = learner._resolve_train_batch_size(experience_size=2048)
        self.assertEqual(resolved, 1024)

    def test_resolve_train_batch_size_falls_back_to_experience_size(self):
        learner = self._make_learner(batch_size=1024, num_batches_per_epoch=1)
        resolved = learner._resolve_train_batch_size(experience_size=1500)
        self.assertEqual(resolved, 1500)


if __name__ == "__main__":
    unittest.main()
