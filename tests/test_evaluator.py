import torch
import torch.nn as nn
import unittest

from cs336_basics.evaluator import Evaluator


class MockModel(nn.Module):
    """Mock model that returns a fixed loss per token."""

    def __init__(self, loss_per_token: float = 1.0):
        super().__init__()
        self.loss_per_token = loss_per_token

    def forward(self, input_ids, target_ids):
        # Return loss_per_token as a scalar tensor
        return torch.tensor(self.loss_per_token, device=input_ids.device)


class MockDataLoader:
    """Mock data loader that yields batches with known shapes."""

    def __init__(self, batches, device="cpu"):
        self.batches = batches
        self.device = device

    def __iter__(self):
        for batch_size, seq_len in self.batches:
            input_ids = torch.zeros((batch_size, seq_len), device=self.device)
            target_ids = torch.zeros((batch_size, seq_len), device=self.device)
            yield input_ids, target_ids


class TestEvaluator(unittest.TestCase):

    def test_avg_loss_single_batch(self):
        """Test avg_loss with a single batch."""
        loss_per_token = 2.5
        batch_size = 4
        seq_len = 10

        model = MockModel(loss_per_token)
        data_loader = MockDataLoader([(batch_size, seq_len)])
        evaluator = Evaluator(model, data_loader)

        avg_loss = evaluator.avg_loss()

        self.assertAlmostEqual(avg_loss, loss_per_token, places=6)

    def test_avg_loss_multiple_batches_same_size(self):
        """Test avg_loss with multiple batches of the same size."""
        loss_per_token = 1.5
        batch_size = 3
        seq_len = 8
        num_batches = 5

        model = MockModel(loss_per_token)
        data_loader = MockDataLoader([(batch_size, seq_len)] * num_batches)
        evaluator = Evaluator(model, data_loader)

        avg_loss = evaluator.avg_loss()

        self.assertAlmostEqual(avg_loss, loss_per_token, places=6)

    def test_avg_loss_multiple_batches_different_losses(self):
        """Test avg_loss with multiple batches having different per-batch losses."""
        batch_size = 4
        seq_len = 6
        num_batches = 3
        losses_per_batch = [1.0, 2.0, 3.0]
        
        # Expected: all batches same size, so simple average of losses
        expected_avg = sum(losses_per_batch) / len(losses_per_batch)  # (1.0 + 2.0 + 3.0) / 3 = 2.0
        
        class VaryingLossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.call_count = 0
                
            def forward(self, input_ids, target_ids):
                loss = losses_per_batch[self.call_count]
                self.call_count += 1
                return torch.tensor(loss, device=input_ids.device)
        
        model = VaryingLossModel()
        data_loader = MockDataLoader([(batch_size, seq_len)] * num_batches)
        evaluator = Evaluator(model, data_loader)
        
        avg_loss = evaluator.avg_loss()
        
        self.assertAlmostEqual(avg_loss, expected_avg, places=6)


if __name__ == "__main__":
    unittest.main()