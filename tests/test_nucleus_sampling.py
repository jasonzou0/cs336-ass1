import unittest
import torch

from .adapters import top_p_sampling


class TestTopPSampling(unittest.TestCase):

    def test_basic_nucleus_filtering(self):
        """
        Tests if sampling is constrained to the nucleus.
        With these logits, the first three tokens have a combined probability of ~0.99.
        With p=0.95, we should only ever sample from tokens 0, 1, or 2.
        """
        logits = torch.tensor([10.0, 8.0, 6.0, 2.0, 1.0]) # vocab size of 5
        # Probabilities are approx: [0.88, 0.12, 0.016, 0.0, 0.0]
        # Cumulative probs: [0.88, 0.999, ...]
        p = 0.95

        # We expect the nucleus to contain only tokens 0 and 1.
        expected_nucleus = {0, 1}

        for _ in range(100): # Run multiple times due to stochastic nature
            token = top_p_sampling(logits, p=p)
            self.assertIn(token.item(), expected_nucleus)

    def test_high_certainty_distribution(self):
        """
        Tests if sampling is deterministic when one token is highly probable.
        Token 0 has a probability > 0.999. With p=0.95, it should be the only choice.
        """
        logits = torch.tensor([20.0, 1.0, 0.5, 0.1])
        p = 0.95

        for _ in range(100):
            token = top_p_sampling(logits, p=p)
            self.assertEqual(token.item(), 0)

    def test_p_equals_one(self):
        """
        Tests if p=1.0 allows any token to be sampled (i.e., regular sampling).
        Although unlikely, the last token (index 4) should be a possible outcome.
        """
        logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        p = 1.0

        # Over many runs, we expect to see all possible tokens sampled
        sampled_tokens = set()
        for _ in range(500):
            token = top_p_sampling(logits, p=p)
            sampled_tokens.add(token.item())

        # Check if all tokens were eventually sampled
        self.assertEqual(sampled_tokens, {0, 1, 2, 3, 4})

    def test_p_is_very_low(self):
        """
        Tests if a very low p value makes the sampling greedy (always picks the top token).
        """
        logits = torch.tensor([10.0, 8.0, 6.0, 2.0, 1.0])
        p = 0.01 # This p is smaller than the probability of the most likely token.

        for _ in range(100):
            token = top_p_sampling(logits, p=p)
            # Should always pick the token with the highest logit (index 0)
            self.assertEqual(token.item(), 0)

    def test_temperature_very_low_is_greedy(self):
        """
        Tests that very low temperature (approaching 0) always picks the highest logit token (greedy sampling).
        Note: We use a very small temperature instead of 0 to avoid division by zero.
        """
        logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        p = 1.0  # Allow all tokens
        temperature = 1e-8  # Very small temperature, effectively greedy

        for _ in range(100):
            token = top_p_sampling(logits, p=p, temperature=temperature)
            # Should always pick the token with the highest logit (index 0)
            self.assertEqual(token.item(), 0)

    def test_temperature_low_makes_sampling_greedy(self):
        """
        Tests that very low temperature makes the distribution more peaked,
        leading to more greedy behavior.
        """
        logits = torch.tensor([5.0, 4.5, 4.0, 3.0, 2.0])
        p = 1.0  # Allow all tokens
        temperature = 0.1

        # With very low temperature, should heavily favor the top token
        token_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for _ in range(1000):
            token = top_p_sampling(logits, p=p, temperature=temperature)
            token_counts[token.item()] += 1

        # Token 0 should be sampled much more frequently than others
        self.assertGreater(token_counts[0], 800)  # Should be very dominant
        self.assertLess(token_counts[4], 10)      # Lowest token should be rare

    def test_temperature_high_makes_sampling_uniform(self):
        """
        Tests that high temperature flattens the distribution,
        making sampling more uniform across tokens.
        """
        logits = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
        p = 1.0  # Allow all tokens
        temperature = 5.0

        # With high temperature, distribution should be more uniform
        sampled_tokens = set()
        for _ in range(500):
            token = top_p_sampling(logits, p=p, temperature=temperature)
            sampled_tokens.add(token.item())

        # Should sample from all tokens with high temperature
        self.assertEqual(sampled_tokens, {0, 1, 2, 3, 4})

    def test_temperature_one_is_baseline(self):
        """
        Tests that temperature=1.0 behaves as the baseline (unchanged probabilities).
        This should behave similarly to existing nucleus sampling tests.
        """
        logits = torch.tensor([10.0, 8.0, 6.0, 2.0, 1.0])
        p = 0.95
        temperature = 1.0

        # With temperature=1.0, should behave like the basic nucleus test
        expected_nucleus = {0, 1}
        for _ in range(100):
            token = top_p_sampling(logits, p=p, temperature=temperature)
            self.assertIn(token.item(), expected_nucleus)

    def test_temperature_with_nucleus_interaction(self):
        """
        Tests how temperature interacts with nucleus sampling by using different p values.
        Low temperature + high p should still be greedy due to temperature dominance.
        """
        logits = torch.tensor([6.0, 5.0, 4.0, 3.0, 2.0])
        p = 0.9  # Allow broader nucleus
        low_temperature = 0.1
        high_temperature = 3.0

        # Low temperature should still be greedy despite high p
        low_temp_tokens = set()
        for _ in range(100):
            token = top_p_sampling(logits, p=p, temperature=low_temperature)
            low_temp_tokens.add(token.item())
        
        # High temperature should use more of the nucleus
        high_temp_tokens = set()
        for _ in range(500):
            token = top_p_sampling(logits, p=p, temperature=high_temperature)
            high_temp_tokens.add(token.item())

        # Low temperature should sample from fewer tokens
        # High temperature should sample from more tokens
        self.assertLessEqual(len(low_temp_tokens), len(high_temp_tokens))
        self.assertGreater(len(high_temp_tokens), 2)  # Should use multiple tokens


# This allows running the tests from the command line
if __name__ == '__main__':
    unittest.main()