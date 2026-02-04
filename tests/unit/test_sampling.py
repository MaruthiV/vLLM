import pytest
import torch

from mini_vllm.sampling.sampling_params import SamplingParams
from mini_vllm.sampling.sampler import Sampler, SamplerOutput

class TestSamplingParams:

    def test_default_params(self):
        params = SamplingParams()
        assert params.temperature == 1.0
        assert params.top_p == 1.0
        assert params.top_k == -1
        assert params.max_tokens == 256

    def test_greedy_detection(self):
        assert SamplingParams(temperature=0).is_greedy
        assert SamplingParams(top_k=1).is_greedy
        assert not SamplingParams(temperature=1.0).is_greedy

    def test_validation_temperature(self):
        with pytest.raises(ValueError, match="temperature"):
            SamplingParams(temperature=-0.5)

    def test_validation_top_p(self):
        with pytest.raises(ValueError, match="top_p"):
            SamplingParams(top_p=1.5)

    def test_validation_max_tokens(self):
        with pytest.raises(ValueError, match="max_tokens"):
            SamplingParams(max_tokens=0)

    def test_validation_min_max_tokens(self):
        with pytest.raises(ValueError, match="min_tokens"):
            SamplingParams(min_tokens=100, max_tokens=50)

    def test_guided_decoding_mutual_exclusion(self):
        with pytest.raises(ValueError, match="guided decoding"):
            SamplingParams(
                guided_json={"type": "string"},
                guided_regex="[a-z]+",
            )

    def test_clone(self):
        params = SamplingParams(temperature=0.5, max_tokens=100, stop=["END"])
        cloned = params.clone()
        assert cloned.temperature == 0.5
        assert cloned.max_tokens == 100
        assert cloned.stop == ["END"]

        cloned.stop.append("STOP")
        assert params.stop == ["END"]

class TestSampler:

    @pytest.fixture
    def sampler(self):
        return Sampler(vocab_size=100)

    @pytest.fixture
    def uniform_logits(self):
        return torch.zeros(1, 100)

    @pytest.fixture
    def peaked_logits(self):
        logits = torch.full((1, 100), -10.0)
        logits[0, 42] = 10.0
        return logits

    def test_greedy_sampling(self, sampler, peaked_logits):
        params = SamplingParams(temperature=0)
        output = sampler.forward(peaked_logits, [params])
        assert output.token_ids[0] == 42

    def test_temperature_scaling(self, sampler, uniform_logits):

        params = SamplingParams(temperature=1.0)
        torch.manual_seed(42)
        output = sampler.forward(uniform_logits, [params])
        assert 0 <= output.token_ids[0] < 100

    def test_top_k_filtering(self, sampler):

        logits = torch.zeros(1, 100)
        for i in range(5):
            logits[0, i] = 10.0

        params = SamplingParams(temperature=1.0, top_k=5)
        torch.manual_seed(42)

        for _ in range(10):
            output = sampler.forward(logits.clone(), [params])
            assert output.token_ids[0] < 5

    def test_top_p_filtering(self, sampler):

        logits = torch.full((1, 100), -10.0)
        logits[0, 0] = 10.0

        params = SamplingParams(temperature=1.0, top_p=0.9)
        torch.manual_seed(42)

        count_zero = 0
        for _ in range(10):
            output = sampler.forward(logits.clone(), [params])
            if output.token_ids[0] == 0:
                count_zero += 1
        assert count_zero >= 8

    def test_batch_sampling(self, sampler):
        batch_logits = torch.zeros(3, 100)
        batch_logits[0, 10] = 10.0
        batch_logits[1, 20] = 10.0
        batch_logits[2, 30] = 10.0

        params = [SamplingParams(temperature=0) for _ in range(3)]
        output = sampler.forward(batch_logits, params)

        assert len(output.token_ids) == 3
        assert output.token_ids[0] == 10
        assert output.token_ids[1] == 20
        assert output.token_ids[2] == 30

    def test_logprobs_output(self, sampler, peaked_logits):
        params = SamplingParams(temperature=0, logprobs=5)
        output = sampler.forward(peaked_logits, [params])

        assert output.logprobs is not None
        assert len(output.logprobs) == 1
        assert output.logprobs[0] is not None
        assert len(output.logprobs[0]) == 5

        assert 42 in output.logprobs[0]

    def test_repetition_penalty(self, sampler):
        logits = torch.zeros(1, 100)
        logits[0, 0] = 5.0
        logits[0, 1] = 5.0

        params = SamplingParams(temperature=1.0, repetition_penalty=1.0)

        params_with_penalty = SamplingParams(temperature=1.0, repetition_penalty=2.0)

        torch.manual_seed(42)
        output = sampler.forward(logits.clone(), [params_with_penalty], [[0, 0, 0]])
