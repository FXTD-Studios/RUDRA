"""The fill says so before it embeds a new prompt, and setup says what to do when PyTorch
is missing or CPU-only (6 Nov 2026 review). Torch-free."""
from __future__ import annotations

import sys

from rudra.genfill.run import prepare_prompts
from rudra.genfill.setup import no_gpu_reason


class FakeBank:
    """A PromptBank stand-in: which prompts are cached, which were fetched, whether it was released."""

    def __init__(self, cached):
        """The prompts cached from the start."""
        self.cached, self.got, self.released = set(cached), [], False

    def has(self, text):
        """Whether a prompt is cached."""
        return text in self.cached

    def get(self, text):
        """Fetch a prompt (it is cached from then on)."""
        self.got.append(text)
        self.cached.add(text)

    def release(self):
        """Release the encoder."""
        self.released = True


def test_a_new_prompt_is_announced_before_it_is_embedded():
    """A prompt not in the cache is announced once, before it is embedded."""
    bank, events = FakeBank({"base", "clear blue sky"}), []
    prepare_prompts(bank, ["base", "my own sky", "", "clear blue sky", "my own sky"], lambda **o: events.append(o))
    assert [e["event"] for e in events] == ["embedding"]
    assert events[0]["prompts"] == ["my own sky"]
    assert bank.got == ["base", "my own sky", "clear blue sky"]   # each once, empty ones skipped
    assert bank.released


def test_cached_prompts_are_silent():
    """Cached prompts load with no event."""
    bank, events = FakeBank({"base", "sun"}), []
    prepare_prompts(bank, ["base", "sun", ""], lambda **o: events.append(o))
    assert events == [] and bank.released


def test_no_gpu_reason_names_the_actual_problem():
    """No PyTorch, a CPU-only build, no GPU and a small GPU each get their own reason."""
    missing = no_gpu_reason({"torch": None, "device": "cpu", "memory_gb": None})
    assert "PyTorch is not installed" in missing and sys.executable in missing
    cpu = no_gpu_reason({"torch": "2.5.1+cpu", "device": "cpu", "memory_gb": None})
    assert "2.5.1+cpu" in cpu and "CUDA build" in cpu
    gpu_less = no_gpu_reason({"torch": "2.5.1+cu128", "device": "cpu", "memory_gb": None})
    assert "found no CUDA GPU" in gpu_less and "CPU-only" not in gpu_less
    assert "needs 16 GB" in no_gpu_reason({"torch": "2.5.1", "device": "cuda", "memory_gb": 8.0})
