"""1.3.2 (26 Nov 2026): a new prompt is read faster (rudra/genfill/model.py PromptBank): the text encoder on the
prompt's own tokens only, and on the GPU when the GPU has the room. Needs PyTorch (and transformers for the
encoder test); the real umT5's timing and agreement are measured by tmp/09V_test/v4_gates/run_prompt_check.py."""
import json

import pytest

torch = pytest.importorskip("torch")

from rudra.genfill import model as M  # noqa: E402


def _tiny_umt5():
    transformers = pytest.importorskip("transformers")
    cfg = transformers.UMT5Config(vocab_size=64, d_model=16, d_kv=4, d_ff=32, num_layers=2, num_heads=4,
                                  relative_attention_num_buckets=8, relative_attention_max_distance=16,
                                  dropout_rate=0.0, feed_forward_proj="gated-gelu")
    torch.manual_seed(0)
    return transformers.UMT5EncoderModel(cfg).eval()


def test_the_prompts_own_tokens_give_the_padded_encoding():
    enc = _tiny_umt5()
    ids = torch.randint(2, 64, (1, 20))
    mask = torch.zeros(1, 20, dtype=torch.long)
    mask[0, :7] = 1
    ids[0, 7:] = 0                                                       # the pad token
    with torch.no_grad():
        full = enc(ids, mask).last_hidden_state.float()
        full[:, 7:] = 0                                                  # as before 1.3.2
        short = M.encode_tokens(enc, ids, mask, "cpu")
    assert short.shape == full.shape == (1, 20, 16)
    assert torch.allclose(short, full, atol=1e-5)
    assert torch.count_nonzero(short[:, 7:]) == 0
    left = torch.zeros(1, 20, dtype=torch.long)
    left[0, 13:] = 1                                                     # a tokenizer padding on the left: whole
    with torch.no_grad():
        whole = enc(ids, left).last_hidden_state.float()
        got = M.encode_tokens(enc, ids, left, "cpu")
    whole[:, 7:] = 0
    assert torch.allclose(got, whole, atol=1e-6)


def test_text_encoder_parameters_come_from_the_safetensors_headers(tmp_path):
    d = tmp_path / "text_encoder"
    d.mkdir()
    for name, tensors in (("a.safetensors", {"w": [2, 3], "b": [4]}), ("b.safetensors", {"v": [5, 1, 2]})):
        head = {k: {"dtype": "BF16", "shape": s, "data_offsets": [0, 0]} for k, s in tensors.items()}
        head["__metadata__"] = {"format": "pt"}
        raw = json.dumps(head).encode()
        (d / name).write_bytes(len(raw).to_bytes(8, "little") + raw)
    assert M.text_encoder_parameters(tmp_path) == 6 + 4 + 10
    assert M.text_encoder_parameters(tmp_path / "nothing") == 0
    assert M.text_encoder_parameters(None) == 0


def test_the_gpu_is_used_only_with_the_room(monkeypatch):
    params = 1000
    need = params * 2 + M.TEXT_ENCODER_MARGIN                             # bfloat16: 2 bytes a parameter
    monkeypatch.setattr(M, "text_encoder_parameters", lambda base: params)
    monkeypatch.setattr(M.torch.cuda, "is_available", lambda: True)
    free = {"now": need}
    monkeypatch.setattr(M, "gpu_free_bytes", lambda: free["now"])
    bank = M.PromptBank("base", None, device="auto")
    assert bank._encoder_room() == "cuda"
    free["now"] = need - 1
    assert bank._encoder_room() == "cpu"                                  # no room, nothing to move
    moved = []

    def make_room():
        moved.append(1)
        free["now"] = need                                                # the transformer stepped off the GPU
    bank.make_room = make_room
    assert bank._encoder_room() == "cuda" and moved == [1]
    free["now"], bank.make_room = 0, lambda: None
    assert bank._encoder_room() == "cpu"
    monkeypatch.setattr(M.torch.cuda, "is_available", lambda: False)
    assert M.PromptBank("base", None, device="auto")._encoder_room() == "cpu"


def test_the_memory_cap_bounds_what_counts_as_free(monkeypatch):
    if not torch.cuda.is_available():
        assert M.gpu_free_bytes() == 0
        return
    monkeypatch.setenv("RUDRA_GENFILL_MEMORY_CAP_GB", "0.001")
    assert M.gpu_free_bytes() <= int(0.001 * 2 ** 30)                   # the cap less what PyTorch holds already
