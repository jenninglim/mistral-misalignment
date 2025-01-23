from typing import List, Optional, Tuple

import torch


@torch.inference_mode()
def generate(
    encoded_prompts: List[List[int]],
    model,
    *,
    max_tokens: int,
    temperature: float,
    chunk_size: Optional[int] = None,
    eos_id: Optional[int] = None,
) -> Tuple[List[List[int]], List[List[float]]]:

    model = model.eval()
    B, V = len(encoded_prompts), model.args.vocab_size

    seqlens = [len(x) for x in encoded_prompts]

    # Bookkeeping
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len


    # Encode prompt by chunks
    prompt_chunks = [p for p in encoded_prompts]
    assert all(len(p) > 0 for p in prompt_chunks)
    prompt_tensor = torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long)
    prelogits = model.forward(
        prompt_tensor,
        seqlens=[len(p) for p in prompt_chunks],
    )

    last_token_prelogits = prelogits[[-1]]
    assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tensors = []
    is_finished = torch.tensor([False for _ in range(B)])

    assert last_token_prelogits is not None
    for _ in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)

        if eos_id is not None:
            is_finished = is_finished | (next_token == eos_id).cpu()

        if is_finished.all():
            break

        generated_tensors.append(next_token[:, None])
        input_tensor = torch.cat(
            [prompt_tensor,
             torch.stack(generated_tensors, dim=1)[0, :, 0]],
            dim=0
        )
        last_token_prelogits = model.forward(
            input_tensor,
            seqlens=[input_tensor.shape[0]],
        )[[-1]]
        assert last_token_prelogits.shape == (B, V), f"{last_token_prelogits.shape} {B} {V}"

    generated_tokens: List[List[int]]
    if generated_tensors:
        generated_tokens = torch.cat(generated_tensors, 1).tolist()
    else:
        generated_tokens = []

    return generated_tokens


def sample(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = torch.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


def sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    assert 0 <= p <= 1

    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, next_token)
