"""Shared concurrency helper for the training pipeline.

`concurrent_map` is an order-preserving, fail-fast parallel map over a thread
pool. It exists to fan blocking LLM calls (vLLM HTTP requests, which `vllm_sim`
documents as thread-safe) out across many items at once, so the server's
continuous batching saturates the GPU instead of seeing ~one request at a time.

Assumptions:
  - `fn` is safe to call concurrently from multiple threads (true for the
    vLLM-backed `call_llm` / policy-sampler closures, the only callers).
  - Results must line up with inputs positionally; callers rely on this to
    regroup staged results back to their prompts.
  - Any exception in a worker should surface immediately (fail-fast), matching
    the rest of the codebase's "let it raise" policy — a swallowed veto/parse
    error would silently corrupt the preference dataset.
"""

import concurrent.futures


def concurrent_map(fn, items, max_workers):
    """Apply `fn` to each element of `items` and return the results in input
    order. Runs up to `max_workers` calls concurrently on a thread pool.

    A `max_workers <= 1` (or a single item) runs inline with no pool, which is
    how the sequential code path is recovered exactly. The first worker
    exception propagates out of this call; remaining work is abandoned.
    """
    items = list(items)
    if not items:
        return []
    if max_workers <= 1 or len(items) == 1:
        return [fn(x) for x in items]

    results = [None] * len(items)
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(max_workers)) as pool:
        future_to_index = {pool.submit(fn, x): i for i, x in enumerate(items)}
        for future in concurrent.futures.as_completed(future_to_index):
            results[future_to_index[future]] = future.result()
    return results
