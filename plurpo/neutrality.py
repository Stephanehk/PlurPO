"""DPO-Neutral's preference signal (App. "Neutral-DPO Implementation Details").

The v5 action-endorsement judge (gpt-5-mini, the paper's evaluation judge;
labels 0 = challenge, 1 = explicit endorse, 2 = implicit endorse, 3 = neutral)
labels every candidate. A candidate is "vetoed" iff its label is NOT 3, so any
neutral candidate is preferred to any other. An unparsable label is not a
confident "neutral" and is vetoed. Unlike PlurPO this signal is external: it
calls the judge API inside the training loop (needs OPENROUTER_API_KEY).

Same interface as plurpo.stakeholders.StakeholderSimulator, so it plugs into
plurpo.online.OnlineTrainer; there is no stakeholder panel.
"""

from plurpo.concurrency import concurrent_map
from plurpo.judges import endorse_one

NEUTRAL_LABEL = 3


class NeutralityJudge:
    """concurrency: in-flight judge calls (bounds the API fan-out, independent
    of the trainer's vLLM-tuned concurrency)."""

    def __init__(self, client, concurrency=16):
        self.client = client
        self.concurrency = concurrency
        self.n_invalid = 0

    def prime_panels(self, posts, concurrency):
        return

    def panel(self, post):
        return []

    def veto_batch(self, items, concurrency):
        """-> (vetoed, excluded): vetoed = candidates not labelled neutral, in
        response order; nothing is excluded."""
        tasks = [(i, post, r) for i, (post, rs) in enumerate(items) for r in rs]
        labels = concurrent_map(lambda t: endorse_one(self.client, "v5", t[1], t[2])[0],
                                tasks, self.concurrency)
        self.n_invalid += sum(1 for lab in labels if lab is None)
        vetoed = [[] for _ in items]
        for (i, _post, r), lab in zip(tasks, labels):
            if lab != NEUTRAL_LABEL:
                vetoed[i].append(r)
        return vetoed, [[] for _ in items]
