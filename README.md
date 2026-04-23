# Arc Sentry

**Whitebox prompt injection detector for self-hosted open-weight LLMs.**

Arc Sentry detects prompt injection attacks on self-hosted LLMs by calibrating on your normal traffic, then flagging prompts that push the model's internal state away from that baseline. Calibration takes ~20 warmup prompts. Detection catches injection patterns, jailbreak attempts, and multi-turn campaigns that input-embedding filters miss.

It is **not** a universal harmful-content classifier. It is a drift detector for *your* deployment.

Available via `pip install arc-sentry`. Requires a GPU.

## Benchmark

On a calibrated SaaS deployment benchmark (130 prompts, held-out test split; detection threshold and centroid locked before evaluation):

| | Detection | False Positive Rate |
| --- | --- | --- |
| **Arc Sentry** | **92%** | **0%** |
| LLM Guard | 70% | 3.3% |

Additional results on the same deployment:

- **Crescendo multi-turn**: caught at Turn 2 with 75% confidence (LLM Guard: 0 of 8 turns caught)

*Baseline LLM Guard run with default configuration on the same 130-prompt dataset.*

## What this is / what this isn't

**What it is:**
- A whitebox monitor that hooks into the model's residual stream before `generate()`
- Deployment-specific: calibrates on your normal traffic, detects drift from that baseline
- Effective against injection attempts that share structural patterns with your normal traffic's *language* but not its internal *representation*

**What it isn't:**
- A universal content filter or harmful-content classifier
- A drop-in solution for arbitrary attack datasets. On out-of-distribution benchmarks (JailbreakBench attacks mixed with OpenOrca benign traffic) Arc Sentry detects 90% of attacks but with 93% FPR — because the benign traffic isn't what it was calibrated on. Arc Sentry works on *your* traffic, calibrated on *your* deployment, not on arbitrary mixed datasets.
- Usable on closed-model APIs (GPT-4, Claude, Gemini) — it needs access to model internals
- Usable without a GPU

## Who this is for

Best fit: teams self-hosting Mistral, Llama, or Qwen in a customer-facing product, where calibrating on ~20 samples of your normal traffic is feasible and you have GPU headroom for a whitebox sidecar. If you're running a closed-model API (GPT, Claude, Gemini) or have no access to model internals, look elsewhere — Arc Sentry needs model internals to work.

## Install

```
pip install arc-sentry
```

## Quickstart

See [`arc_sentry_quickstart.ipynb`](arc_sentry_quickstart.ipynb) for a runnable end-to-end example, or:

```python
from arc_sentry import ArcSentryV3, MistralAdapter
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.2",
    torch_dtype=torch.float16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")

adapter = MistralAdapter(model, tokenizer)
sentry = ArcSentryV3(adapter)

# warmup_prompts: list of ~20 strings representing your normal traffic
sentry.calibrate(warmup_prompts)

# Block before generate() is called
response, result = sentry.observe_and_block(user_prompt)
if result["blocked"]:
    # model.generate() was never called; request was blocked pre-generation.
    pass
```

## How it works

Three detection layers, in order:

1. **Phrase check** — 80+ injection patterns, zero latency. Catches obvious attempts before anything else runs.
2. **Geometric detection** — mean-pooled hidden states at layer L=16 on Mistral-7B (layer selection documented in Nine 2026; see *Research background*), L2-normalized, Fisher-Rao distance to calibrated centroid. Blocks when distance crosses threshold. Catches injections with no explicit adversarial language.
3. **Session D(t) monitor** — stability scalar over rolling request history. Catches the gradual multi-turn campaigns (Crescendo-style) mentioned in the opener, which are invisible to single-request detection.

Detection pipeline:

```
1. Mean-pool hidden states at layer L
2. L2-normalize: h = h / ||h||
3. Fisher-Rao distance to warmup centroid
4. Distance > threshold → BLOCK (model.generate() is never called)
```

## Requirements

**Status:** beta. Validated on Mistral 7B, Qwen 2.5 7B, and Llama 3.1 8B. Public API stable within 3.x.

- Python 3.8+
- GPU with enough VRAM for your target model (Mistral-7B needs ~14GB in fp16)
- A self-hosted open-weight model: Mistral, Llama, or Qwen families validated. Other transformer architectures are likely compatible but untested.
- 20 calibration prompts that represent your normal deployment traffic

## License

Arc Sentry is dual-licensed.

**Open source (free):** [AGPL-3.0](LICENSE). Use it for research, evaluation, personal projects, or inside your organization if you're comfortable with AGPL-3.0 terms. If you modify Arc Sentry or build it into a networked service, the AGPL requires you to release your modifications under AGPL-3.0.

**Commercial:** If you want to embed Arc Sentry in a proprietary product or service, or your legal team cannot approve AGPL, you need a commercial license. See [COMMERCIAL-LICENSE.md](COMMERCIAL-LICENSE.md) or email 9hannahnine@gmail.com.

Patent pending. Methods covered by provisional patent applications filed by Hannah Nine / Bendex Geometry LLC (priority dates November 2025, February 2026, March 2026). A commercial license includes a patent grant for the licensed deployment.

## Research background

Arc Sentry's detection method is grounded in the second-order Fisher manifold framework (Nine 2026). For the mathematical foundations, see [bendexgeometry.com/theory](https://bendexgeometry.com/theory).

---

Bendex Geometry LLC · 2026 Hannah Nine · [bendexgeometry.com](https://bendexgeometry.com)
