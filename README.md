# promptxray

Find out which line of your prompt is causing your classifier's errors.

Every eval tool tells you your prompt scores 0.83. None of them tell you *which
part of the prompt* earned that number. promptxray removes each block of your
prompt one at a time, re-runs the dataset, and shows you the damage.

![Report showing the prompt with each block coloured by its measured effect](docs/report.png)

The report is your prompt, coloured block by block. Blue means the score drops
without that block. Amber means the score goes *up* without it. Hover any block
and it tells you why it got that colour, which examples flipped, and what to do
about it.

## The 30-second version

No API key needed for the demo — a built-in mock classifier lets you see the
whole thing work offline.

```bash
pip install promptxray

promptxray ablate \
  --prompt examples/prompt.txt \
  --data examples/data.csv \
  --provider mock \
  --report report.html
```

```
baseline macro F1 0.836 on 43 examples
ablation subset   43 examples

  #3   -0.125  carries its weight   Treat ALL-CAPS messages as urgent, because customers...
  #1   +0.000  no effect            You are a message triage assistant for a customer...
  #2   +0.000  no effect            Classify the message into exactly one of: spam, urgent...
  #5   +0.000  no effect            Be thorough and think carefully about the context...
  #6   +0.000  no effect            Remember that accuracy matters a lot to our team...
  #4   +0.096  hurts the score      Ignore promotional wording in quotes - people often...

4 of 6 tested blocks changed nothing at all.
```

Read that bottom line again. Two thirds of that prompt is decoration, and one
instruction someone added in good faith is actively costing 0.096 macro F1.

## How it works

The technique is occlusion, borrowed from explainable ML. There you hide one
input feature and watch the prediction move; here the features are the blocks
of your prompt.

1. Your prompt is split into blocks on blank lines. One block is one
   instruction, one few-shot example, or one formatting rule.
2. The full prompt runs over your labelled dataset. That is the baseline.
3. Each block is removed on its own and the prompt is re-run.
4. The change in macro F1, and the individual examples that flipped, are
   attributed to that block.

Two things keep this affordable:

**Error-enriched subsets.** Ablated runs use `--subset-size` examples, not the
whole dataset: every example the baseline got wrong, plus a sample of the ones
it got right so that a block which *breaks* working cases still shows up. Every
delta is measured against the baseline on that same subset, never against the
full-dataset number.

**A disk cache.** Every answer is stored under a hash of the exact prompt text
and model. Change one block and only that block's run costs money; run the same
thing twice and the second run is free.

## Commands

### `run` — score a prompt

```bash
promptxray run --prompt prompt.txt --data data.csv \
  --provider openai --model gpt-4o-mini --report report.html
```

### `ablate` — measure what every block does

```bash
promptxray ablate --prompt prompt.txt --data data.csv \
  --provider openai --model gpt-4o-mini \
  --subset-size 60 --report report.html
```

### `diff` — compare two prompt versions, example by example

Aggregate metrics hide the trade. This shows you exactly what you gained and
what you quietly broke.

Delete the block `ablate` blamed, and check the result:

```bash
promptxray diff --before examples/prompt.txt --after examples/prompt_v2.txt \
  --data examples/data.csv --provider mock --fail-under 0.80
```

```
prompt.txt:    macro F1 0.836
prompt_v2.txt: macro F1 0.932 (+0.096)

fixed  4
  + [spam] Your ad says "click here for a free prize" - please stop mailing me...
  + [spam] I got another one: "buy now, winner selected today" from your domain.
  + [spam] Message reads "FREE upgrade, click here" and it came from your server.
  + [spam] Third copy of "claim your prize now" offer this week.
broken 0
```

`--fail-under` exits with code 1, so this works as a CI gate on prompt changes.

## A harder example

`examples/case-study/` holds a real one: 100 labelled messages from a game
chat, three classes, and a fourteen-block moderation prompt written the way
prompts actually get written. Two of its rules are overbroad on purpose. Run
the ablation and see whether the tool finds them before you do.

See [examples/case-study/RUNBOOK.md](examples/case-study/RUNBOOK.md).

## Writing a prompt file

Plain text. Blocks are separated by blank lines.

```
You are a message triage assistant.

Classify the message into exactly one of: spam, urgent, normal.

Treat ALL-CAPS messages as urgent.

Answer with the single label and nothing else. [[keep]]

Text: {input}
```


- `{input}` is where each row of your dataset goes. Required.
- `[[keep]]` pins a block so it is never removed. Use it on the output-format
  instruction: deleting that one does not test an idea, it just breaks parsing
  and produces a meaningless delta.
- Blocks containing `{input}` are pinned automatically.

## Dataset format

CSV or JSONL with a text column and a label column:

```csv
text,label
"Buy now and get 70% off",spam
"When is my next invoice?",normal
```

Different column names: `--text-column message --label-column category`.

## Providers

You never have to pay to use this tool. The first three cost nothing:

| `--provider` | What it is | Cost | Credentials |
| --- | --- | --- | --- |
| `mock` | Built-in fake classifier for the demo and CI | free | none |
| `ollama` | Local models on your own machine | free | none |
| `lmstudio` | Same, via LM Studio | free | none |
| `openrouter` | Hosted; has free models (`:free` suffix) | free tier | `OPENROUTER_API_KEY` |
| `groq` | Hosted, fast | free tier | `GROQ_API_KEY` |
| `openai` | OpenAI and compatible servers | paid | `OPENAI_API_KEY` |
| `anthropic` | Claude | paid | `ANTHROPIC_API_KEY` |

The free path, start to finish — install [Ollama](https://ollama.com), then:

```bash
ollama pull llama3.2

promptxray ablate --prompt prompt.txt --data data.csv \
  --provider ollama --model llama3.2
```

Nothing leaves your machine and nothing is billed. Any OpenAI-compatible server
works via `--base-url`.

## What it costs

Roughly `dataset_size + (unpinned_blocks × subset_size)` calls on the first
run. A 200-row dataset with 10 unpinned blocks and `--subset-size 60` is 800
calls — a few cents on a small model, and the cache makes every repeat free.

## What this is not

Not an observability platform, not a tracing backend, not a judge framework.
No server, no account, no dashboard. It is a library and a CLI that answer one
question, and the report is a single HTML file you can commit, mail, or attach
to a pull request.

For tracing and production monitoring, use Langfuse or Phoenix. For adversarial
red-teaming, use promptfoo. promptxray fits in between: you have a prompt, you
have labelled data, and you want to know which line is pulling its weight.

## Development

```bash
git clone https://github.com/godwire/promptxray
cd promptxray
pip install -e ".[dev]"
pytest
```

The test suite runs entirely on the mock provider, so it needs no network and
no API key.

## License

MIT