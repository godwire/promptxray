# Case study: moderating game chat

A worked example on a harder problem than the toy one in `examples/`.

`chat-moderation.csv` holds 100 labelled messages from a competitive game chat,
split into three classes:

| label | meaning | count |
| --- | --- | --- |
| `toxic` | attacks a person: insults, telling someone to quit, threats | 30 |
| `heated` | angry or blaming, but about how someone is *playing* | 30 |
| `fine` | ordinary chat, including harsh complaints about the game itself | 40 |

The line between `toxic` and `heated` is the whole difficulty, and the dataset
is built to punish lazy rules on purpose:

- messages in caps that are perfectly friendly (`GG WELL PLAYED EVERYONE`)
- insult words aimed at the game, not a person (`this map is garbage`)
- politely worded abuse (`kindly uninstall, you are ruining it for everyone`)
- self-criticism that reads like an insult (`my aim is trash tonight`)

`prompt-v1.txt` is written the way prompts actually get written: a role, three
definitions, a couple of rules someone added after a bad week, two pieces of
encouragement aimed at nobody, and two few-shot examples. Fourteen blocks, two
of them pinned.

Two of those rules are deliberately overbroad. Don't look for them in the file —
the point of the exercise is to let the tool find them.

## Run it without installing anything

Fork the repository on GitHub, open the **Actions** tab, pick **real model** in
the list on the left, and press **Run workflow**. GitHub installs Ollama on one
of its own machines, downloads an open model, runs the ablation below, and
attaches `case-study-report.html` to the run under **Artifacts**. Free for public
repositories. A CPU runner is slow, so expect it to take a while.

## Run it — free, on your own machine

Install [Ollama](https://ollama.com) and pull a small model. `llama3.2` is
about 2 GB and runs on a laptop CPU.

```bash
ollama pull llama3.2

promptxray ablate \
  --prompt examples/case-study/prompt-v1.txt \
  --data examples/case-study/chat-moderation.csv \
  --provider ollama --model llama3.2 \
  --subset-size 50 \
  --report case-v1.html
```

That is roughly 100 + 12 × 50 = 700 calls. Nothing is billed, nothing leaves
your machine, and every repeat is free anyway because answers are cached.

On a CPU-only laptop expect this to take a while. Two ways to shorten it:
`--subset-size 30` on the first pass, and `--workers 2` if the machine starts
to struggle. Once the cache is warm, re-running is instant.

If a small local model returns too many unparsable answers, try `qwen2.5:3b` or
`llama3.1:8b` before concluding anything about your prompt.

Hosted free tiers work the same way — `--provider openrouter` with a model
ending in `:free`, or `--provider groq`. Both need a key in
`OPENROUTER_API_KEY` / `GROQ_API_KEY`, neither needs a card.

## What to look at

Open `case-v1.html` and read the coloured prompt top to bottom.

1. **Amber blocks.** These are rules that cost you score. Hover one: the tooltip
   names the examples that become correct once the rule is gone.
2. **Dotted blocks.** Instructions the model ignores completely. Encouragement
   and "be thorough" phrasing usually lands here.
3. **The confusion matrix.** Check which direction the errors run. A moderation
   prompt that pushes `heated` into `toxic` will over-punish players; the
   opposite lets abuse through. The aggregate F1 hides which mistake you are
   making.
4. **Unparsable answers.** Small local models often reply with a sentence
   instead of a label. A high count here is a formatting problem, not a
   reasoning problem.

## Let the tool write the shorter prompt

```bash
promptxray suggest \
  --prompt examples/case-study/prompt-v1.txt \
  --data examples/case-study/chat-moderation.csv \
  --provider ollama --model llama3.2 \
  --holdout 0.3 --subset-size 40 --out prompt-v2.txt
```

Blocks are judged on 70% of the dataset and the result is scored on the other
30%, which the selection never saw. If the shorter prompt does not hold up on
that holdout, nothing is written.

The search removes one block at a time and re-measures after each removal. The
case-study prompt has two few-shot examples and three class definitions that
partly repeat each other; if the model only needs one of a pair, the output
names the pair instead of silently cutting both.

It prints the worst-case number of calls before it starts. With twelve
unpinned blocks that number is large on a laptop CPU. Two ways to shorten it:
`--strategy one-shot` (a single round, about the cost of `ablate`), or
`--max-steps 4` to stop after four removals.

## Then fix it and prove the fix

Copy the prompt, delete or narrow the blocks the report blamed, and compare:

```bash
cp examples/case-study/prompt-v1.txt prompt-v2.txt
# edit prompt-v2.txt

promptxray diff \
  --before examples/case-study/prompt-v1.txt \
  --after prompt-v2.txt \
  --data examples/case-study/chat-moderation.csv \
  --provider ollama --model llama3.2
```

`diff` lists the examples that were fixed and the ones that broke. A change
that gains four and breaks five is not an improvement, and the aggregate metric
would have hidden that.

## Reading the numbers honestly

- Run with temperature 0, which is the default here.
- Read the confidence interval, not the point estimate. A block marked "too
  noisy to call" has an interval crossing zero: the run genuinely cannot say
  which way it goes, and raising `--subset-size` is the fix.
- Expect several noisy verdicts on 50 examples with a small local model. That
  is the dataset talking, not a bug.
- Run the same ablation with `--seed 1` and compare. Confident verdicts should
  survive; if they flip, treat the whole report as underpowered.
- These 100 examples are labelled by one person. Treat the result as a signal
  about your prompt, not as a benchmark of the model.