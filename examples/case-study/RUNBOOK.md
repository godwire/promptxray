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
- A delta smaller than about 0.02 on a 50-example subset is noise. Raise
  `--subset-size` before you trust a small number.
- Run the same ablation twice with `--seed 1` and compare. Verdicts that flip
  between seeds are not verdicts.
- These 100 examples are labelled by one person. Treat the result as a signal
  about your prompt, not as a benchmark of the model.
