"""Unified instruction-format examples (McDaniel 2023) built from tp-dataset/corpus.

Every task, monolingual or bilingual, is a chat turn so small models stay anchored to
the request instead of free-running toki pona.
"""

import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TP_DATASET = os.path.join(HERE, "..", "tp-dataset")
CORPUS = os.path.join(TP_DATASET, "corpus")
sys.path.insert(0, TP_DATASET)

from tp_tokenizer import TokiPonaTokenizer  # noqa: E402

SYSTEM = "You are ilo ante toki, a careful translator between toki pona, English, and Chinese."
LANG = {"tok": "toki pona", "en": "English", "zh": "Chinese"}

# the first phrasing of each task is the canonical one used at evaluation time
TRANSLATE = [
    "Translate this {src} text into {tgt}.\n\n{text}",
    "{text}\n\nWhat is this in {tgt}?",
    "Please render the following {src} in {tgt}:\n{text}",
]
CONTINUE = [
    "Continue the following toki pona passage.\n\n{text}",
    "Write the next part of this toki pona text:\n{text}",
]
DEFINE = {
    "en": ['What does the toki pona word "{word}" mean?', 'Define the toki pona word "{word}".'],
    "zh": ['道本语（toki pona）单词"{word}"是什么意思？'],
}
REVERSE_DEFINE = 'How do you say "{gloss}" in toki pona, using a single word?'


def translate_prompt(src, tgt, variant=0):
    """Instruction with a literal {text} slot for the payload."""
    return TRANSLATE[variant].format(src=LANG[src], tgt=LANG[tgt], text="{text}")


def load_jsonl(name):
    with open(os.path.join(CORPUS, name)) as f:
        return [json.loads(line) for line in f]


def example(prompt, text, assistant, task, tp_user=False, tp_assistant=False):
    """`prompt` has a {text} slot; `text` is filled in (and marked if toki pona) at encode time."""
    return {"prompt": prompt, "text": text, "assistant": assistant, "task": task,
            "tp_user": tp_user, "tp_assistant": tp_assistant}


def build_examples(split="train", seed=0, max_zh_silver=40000, max_mono=40000, dict_repeat=4):
    """Returns a list of chat examples; toki pona spans are marked later by the tokenizer."""
    rng = random.Random(seed)
    parallel = [r for r in load_jsonl("parallel.jsonl") if r["split"] == split]
    out = []

    def variant():
        return 0 if rng.random() < 0.7 else rng.randrange(1, len(TRANSLATE))

    zh_silver = 0
    for r in parallel:
        if r["en"]:
            out.append(example(translate_prompt("tok", "en", variant()), r["tok"], r["en"], "tok-en", tp_user=True))
            out.append(example(translate_prompt("en", "tok", variant()), r["en"], r["tok"], "en-tok", tp_assistant=True))
        if r["zh"]:
            if r["zh_quality"] != "gold":
                zh_silver += 1
                if zh_silver > max_zh_silver:
                    continue
            out.append(example(translate_prompt("tok", "zh", variant()), r["tok"], r["zh"], "tok-zh", tp_user=True))
            out.append(example(translate_prompt("zh", "tok", variant()), r["zh"], r["tok"], "zh-tok", tp_assistant=True))

    if split == "train":
        mono = [m for m in load_jsonl("mono.jsonl") if m["n_words"] >= 12 and m["source"] != "tatoeba"]
        rng.shuffle(mono)
        for m in mono[:max_mono]:
            words = m["text"].split()
            cut = rng.randint(len(words) // 3, 2 * len(words) // 3)
            first, rest = " ".join(words[:cut]), " ".join(words[cut:])
            out.append(example(rng.choice(CONTINUE), first, rest, "continue",
                               tp_user=True, tp_assistant=True))

        for d in load_jsonl("dictionary.jsonl"):
            for _ in range(dict_repeat):
                if d["en"]:
                    out.append(example(rng.choice(DEFINE["en"]).replace("{word}", "{text}"), d["word"], d["en"], "define-en"))
                if d["zh"]:
                    out.append(example(DEFINE["zh"][0].replace("{word}", "{text}"), d["word"], d["zh"], "define-zh"))
            for gloss, weight in d["ku_en"].items():
                if weight >= 60 and d["usage"] in ("core", "common"):
                    out.append(example(REVERSE_DEFINE.replace("{gloss}", "{text}"), gloss, d["word"], "reverse-define",
                                       tp_assistant=True))

    rng.shuffle(out)
    return out


class ChatEncoder:
    """Turns examples into (input_ids, labels) with loss only on the assistant turn."""

    def __init__(self, base_tokenizer, max_len=384):
        self.tp = TokiPonaTokenizer(base_tokenizer)
        self.tok = base_tokenizer
        self.max_len = max_len
        self.has_system = "system" in (base_tokenizer.chat_template or "")
        vocab = base_tokenizer.get_vocab()
        self.stop_ids = {base_tokenizer.eos_token_id} | {vocab[t] for t in ("<|im_end|>", "<turn|>", "<end_of_turn>")
                                                         if t in vocab}

    def messages(self, user, assistant=None):
        msgs = [{"role": "system", "content": SYSTEM}] if self.has_system else []
        msgs.append({"role": "user", "content": user})
        if assistant is not None:
            msgs.append({"role": "assistant", "content": assistant})
        return msgs

    def _ids(self, msgs, gen_prompt):
        out = self.tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=gen_prompt)
        return list(out["input_ids"]) if isinstance(out, dict) or hasattr(out, "keys") else list(out)

    def user_text(self, ex):
        text = self.tp.mark(ex["text"]) if ex["tp_user"] else ex["text"]
        return ex["prompt"].replace("{text}", text)

    def prompt_ids(self, user_marked):
        return self._ids(self.messages(user_marked), gen_prompt=True)

    def encode(self, ex):
        user = self.user_text(ex)
        assistant = self.tp.mark(ex["assistant"]) if ex["tp_assistant"] else ex["assistant"]
        prompt = self.prompt_ids(user)
        full = self._ids(self.messages(user, assistant), gen_prompt=False)
        if full[:len(prompt)] != prompt:
            raise ValueError("chat template is not prefix-stable; cannot mask the prompt")
        # drop the trailing newline some templates add after the end-of-turn token
        while len(full) > len(prompt) + 1 and self.tok.decode(full[-1:]).strip() == "":
            full = full[:-1]
        if len(full) > self.max_len:
            return None
        labels = [-100] * len(prompt) + full[len(prompt):]
        return {"input_ids": full, "labels": labels}
