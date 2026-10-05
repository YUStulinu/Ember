"""
Produces the reference ("golden") data Ember's tests compare against, using
Hugging Face transformers and tokenizers in float32 on the CPU.

    reference/.venv/Scripts/python reference/golden.py            (Windows)
    reference/.venv/bin/python reference/golden.py                (Linux/macOS)

Output, under tests/golden/:
  tokenizer.json            texts and the token ids the official tokenizer gives them
  chat_template.json        conversations and the exact prompt text they render to
  <model>/meta.json         prompt ids, greedy continuation, top logits
  <model>/hidden.f32        hidden state of the last prompt token after the
                            embedding and after every layer ([n_layers + 1, hidden])
  <model>/logits.f32        logits of the last prompt token ([vocab])
"""
import json
import os
import sys

import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODELS = os.path.join(ROOT, "models")
OUT = os.path.join(ROOT, "tests", "golden")

TOKENIZER_TEXTS = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "Bună ziua! Ce mai faci? Mâine mergem la Iași și la Timișoara.",
    "ȘTEFAN CEL MARE ȘI SFÂNT, ȚARA ROMÂNEASCĂ",
    "și ț (combining comma below must normalize to ș ț)",
    "é à ô ñ ü",          # decomposed accents -> NFC
    "Å Ω ﬁ",                             # singleton decompositions, ligature (kept)
    "I'm sure you've seen it; they'll say we'd won. IT'S DONE, ISN'T IT? 'quoted'",
    "def f(x):\n    return x ** 2  # square\n\n\n\tindented\r\nwindows line",
    "    leading spaces and trailing spaces    ",
    "multiple   spaces nbsp　ideographic thin",
    "numbers 1234567 3.14159 -42 1e10 ١٢٣ ⅷ",
    "emoji 😀🎉👍🏽 family 👨‍👩‍👧 flags 🇷🇴",
    "中文分词测试，日本語のテキスト、한국어 텍스트",
    "Привет, мир! Ελληνικά κείμενα. עברית. العربية",
    "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    "special tokens <|endoftext|> mid<|im_end|>text and <tool_call>{\"a\":1}</tool_call>",
    "<|im_start without end <| |> < | im_end | >",
    "",
    " ",
    "\n\n\n",
    "a" * 300,
    "https://example.com/path?query=1&x=y#frag user@mail.ro",
    "JSON: {\"key\": [1, 2, {\"nested\": null}]}",
    "tabs\t\tand\u000bvertical\u000cform feed",
    "mixed 123abc abc123 a1b2c3 ..., !!! ??? ---",
    "\U0001F600" * 5 + "x",
    "Ünïcödé ñ ß ﬀ ǅ ǆ İstanbul ı",
    "\x00\x01 control \x7f chars",
    "x" + "́" * 3 + " stacked marks",
    "café naïve résumé coöperate",
    "ά έ ή ί ό ύ ώ (Greek tonos)",
]

CONVERSATIONS = [
    {"messages": [{"role": "user", "content": "Salut! Cum te cheamă?"}], "enable_thinking": False},
    {"messages": [{"role": "user", "content": "What is 2+2?"}], "enable_thinking": True},
    {"messages": [{"role": "system", "content": "You are a concise assistant."},
                  {"role": "user", "content": "Name three colors."}], "enable_thinking": False},
    {"messages": [{"role": "system", "content": "Be brief."},
                  {"role": "user", "content": "Hi"},
                  {"role": "assistant", "content": "<think>\nThe user greets.\n</think>\n\nHello! How can I help?"},
                  {"role": "user", "content": "Tell me a joke."}], "enable_thinking": False},
    {"messages": [{"role": "user", "content": "A"},
                  {"role": "assistant", "content": "B"},
                  {"role": "user", "content": "C"},
                  {"role": "assistant", "content": "D"},
                  {"role": "user", "content": "E"}], "enable_thinking": True},
]

MODEL_PROMPT = [{"role": "user", "content": "Explain in one paragraph why the sky is blue."}]
GREEDY_TOKENS = 48


def tokenizer_golden(model_dir):
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    cases = []
    for text in TOKENIZER_TEXTS:
        ids = tok.encode(text, add_special_tokens=False).ids
        cases.append({"text": text, "ids": ids, "decoded": tok.decode(ids, skip_special_tokens=False)})
    with open(os.path.join(OUT, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, ensure_ascii=False, indent=1)
    print(f"tokenizer: {len(cases)} cases")


def chat_golden(model_dir):
    tok = AutoTokenizer.from_pretrained(model_dir)
    cases = []
    for conv in CONVERSATIONS:
        text = tok.apply_chat_template(conv["messages"], tokenize=False, add_generation_prompt=True,
                                       enable_thinking=conv["enable_thinking"])
        cases.append({**conv, "prompt": text})
    with open(os.path.join(OUT, "chat_template.json"), "w", encoding="utf-8") as f:
        json.dump({"cases": cases}, f, ensure_ascii=False, indent=1)
    print(f"chat template: {len(cases)} cases")


@torch.no_grad()
def model_golden(name):
    model_dir = os.path.join(MODELS, name)
    out_dir = os.path.join(OUT, name)
    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
    model.eval()

    prompt = tok.apply_chat_template(MODEL_PROMPT, tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids])
    out = model(x, output_hidden_states=True)
    # hidden_states: embedding output, then each layer's output (the last one after the final norm in HF,
    # so recompute the un-normed last layer output separately).
    hs = [h[0, -1].float().numpy() for h in out.hidden_states]
    logits = out.logits[0, -1].float().numpy()

    # HF applies the final norm to the last entry of hidden_states; store the raw residual stream instead.
    captured = {}

    def hook(_m, _inp, output):
        captured["h"] = (output[0] if isinstance(output, tuple) else output)[0, -1].float().numpy()

    handle = model.model.layers[-1].register_forward_hook(hook)
    model(x)
    handle.remove()
    hs[-1] = captured["h"]
    np.stack(hs).astype(np.float32).tofile(os.path.join(out_dir, "hidden.f32"))
    logits.astype(np.float32).tofile(os.path.join(out_dir, "logits.f32"))

    gen = model.generate(x, max_new_tokens=GREEDY_TOKENS, do_sample=False, temperature=None, top_p=None,
                         top_k=None)
    cont = gen[0, len(ids):].tolist()
    top = np.argsort(-logits)[:20]
    meta = {
        "model": name,
        "prompt": prompt,
        "prompt_ids": ids,
        "greedy_ids": cont,
        "greedy_text": tok.decode(cont, skip_special_tokens=False),
        "top_ids": top.tolist(),
        "top_logits": logits[top].tolist(),
        "hidden_rows": len(hs),
        "hidden_size": int(hs[0].shape[0]),
        "vocab_size": int(logits.shape[0]),
        "max_abs_hidden": float(np.abs(np.stack(hs)).max()),
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=1)
    print(f"{name}: {len(ids)} prompt tokens, greedy: {meta['greedy_text']!r}")
    print(f"  max |hidden| = {meta['max_abs_hidden']:.1f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(os.cpu_count() or 4)
    tokenizer_golden(os.path.join(MODELS, "Qwen3-0.6B"))
    chat_golden(os.path.join(MODELS, "Qwen3-0.6B"))
    names = sys.argv[1:] or ["Qwen3-0.6B", "Qwen3-1.7B"]
    for name in names:
        model_golden(name)


if __name__ == "__main__":
    main()
