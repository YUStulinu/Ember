"""
Builds the perplexity evaluation texts in eval/ (not committed):
  eval/en.txt  English prose: the Python tutorial (from the local Python docs)
  eval/ro.txt  Romanian prose: the held-out split of the Kindling corpus, if present

    python reference/make_eval_text.py
"""
import glob
import html
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUT = os.path.join(ROOT, "eval")
LIMIT = 80_000  # characters per file


def python_tutorial():
    base = os.path.join(os.path.dirname(sys.executable), "Doc", "html", "tutorial")
    pages = sorted(glob.glob(os.path.join(base, "*.html")))
    text = []
    for page in pages:
        raw = open(page, encoding="utf-8", errors="replace").read()
        for para in re.findall(r"<p>(.*?)</p>", raw, flags=re.S):
            para = " ".join(html.unescape(re.sub(r"<[^>]+>", "", para)).split())
            if len(para) > 120:
                text.append(para)
    return "\n\n".join(text)


def romanian():
    path = os.path.join(ROOT, "..", "Kindling", "data", "test.txt")
    if not os.path.exists(path):
        return ""
    return open(path, encoding="utf-8").read()


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, text in (("en.txt", python_tutorial()), ("ro.txt", romanian())):
        if not text:
            print(f"skipped {name}: source not found")
            continue
        text = text[:LIMIT]
        with open(os.path.join(OUT, name), "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        print(f"wrote eval/{name}: {len(text)} characters")


if __name__ == "__main__":
    main()
