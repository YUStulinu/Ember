"""
Ember vs llama.cpp on the same GPU, same model (Qwen3-1.7B), comparable formats.

    python bench/compare.py            (after building Ember and fetching bench/external, see BENCHMARKS.md)

The two engines are run alternately, format by format, with a cool-down pause
before each run: a laptop GPU slows down as it heats, and alternating keeps
both engines under the same conditions. Writes bench/results/compare.md.
"""
import os
import re
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXT = os.path.join(ROOT, "bench", "external")
EMBER = os.path.join(ROOT, "build", "Release", "ember.exe") if os.name == "nt" else os.path.join(ROOT, "build", "ember")
LLAMA_DIR = os.path.join(EXT, "llama.cpp")
COOL_DOWN = 25  # seconds

PAIRS = [  # (Ember format, llama.cpp GGUF)
    ("f16", "Qwen3-1.7B-F16.gguf"),
    ("int8", "Qwen3-1.7B-Q8_0.gguf"),
    ("int4", "Qwen3-1.7B-Q4_K_M.gguf"),
]


def gpu_temp():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


def cool():
    time.sleep(COOL_DOWN)
    t = gpu_temp()
    while t is not None and t > 60:
        time.sleep(5)
        t = gpu_temp()


def run(cmd, env=None):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    return r.stdout + r.stderr


def ember(fmt):
    out = run([EMBER, "bench", "-q", fmt, "-m", os.path.join(ROOT, "models", "Qwen3-1.7B"), "prompt=512", "batch=1,8,32"])
    res = {"prefill": float(re.search(r"prefill, 512-token prompt\s+([\d.]+)", out).group(1))}
    for b, g in re.findall(r"^\s+(\d+) sequences?\s+x \d+ tokens\s+([\d.]+) t/s", out, re.M):
        res[f"tg{b}"] = float(g)
    return res


def llama(gguf):
    env = dict(os.environ)
    cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4\bin\x64"
    env["PATH"] = cuda_bin + os.pathsep + env.get("PATH", "")
    exe = lambda name: os.path.join(LLAMA_DIR, name + (".exe" if os.name == "nt" else ""))
    model = os.path.join(EXT, "gguf", gguf)
    res = {}
    out = run([exe("llama-bench"), "-m", model, "-p", "512", "-n", "0", "-ngl", "99", "-fa", "1", "-r", "3"], env)
    m = re.search(r"pp512 \|\s+([\d.]+)", out)
    res["prefill"] = float(m.group(1)) if m else float("nan")
    out = run([exe("llama-batched-bench"), "-m", model, "-c", "16384", "-b", "2048", "-ub", "512", "-npp", "64", "-ntg", "128",
               "-npl", "1,8,32", "-ngl", "99", "-fa", "on"], env)
    for row in re.findall(r"^\|\s+64 \|\s+128 \|\s+(\d+) \|.*$", out, re.M):
        pass
    for line in out.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 10 and cells[0] == "64" and cells[1] == "128":
            res[f"tg{cells[2]}"] = float(cells[7])  # S_TG t/s
    return res


def main():
    if not os.path.exists(EMBER) or not os.path.isdir(LLAMA_DIR):
        sys.exit("build Ember and fetch bench/external first (see docs/BENCHMARKS.md)")
    rows = []
    for fmt, gguf in PAIRS:
        print(f"== {fmt} vs {gguf}", flush=True)
        cool()
        e = ember(fmt)
        print("   ember", e, flush=True)
        cool()
        l = llama(gguf)
        print("   llama.cpp", l, flush=True)
        rows.append((fmt, gguf, e, l))

    def cell(e, l, key):
        a, b = e.get(key, float("nan")), l.get(key, float("nan"))
        return f"{a:,.0f} | {b:,.0f} | **{a / b:.2f}x**" if b == b and b > 0 else f"{a:,.0f} | - | -"

    lines = ["| Format (Ember / llama.cpp) | Metric | Ember | llama.cpp | Ratio |", "|---|---|---:|---:|---:|"]
    for fmt, gguf, e, l in rows:
        name = f"{fmt} / {gguf.replace('Qwen3-1.7B-', '').replace('.gguf', '')}"
        for key, label in (("prefill", "prefill, 512 tokens (t/s)"), ("tg1", "generation, 1 sequence (t/s)"),
                           ("tg8", "generation, 8 sequences (t/s)"), ("tg32", "generation, 32 sequences (t/s)")):
            lines.append(f"| {name} | {label} | {cell(e, l, key)} |")
    os.makedirs(os.path.join(ROOT, "bench", "results"), exist_ok=True)
    out = os.path.join(ROOT, "bench", "results", "compare.md")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
