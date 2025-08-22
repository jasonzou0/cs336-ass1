
Here’s a quick, practical toolkit for profiling Python code—CPU time, hot lines, and memory—plus copy-paste recipes.

# 1) Built-in CPU profiler (deterministic): `cProfile`

Great first pass to find hot *functions*.

**CLI (entire script)**

```bash
python -m cProfile -o out.prof your_script.py --arg1 foo
```

**Inspect results (text)**

```bash
python -m pstats out.prof
# then in the pstats shell:
sort cumtime
stats 30
```

**Pretty UI**

```bash
pip install snakeviz
snakeviz out.prof
```

**In-code (profile just a block)**

```python
import cProfile, pstats, io, time

def main():
    ...

pr = cProfile.Profile()
pr.enable()
main()
pr.disable()

s = io.StringIO()
pstats.Stats(pr).sort_stats("cumtime").print_stats(30, stream=s)
print(s.getvalue())
```

> Read the columns: **tottime** = time spent *in* the function (excluding callees), **cumtime** = time in function + all callees. Optimize by **cumtime** hotspots first.

# 2) Line-level CPU hotspots: `line_profiler`

Shows which *lines* are slow (pure Python only).

```bash
pip install line_profiler
```

Annotate and run:

```python
# file: app.py
@profile
def encode_many(texts):
    ...
```

```bash
kernprof -l -v app.py
```

# 3) Low-overhead sampling + flamegraphs: `py-spy`

Good for real workloads or production (doesn’t require code changes).

```bash
pip install py-spy
# Live top-like view of hottest lines
py-spy top -- python your_script.py
# Record a flamegraph
py-spy record -o profile.svg -- python your_script.py
```

Open `profile.svg` in a browser; the widest stacks are your bottlenecks.

# 4) CPU + Memory (per line) in one: `scalene`

Excellent for mixed CPU/memory diagnosis.

```bash
pip install scalene
scalene your_script.py
# or focus on a file:
scalene --profile-only app.py your_script.py
```

# 5) Memory leaks / allocations

**Built-in `tracemalloc`:**

```python
import tracemalloc
tracemalloc.start()
... run workload ...
current, peak = tracemalloc.get_traced_memory()
print(f"current={current/1e6:.1f}MB peak={peak/1e6:.1f}MB")
```

**Per-line memory:** `memory_profiler`

```bash
pip install memory_profiler
python -m memory_profiler app.py
```

or annotate with `@profile` (like line\_profiler).

# 6) Microbenchmarks: `timeit`

For tight loops / alternative implementations:

```python
python -m timeit -s "from yourmod import fn; data=list(range(10**5))" "fn(data)"
```

# 7) Tips for reliable results

* **Use a representative input** (same size & characteristics as production).
* Run multiple times; ignore the first for warm-up and caching.
* Prefer **wall time** (`time.perf_counter()`) when I/O is involved; CPU time for compute.
* cProfile won’t break into C/NumPy internals—use **py-spy** or **scalene** to see those costs.
* For async/multiprocessing, profile the **top-level runner** (e.g., around `asyncio.run(...)`) or use **py-spy** against the process.

---

### Quick recipe for your tokenizer

```python
# profile_tokenizer.py
import cProfile, pstats, io
from my_tokenizer import get_tokenizer

def run():
    tok = get_tokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    with open("tinystories_sample_5M.txt") as f:
        for _ in range(3):  # small subset to iterate quickly
            for _id in tok.encode_iterable(f):
                pass

if __name__ == "__main__":
    pr = cProfile.Profile()
    pr.enable()
    run()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr).sort_stats("cumtime").print_stats(40, stream=s)
    print(s.getvalue())
```

If you want, share your current hotpath; I can suggest targeted micro-optimizations (e.g., regex iteration, reducing temporary `bytes`, faster merge loops).
