import re, glob, sys, os

def norm(t):
    t = t.lower()
    t = re.sub(r'[^a-z0-9 ]', ' ', t)
    return t.split()

def grams(ws, n):
    return {(" ".join(ws[i:i+n])): i for i in range(len(ws)-n+1)}

paper_raw = open('paper_plaintext.txt').read()
pw = norm(paper_raw)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6

hits = {}
for f in sorted(glob.glob('sources/*.txt')):
    sw = norm(open(f).read())
    sg = set(grams(sw, N))
    for g, idx in grams(pw, N).items():
        if g in sg:
            hits.setdefault(g, []).append(os.path.basename(f)[:-4])

# merge overlapping n-grams into maximal runs
idxmap = grams(pw, N)
matched_positions = set()
for g in hits:
    i = idxmap[g]
    matched_positions.update(range(i, i+N))

runs, cur = [], []
for i in sorted(matched_positions):
    if cur and i == cur[-1]+1: cur.append(i)
    else:
        if cur: runs.append(cur)
        cur=[i]
if cur: runs.append(cur)

print(f"=== {N}-gram overlap vs {len(glob.glob('sources/*.txt'))} cited sources ===")
print(f"paper words: {len(pw)}   matched words: {len(matched_positions)}  "
      f"({100*len(matched_positions)/len(pw):.2f}%)\n")
for r in sorted(runs, key=len, reverse=True):
    phrase = " ".join(pw[r[0]:r[-1]+1])
    src = set()
    for g,s in hits.items():
        if r[0] <= idxmap[g] <= r[-1]: src.update(s)
    print(f"[{len(r):2d}w] {phrase}   <- {','.join(sorted(src))}")
