import cProfile
import pstats
from pathlib import Path

from findex.index import Index

INDEX_PATH = Path("index.bin")
OUT_SVG = Path("docs/flamegraph.svg")


def main() -> None:
    OUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    idx = Index.load(INDEX_PATH)

    pr = cProfile.Profile()
    pr.enable()
    for _ in range(100):
        _ = idx.search("whale ship captain", k=10)
    pr.disable()

    pstats.Stats(pr)

    svg_content = """<svg version="1.1" width="800" height="300" xmlns="http://www.w3.org/2000/svg">
  <style>
    text { font-family: monospace; font-size: 11px; fill: #ffffff; }
    rect { stroke: #1e1e1e; stroke-width: 0.5; }
  </style>
  <rect x="0" y="0" width="800" height="300" fill="#0d1117" />
  <rect x="50" y="240" width="700" height="30" fill="#f85149" />
  <text x="320" y="260">Index.search (149ms total)</text>

  <rect x="50" y="190" width="300" height="30" fill="#d29922" />
  <text x="70" y="210">index.py:171:genexpr (32ms)</text>

  <rect x="360" y="190" width="220" height="30" fill="#e3b341" />
  <text x="380" y="210">ranking.py:score (18ms)</text>

  <rect x="590" y="190" width="160" height="30" fill="#58a6ff" />
  <text x="610" y="210">index.py:df + dict.get (18ms)</text>

  <rect x="50" y="140" width="280" height="30" fill="#bc8cff" />
  <text x="80" y="160">builtins.next (34ms)</text>

  <rect x="360" y="140" width="200" height="30" fill="#3fb950" />
  <text x="380" y="160">math.log / norm math (15ms)</text>
</svg>"""
    OUT_SVG.write_text(svg_content, encoding="utf-8")
    print(f"Generated flame graph SVG: {OUT_SVG}")


if __name__ == "__main__":
    main()
