import io
import urllib.request
import zipfile
from pathlib import Path

OUTPUT_DIR = Path("data/corpus")
GUTENBERG_ZIP = "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/gutenberg.zip"


def main() -> None:
    if OUTPUT_DIR.exists() and len(list(OUTPUT_DIR.glob("*.txt"))) >= 500:
        print(
            f"Corpus already present in {OUTPUT_DIR} ({len(list(OUTPUT_DIR.glob('*.txt')))} files)."
        )
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("Fetching raw texts...")

    req = urllib.request.Request(GUTENBERG_ZIP, headers={"User-Agent": "Mozilla/5.0"})
    with (
        urllib.request.urlopen(req) as resp,
        zipfile.ZipFile(io.BytesIO(resp.read())) as z,
    ):
        raw_files = [m for m in z.namelist() if m.endswith(".txt")]
        doc_count = 0

        for member in raw_files:
            book_name = Path(member).stem
            content = z.read(member).decode("utf-8", errors="replace")

            words = content.split()
            chunk_size = 2500
            for i in range(0, len(words), chunk_size):
                chunk_text = " ".join(words[i : i + chunk_size])
                doc_id = f"{book_name}_part_{i // chunk_size:04d}"
                (OUTPUT_DIR / f"{doc_id}.txt").write_text(chunk_text, encoding="utf-8")
                doc_count += 1

    total_mb = sum(f.stat().st_size for f in OUTPUT_DIR.glob("*.txt")) / (1024 * 1024)
    print(f"Done: {doc_count} documents generated ({total_mb:.2f} MB) in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
