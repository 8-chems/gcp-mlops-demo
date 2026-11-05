"""Download the UCI Bank Marketing dataset to ./data/bank-additional-full.csv."""
import io
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"
OUT = Path("data")


def main():
    OUT.mkdir(exist_ok=True)
    print(f"Downloading {URL} ...")
    outer = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL, timeout=120).read()))
    inner_name = next(n for n in outer.namelist() if n.endswith("bank-additional.zip"))
    inner = zipfile.ZipFile(io.BytesIO(outer.read(inner_name)))
    member = next(n for n in inner.namelist()
                  if n.endswith("bank-additional-full.csv") and "__MACOSX" not in n)
    (OUT / "bank-additional-full.csv").write_bytes(inner.read(member))
    print("Saved data/bank-additional-full.csv")


if __name__ == "__main__":
    main()
