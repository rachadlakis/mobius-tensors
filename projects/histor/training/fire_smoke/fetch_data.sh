#!/usr/bin/env bash
# Download the three datasets into data/raw/ on a Linux machine (e.g. a RunPod pod), check them,
# and unpack each into its own folder. Faster than uploading from a laptop.
#   bash fetch_data.sh && python prepare_data.py
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data/downloads data/raw

get() {  # url file md5 (empty md5: download once, no check)
  local f="data/downloads/$2"
  if [ -n "$3" ] && [ -f "$f" ] && echo "$3  $f" | md5sum -c --quiet 2>/dev/null; then return; fi
  if [ -z "$3" ] && [ -f "$f" ]; then return; fi
  curl -fL --retry 10 -o "$f.part" "$1" && mv "$f.part" "$f"
  [ -z "$3" ] || echo "$3  $f" | md5sum -c
}

# CQU fire/smoke, CC BY 4.0 (Figshare 10.25946/28747046)
get https://ndownloader.figshare.com/files/53486522 cqu.zip fd9ed42cd67fd5f7c016ca127985e245
# Indoor Fire Smoke, CC BY 4.0 (Zenodo 15826133)
get "https://zenodo.org/api/records/15826133/files/Indoor%20Fire%20Smoke.zip/content" indoor.zip 086fbc3b874139276097f4057bc45a3c
# D-Fire, CC0 (gaiasd/DFireDataset); unmodified mirror on Hugging Face with labels included
get https://huggingface.co/datasets/hanvithSai/gavin-dfire-raw/resolve/main/train.tar dfire_train.tar ""
get https://huggingface.co/datasets/hanvithSai/gavin-dfire-raw/resolve/main/test.tar dfire_test.tar ""

# Pyro-SDIS v2.0, Apache-2.0 (pyronear/pyro-sdis): long-range smoke, parquet
mkdir -p data/downloads/pyro
PYRO=https://huggingface.co/datasets/pyronear/pyro-sdis/resolve/main/data
for p in train-00000-of-00003 train-00001-of-00003 train-00002-of-00003 val-00000-of-00001; do
  get "$PYRO/$p.parquet" "pyro/$p.parquet" ""
done

rm -rf data/raw/cqu data/raw/indoor data/raw/dfire data/raw/pyro
mkdir -p data/raw/cqu data/raw/indoor data/raw/dfire
${PYTHON:-python} unpack_pyro.py data/downloads/pyro/*.parquet
unzip -q data/downloads/cqu.zip -d data/raw/cqu
unzip -q data/downloads/indoor.zip -d data/raw/indoor -x "__MACOSX/*" "*.DS_Store"
tar -xf data/downloads/dfire_train.tar -C data/raw/dfire --exclude="._*"
tar -xf data/downloads/dfire_test.tar -C data/raw/dfire --exclude="._*"

for d in cqu indoor dfire pyro; do
  echo "$d: $(find data/raw/$d -iname '*.jpg' ! -name '._*' | wc -l) images, $(find data/raw/$d -name '*.txt' ! -name '._*' | wc -l) txt"
done
# Expected about: cqu 11021 images, indoor 5000, dfire 21527 (17221 train + 4306 test), pyro 12378
