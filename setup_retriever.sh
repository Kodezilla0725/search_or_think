#!/usr/bin/env bash
# Sets up and starts the BM25 retrieval server on a fresh Colab VM.
# usage: bash setup_retriever.sh <project_dir> [save_index: 1|0] [code_dir]
# save_index 0 skips caching the index tar under <project_dir>/index (use it when <project_dir> is not on Drive).
set -euo pipefail
PROJECT="$1"
SAVE_TO_DRIVE="${2:-1}"
CODE_DIR="${3:-$PROJECT/code}"
INDEX_NAME="wikipedia-dpr-100w"
LOCAL_INDEX="/content/index/$INDEX_NAME"
TAR="$PROJECT/index/$INDEX_NAME.tar"
VENV=/content/retenv
LOG=/content/retriever.log

# 1. Java 21 (Pyserini/Anserini requirement)
if [ -d /usr/lib/jvm/java-21-openjdk-amd64 ]; then JH=/usr/lib/jvm/java-21-openjdk-amd64
elif [ -x /opt/jdk21/bin/java ]; then JH=/opt/jdk21
else
  echo "[setup] installing Java 21"
  if apt-get -qq update && apt-get -qq install -y openjdk-21-jdk-headless > /dev/null 2>&1; then
    JH=/usr/lib/jvm/java-21-openjdk-amd64
  else
    echo "[setup] apt failed, using Temurin 21"
    mkdir -p /opt/jdk21
    curl -sL "https://api.adoptium.net/v3/binary/latest/21/ga/linux/x64/jdk/hotspot/normal/eclipse" \
      | tar xz -C /opt/jdk21 --strip-components=1
    JH=/opt/jdk21
  fi
fi
export JAVA_HOME="$JH"
"$JAVA_HOME/bin/java" -version 2>&1 | head -1

# 2. Separate Python 3.11 venv for pyserini, so it never fights vLLM over torch versions
if [ ! -x "$VENV/bin/python" ]; then
  echo "[setup] building retriever venv"
  pip install -q uv
  uv venv -q --python 3.11 "$VENV"
  uv pip install -q --python "$VENV/bin/python" torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install -q --python "$VENV/bin/python" pyserini fastapi uvicorn requests
fi

# 3. Index: restore from Drive if saved, else download once from Pyserini
if [ ! -d "$LOCAL_INDEX" ]; then
  mkdir -p /content/index
  if [ -f "$TAR" ]; then
    echo "[setup] restoring index from Drive"
    tar xf "$TAR" -C /content/index
  else
    echo "[setup] downloading prebuilt $INDEX_NAME index (one time, several GB)"
    "$VENV/bin/python" - "$INDEX_NAME" "$LOCAL_INDEX" <<'EOF'
import glob, os, shutil, sys
name, dest = sys.argv[1], sys.argv[2]
try:
    from pyserini.util import download_prebuilt_index
    path = download_prebuilt_index(name)
except Exception as e:
    print("download_prebuilt_index failed, falling back:", e)
    from pyserini.search.lucene import LuceneSearcher
    LuceneSearcher.from_prebuilt_index(name)
    path = sorted(glob.glob(os.path.expanduser(f"~/.cache/pyserini/indexes/*{name}*")))[-1]
shutil.move(path, dest)
print("index at", dest)
EOF
    if [ "$SAVE_TO_DRIVE" = "1" ]; then
      mkdir -p "$PROJECT/index"
      NEED=$(du -sb "$LOCAL_INDEX" | cut -f1)
      FREE=$(df -B1 --output=avail "$PROJECT" | tail -1)
      if [ "$FREE" -gt $((NEED + 2000000000)) ]; then
        echo "[setup] saving index tar to Drive (faster restore next session)"
        tar cf "$TAR" -C /content/index "$INDEX_NAME"
      else
        echo "[setup] not enough Drive space, skipping save (index will be re-downloaded next session)"
      fi
    fi
  fi
fi
du -sh "$LOCAL_INDEX"

# 4. Start the server in the background and wait until it answers
if ! curl -s localhost:8000/health > /dev/null; then
  cd "$CODE_DIR"
  INDEX_DIR="$LOCAL_INDEX" setsid nohup "$VENV/bin/python" retriever_server.py > "$LOG" 2>&1 &
  for _ in $(seq 1 90); do
    curl -s localhost:8000/health > /dev/null && break
    sleep 2
  done
fi
curl -s localhost:8000/health || { echo "[setup] server failed, log:"; tail -50 "$LOG"; exit 1; }
echo