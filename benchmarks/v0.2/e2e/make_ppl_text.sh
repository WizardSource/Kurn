#!/usr/bin/env bash
# Fixed ~50 KB English perplexity text: Jane Austen, "Pride and Prejudice" (Project Gutenberg
# #1342, public domain), 51,200 characters (52,140 bytes) from "It is a truth universally acknowledged", CRLF -> LF,
# cut at the last full line. Verified by sha256 so every run and branch scores the same text.
#   make_ppl_text.sh [OUT]   (default /tmp/kurn-e2e-ppl.txt)
set -euo pipefail
OUT=${1:-/tmp/kurn-e2e-ppl.txt}
SHA=7265711fa147bb438d4eb0ec1e4a10c256053f09277fe84ae3e2911f43e5adfe
if [ -s "$OUT" ] && [ "$(sha256sum "$OUT" | cut -d' ' -f1)" = "$SHA" ]; then exit 0; fi
SRC=$(mktemp)
curl -sfL https://www.gutenberg.org/cache/epub/1342/pg1342.txt -o "$SRC"
python3 - "$SRC" "$OUT" <<'EOF'
import sys
t = open(sys.argv[1], encoding="utf-8-sig").read().replace("\r\n", "\n")
i = t.index("It is a truth universally acknowledged")
s = t[i:i + 51200]
open(sys.argv[2], "w").write(s[:s.rindex("\n") + 1])
EOF
rm -f "$SRC"
GOT=$(sha256sum "$OUT" | cut -d' ' -f1)
if [ "$GOT" != "$SHA" ]; then
  echo "warning: $OUT sha256 $GOT differs from the recorded $SHA (Gutenberg edition changed?)" >&2
fi
echo "$OUT $(wc -c < "$OUT") bytes sha256 $GOT"
