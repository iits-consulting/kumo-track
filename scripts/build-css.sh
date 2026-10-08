#!/bin/sh
# Compile static/css/app.css → app.build.css with the Tailwind standalone CLI
# (no Node toolchain; the binary is fetched once into .tailwindcss-<version>).
# Usage: scripts/build-css.sh [--watch]
set -e
cd "$(dirname "$0")/.."
VERSION=v4.3.3
BIN="$PWD/.tailwindcss-$VERSION"
if [ ! -x "$BIN" ]; then
  case "$(uname -s)-$(uname -m)" in
    Linux-x86_64)  target=linux-x64 ;;
    Linux-aarch64) target=linux-arm64 ;;
    Darwin-arm64)  target=macos-arm64 ;;
    Darwin-x86_64) target=macos-x64 ;;
    *) echo "unsupported platform: $(uname -s) $(uname -m)" >&2; exit 1 ;;
  esac
  echo "fetching tailwindcss $VERSION ($target)…" >&2
  curl -fsSL -o "$BIN" "https://github.com/tailwindlabs/tailwindcss/releases/download/$VERSION/tailwindcss-$target"
  chmod +x "$BIN"
fi
# run from the static dir so Tailwind's source scan sees exactly index.html + js/
cd src/kumo_track/annotate/static
exec "$BIN" -i css/app.css -o css/app.build.css --minify "$@"
