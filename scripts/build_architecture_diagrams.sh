#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
figure_dir="$project_root/docs/figures/model-architectures"
source_dir="$figure_dir/source"
config="$figure_dir/mermaid-config.json"
css="$figure_dir/poster.css"

export npm_config_cache="${TMPDIR:-/tmp}/ncr-mermaid-cache"
export PUPPETEER_SKIP_DOWNLOAD=true

if [[ -z "${PUPPETEER_EXECUTABLE_PATH:-}" ]]; then
  for browser in \
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge" \
    "/usr/bin/google-chrome" \
    "/usr/bin/chromium"; do
    if [[ -x "$browser" ]]; then
      export PUPPETEER_EXECUTABLE_PATH="$browser"
      break
    fi
  done
fi

if [[ -z "${PUPPETEER_EXECUTABLE_PATH:-}" ]]; then
  echo "No Chrome/Chromium executable found. Set PUPPETEER_EXECUTABLE_PATH." >&2
  exit 1
fi

render=(npx --yes @mermaid-js/mermaid-cli@11.12.0)

for source in "$source_dir"/*.mmd; do
  name="$(basename "$source" .mmd)"
  "${render[@]}" -i "$source" -o "$figure_dir/$name.svg" \
    -c "$config" -C "$css" -b white
  "${render[@]}" -i "$source" -o "$figure_dir/$name.pdf" \
    -c "$config" -C "$css" -b white -f
  "${render[@]}" -i "$source" -o "$figure_dir/$name.png" \
    -c "$config" -C "$css" -b white -w 2400
done

echo "Rendered poster diagrams in $figure_dir"
