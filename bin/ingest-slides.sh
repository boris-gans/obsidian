#!/bin/bash
set -e
COURSE="${1:?Usage: $0 <course-folder>}"
VAULT="/Users/mattporteous/Obsidian/School"
SRC="$VAULT/30-Sources/$COURSE"

[ -d "$SRC" ] || { echo "No source folder: $SRC"; exit 1; }

missing=()
for tool in soffice pdfimages pdfinfo pdftoppm; do
  command -v "$tool" >/dev/null 2>&1 || missing+=("$tool")
done
if [ ${#missing[@]} -gt 0 ]; then
  echo "Error: missing required tool(s): ${missing[*]}" >&2
  echo "" >&2
  echo "Install on macOS:" >&2
  for tool in "${missing[@]}"; do
    case "$tool" in
      soffice)
        echo "  $tool   → brew install --cask libreoffice" >&2
        ;;
      pdfimages|pdfinfo|pdftoppm)
        echo "  $tool   → brew install poppler" >&2
        ;;
    esac
  done
  exit 1
fi

mkdir -p "$SRC/pdf" "$SRC/images" "$SRC/slides-png"

for f in "$SRC"/*.pptx; do
  [ -e "$f" ] || continue
  base=$(basename "$f" .pptx)
  [ -f "$SRC/pdf/$base.pdf" ] || soffice --headless --convert-to pdf --outdir "$SRC/pdf" "$f"
done

for pdf in "$SRC"/pdf/*.pdf; do
  base=$(basename "$pdf" .pdf)

  imgdir="$SRC/images/$base"
  if [ ! -d "$imgdir" ]; then
    mkdir -p "$imgdir"
    pdfimages -png "$pdf" "$imgdir/img"
  fi

  slidedir="$SRC/slides-png/$base"
  pages=$(pdfinfo "$pdf" | awk '/^Pages:/ {print $2}')
  mkdir -p "$slidedir"
  existing=$(ls "$slidedir" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$existing" != "$pages" ]; then
    rm -f "$slidedir"/*.png
    pdftoppm -png -r 120 "$pdf" "$slidedir/slide" -f 1 -l "$pages"
    for f in "$slidedir"/slide-*.png; do
      [ -e "$f" ] || continue
      n=$(basename "$f" .png | sed 's/slide-//' | sed 's/^0*//')
      [ -z "$n" ] && n=0
      printf -v padded "%02d" "$n"
      new="$slidedir/slide-$padded.png"
      [ "$f" = "$new" ] || mv "$f" "$new"
    done
  fi
done

if [ ! -f "$SRC/_course.md" ]; then
  echo ""
  echo "⚠️  Missing manifest: $SRC/_course.md"
  echo "   Create it before asking Claude to ingest these slides."
  echo "   See CLAUDE.md → 'Per-course manifest' for the template."
fi

echo "Done."
