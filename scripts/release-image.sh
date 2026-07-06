#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/release-image.sh [minor|major]

Creates and pushes the next permanent BenchFlow image release tag.

The tag format is vMAJOR.MINOR. Patch releases are intentionally unsupported.
The existing build-images GitHub Actions workflow publishes the matching image:

  ghcr.io/<repository-owner>/benchflow:vMAJOR.MINOR

Examples:
  scripts/release-image.sh        # bump latest vMAJOR.MINOR by minor
  scripts/release-image.sh minor  # same as default
  scripts/release-image.sh major  # bump major and reset minor to 0
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "$#" -gt 1 ]]; then
  usage >&2
  exit 2
fi

bump="${1:-minor}"
case "${bump}" in
  major | minor)
    ;;
  *)
    usage >&2
    die "bump must be 'minor' or 'major'"
    ;;
esac

require_cmd git
require_cmd python3

git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "must be run from inside the BenchFlow git repository"

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

branch="$(git branch --show-current)"
if [[ "${branch}" != "main" ]]; then
  die "release tags must be created from local main; current branch is '${branch:-detached}'"
fi

if [[ -n "$(git status --porcelain)" ]]; then
  die "working tree must be clean before creating a permanent release tag"
fi

git fetch --tags --prune-tags origin

head_sha="$(git rev-parse HEAD)"
origin_main_sha="$(git rev-parse origin/main)"
if [[ "${head_sha}" != "${origin_main_sha}" ]]; then
  die "local main must match origin/main before releasing; push or pull first"
fi

latest="$(
  git tag --list 'v*' | python3 -c '
import re
import sys

versions = []
for line in sys.stdin:
    tag = line.strip()
    match = re.fullmatch(r"v([0-9]+)\.([0-9]+)", tag)
    if match:
        versions.append((int(match.group(1)), int(match.group(2)), tag))

if versions:
    versions.sort()
    print(versions[-1][2])
'
)"
latest="${latest:-v0.0}"

if [[ ! "${latest}" =~ ^v([0-9]+)\.([0-9]+)$ ]]; then
  die "latest release tag '${latest}' does not match vMAJOR.MINOR"
fi

major="${BASH_REMATCH[1]}"
minor="${BASH_REMATCH[2]}"

case "${bump}" in
  major)
    major=$((major + 1))
    minor=0
    ;;
  minor)
    minor=$((minor + 1))
    ;;
esac

next_tag="v${major}.${minor}"

if git rev-parse --verify --quiet "refs/tags/${next_tag}" >/dev/null; then
  die "tag ${next_tag} already exists"
fi

echo "Latest release tag: ${latest}"
echo "Next release tag:   ${next_tag}"
echo "Release commit:     ${head_sha}"

git tag -a "${next_tag}" -m "Release ${next_tag}"
git push origin "${next_tag}"

owner="$(git config --get remote.origin.url | python3 -c '
import re
import sys

url = sys.stdin.read().strip()
match = re.search(r"github\.com[:/](?P<owner>[^/]+)/(?P<repo>[^/.]+)(?:\.git)?$", url)
print(match.group("owner") if match else "<repository-owner>")
'
)"

echo "Pushed ${next_tag}."
echo "GitHub Actions will publish ghcr.io/${owner}/benchflow:${next_tag}."
