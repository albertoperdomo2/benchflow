#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/release-image.sh [major|minor|patch] [options]

Cuts a full BenchFlow release from your laptop:

  1. creates the next annotated vMAJOR.MINOR.PATCH tag on origin/main
  2. creates the matching GitHub release (auto-generated notes)
  3. makes sure the build-images workflow publishes the permanent image
     and waits for it to finish

Everything is created on the remote from the current tip of origin/main, so
the local branch, worktree, and staged or unstaged changes do not matter and
are never part of the release. The first ever release is always v0.1.0,
regardless of the bump argument.

Options:
  -y, --yes       do not ask for confirmation
      --no-wait   do not wait for the image build to finish
      --draft     create the GitHub release as a draft
      --dry-run   print what would happen and exit
  -h, --help      show this help

Published image:

  ghcr.io/<repository-owner>/benchflow:vMAJOR.MINOR.PATCH

Release image tags are permanent: the workflow cleanup jobs never delete a
package version tagged vMAJOR.MINOR.PATCH.

Examples:
  scripts/release-image.sh          # bump minor (default)
  scripts/release-image.sh patch    # bump patch
  scripts/release-image.sh major    # bump major, reset minor and patch to 0
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

warn() {
  echo "note: $*" >&2
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "$1 is required"
}

bump="minor"
bump_seen=false
assume_yes=false
wait_for_build=true
draft=false
dry_run=false

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    -h | --help)
      usage
      exit 0
      ;;
    -y | --yes)
      assume_yes=true
      ;;
    --no-wait)
      wait_for_build=false
      ;;
    --draft)
      draft=true
      ;;
    --dry-run)
      dry_run=true
      ;;
    major | minor | patch)
      [[ "${bump_seen}" == true ]] && die "bump argument given more than once"
      bump="$1"
      bump_seen=true
      ;;
    *)
      usage >&2
      die "unknown argument '$1'"
      ;;
  esac
  shift
done

require_cmd git
require_cmd gh
require_cmd python3

git rev-parse --is-inside-work-tree >/dev/null 2>&1 \
  || die "must be run from inside the BenchFlow git repository"

repo_root="$(git rev-parse --show-toplevel)"
cd "${repo_root}"

gh auth status >/dev/null 2>&1 || die "gh is not authenticated; run 'gh auth login'"

nwo="$(gh repo view --json nameWithOwner --jq '.nameWithOwner')"
owner="${nwo%%/*}"

# The release is cut from the remote tip of main. Nothing local is read into
# the tag, the release, or the image, so a dirty worktree is fine.
target_sha="$(git ls-remote origin refs/heads/main | awk 'NR == 1 { print $1 }')"
[[ -n "${target_sha}" ]] || die "could not resolve origin/main"

release_tags="$(git ls-remote --tags --refs origin 'v*' | awk '{ sub("refs/tags/", "", $2); print $2 }')"

latest="$(
  printf '%s\n' "${release_tags}" | python3 -c '
import re
import sys

versions = []
for line in sys.stdin:
    tag = line.strip()
    match = re.fullmatch(r"v([0-9]+)\.([0-9]+)\.([0-9]+)", tag)
    if match:
        versions.append((int(match.group(1)), int(match.group(2)), int(match.group(3)), tag))

if versions:
    versions.sort()
    print(versions[-1][3])
'
)"

if [[ -z "${latest}" ]]; then
  latest="none"
  next_tag="v0.1.0"
else
  [[ "${latest}" =~ ^v([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] \
    || die "latest release tag '${latest}' does not match vMAJOR.MINOR.PATCH"

  major="${BASH_REMATCH[1]}"
  minor="${BASH_REMATCH[2]}"
  patch="${BASH_REMATCH[3]}"

  case "${bump}" in
    major)
      major=$((major + 1))
      minor=0
      patch=0
      ;;
    minor)
      minor=$((minor + 1))
      patch=0
      ;;
    patch)
      patch=$((patch + 1))
      ;;
  esac

  next_tag="v${major}.${minor}.${patch}"
fi

if grep -qxF "${next_tag}" <<< "${release_tags}"; then
  die "tag ${next_tag} already exists on origin"
fi

if gh release view "${next_tag}" >/dev/null 2>&1; then
  die "GitHub release ${next_tag} already exists"
fi

image="ghcr.io/${owner}/benchflow:${next_tag}"
workflow="build-images.yaml"
subject="$(gh api "repos/${nwo}/commits/${target_sha}" --jq '.commit.message' 2>/dev/null | head -n 1 || true)"

echo "Latest release tag: ${latest}"
echo "Next release tag:   ${next_tag}"
echo "Release commit:     ${target_sha} (origin/main)${subject:+ ${subject}}"
echo "Release image:      ${image}"

local_head="$(git rev-parse --verify --quiet HEAD || true)"
if [[ -n "${local_head}" && "${local_head}" != "${target_sha}" ]]; then
  warn "local HEAD ${local_head} differs from origin/main; the release uses origin/main"
fi
if [[ -n "$(git status --porcelain)" ]]; then
  warn "local changes are not committed or pushed, so they are not in this release"
fi

if [[ "${dry_run}" == true ]]; then
  echo "Dry run: no tag, release, or image was created."
  exit 0
fi

if [[ "${assume_yes}" != true ]]; then
  read -r -p "Create tag, GitHub release, and image for ${next_tag}? [y/N] " answer
  [[ "${answer}" =~ ^[Yy]$ ]] || die "aborted"
fi

tag_object_sha="$(
  gh api "repos/${nwo}/git/tags" \
    -f tag="${next_tag}" \
    -f message="Release ${next_tag}" \
    -f object="${target_sha}" \
    -f type=commit \
    --jq '.sha'
)"

gh api "repos/${nwo}/git/refs" \
  -f ref="refs/tags/${next_tag}" \
  -f sha="${tag_object_sha}" \
  --jq '.ref' >/dev/null \
  || die "could not create tag ${next_tag} on origin"

echo "Created tag ${next_tag} on origin."

release_args=(
  "${next_tag}"
  --verify-tag
  --target "${target_sha}"
  --title "${next_tag}"
  --generate-notes
  --notes "Container image: \`${image}\`"
)
[[ "${draft}" == true ]] && release_args+=(--draft)

gh release create "${release_args[@]}" \
  || die "tag ${next_tag} was created but the release was not; retry after fixing the failure"

echo "Created GitHub release ${next_tag}."

git fetch --tags --quiet origin "refs/tags/${next_tag}:refs/tags/${next_tag}" 2>/dev/null || true

# The new tag normally triggers build-images. Path filters can suppress that
# trigger, so fall back to an explicit dispatch when no run shows up.
find_run() {
  gh run list \
    --workflow "${workflow}" \
    --branch "${next_tag}" \
    --limit 10 \
    --json databaseId \
    --jq '.[0].databaseId // empty'
}

run_id=""
for _ in $(seq 1 12); do
  run_id="$(find_run || true)"
  [[ -n "${run_id}" ]] && break
  sleep 5
done

if [[ -z "${run_id}" ]]; then
  echo "No image build was triggered by the new tag; dispatching one."
  gh workflow run "${workflow}" --ref "${next_tag}" -f tag="${next_tag}"
  for _ in $(seq 1 12); do
    run_id="$(find_run || true)"
    [[ -n "${run_id}" ]] && break
    sleep 5
  done
  [[ -n "${run_id}" ]] || die "could not find the image build run for ${next_tag}"
fi

run_url="$(gh run view "${run_id}" --json url --jq '.url')"
echo "Image build: ${run_url}"

if [[ "${wait_for_build}" != true ]]; then
  echo "Not waiting for the build. The image will be ${image}."
  exit 0
fi

gh run watch "${run_id}" --exit-status \
  || die "image build failed; see ${run_url}"

echo "Released ${next_tag}: ${image}"
