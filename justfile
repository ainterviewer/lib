[private]
default:
    @just --list

[doc("Bump: supply TYPE (major/minor/patch/rc/stable). Add --rc to initialize a new release candidate (used in staging builds).
Examples:
  just bump patch
  just bump minor --rc
  just bump stable")]
[group("Release")]
bump TYPE RC="": && publish
    #!/usr/bin/env bash
    uv run prek -a

    set -euo pipefail

    if [ -z "{{ TYPE }}" ]; then
        echo "ERROR: pass TYPE=major|minor|patch|rc"
        exit 2
    fi

    if [ -n "{{ RC }}" ]; then
        echo "Initializing a new {{ TYPE }} release candidate"
        uv version --bump {{ TYPE }} --bump rc
    else
        echo "Bumping {{ TYPE }}"
        uv version --bump {{ TYPE }}
    fi

# Install this clone's git hooks (pre-commit + commit-msg).
[group("Release")]
install-hooks:
    uv run prek install

[private]
publish:
    #!/usr/bin/env bash
    set -euo pipefail

    VERSION="$(uv version --short)"

    uv sync
    # Prepend this release's section; --prepend needs the file to exist.
    touch CHANGELOG.md
    uvx git-cliff@2.13.1 --unreleased --tag "v${VERSION}" --prepend CHANGELOG.md
    git add .
    git commit -m "chore(release): v${VERSION}"
    git tag -a "v${VERSION}" -m "v${VERSION}"

    # If VERSION does NOT end with "rc"
    if [[ ! "$VERSION" =~ rc ]]; then
        echo "Creating/Updating 'latest-stable' tag"
        git tag -f latest-stable
    fi

    git push --follow-tags
    # Push latest tag if updated
    if [[ ! "$VERSION" =~ rc ]]; then
        git push -f origin latest-stable
    fi

# Manually build artifacts and create the GitHub release (fallback for when CI is down).
# Uses the gh CLI; run `gh auth status` first to confirm you're logged in.
[group("Release")]
publish-artifacts:
    #!/usr/bin/env bash
    set -euo pipefail

    VERSION="$(uv version --short)"
    TAG="v${VERSION}"

    rm -rf dist
    uv build

    PRERELEASE=()
    if [[ "$VERSION" =~ rc ]]; then
        PRERELEASE=(--prerelease)
    fi

    # Same notes CI would have produced, so the release body is identical either way.
    NOTES="$(mktemp)"
    trap 'rm -f "${NOTES}"' EXIT
    uvx git-cliff@2.13.1 --latest --strip header > "${NOTES}"

    gh release create "${TAG}" dist/*.whl dist/*.tar.gz \
        --title "${TAG}" \
        --notes-file "${NOTES}" \
        "${PRERELEASE[@]}"
