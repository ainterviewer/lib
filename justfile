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

[private]
publish:
    #!/usr/bin/env bash
    set -euo pipefail

    VERSION="$(uv version --short)"

    uv sync
    git add .
    git commit -m "Release v${VERSION}"
    git tag -a "v${VERSION}" -m "Release v${VERSION}"

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
