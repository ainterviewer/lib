[private]
default:
    @just --list

# Supply `major, minor, patch` (followed by `--rc` to initialize as a new staging release). Or `just bump rc` to bump the current rc version.
[group("Release")]
bump TYPE RC="": && publish
    uv version --bump {{ TYPE }} {{ if RC == "--rc" { "--bump rc" } else { "" } }}

[group("Release")]
publish:
    #!/usr/bin/env bash
    VERSION="$(uv version --short)"

    uv sync
    git add .
    git commit -m "Release v${VERSION}"
    git tag -a "v${VERSION}" -m "Release v${VERSION}"
    git push --follow-tags
