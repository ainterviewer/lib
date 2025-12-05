[private]
default:
    @just --list

[group("Release")]
release VERSION: && publish
    uv version {{ VERSION }}

# TYPE = major, minor, patch. RC=rc to add dev
[group("Release")]
bump TYPE RC="": && publish
    uv version --bump {{ TYPE }} {{ if RC == "rc" { "--bump rc" } else { "" } }}

[group("Release")]
stage:
    uv version --bump rc

[group("Release")]
publish:
    #!/usr/bin/env bash
    VERSION="$(uv version --short)"

    uv sync
    git add .
    git commit -m "Release v${VERSION}"
    git tag -a "v${VERSION}" -m "Release v${VERSION}"
    git push --follow-tags
