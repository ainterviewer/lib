release VERSION: && publish
  uv version {{VERSION}}

bump TYPE: && publish
  uv version --bump {{TYPE}}

publish:
  #!/usr/bin/env bash
  VERSION="$(uv version --short)"

  uv sync
  git add .
  git commit -m "Release v${VERSION}"
  git tag -a "v${VERSION}" -m "Release v${VERSION}"
  git push --follow-tags
