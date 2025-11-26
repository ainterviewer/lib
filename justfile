release VERSION:
  uv version {{VERSION}}
  uv sync
  git add .
  git commit -m "Bump version to v{{VERSION}}"
  git tag "v{{VERSION}}"
