release VERSION:
  uv version {{VERSION}}
  uv sync
  git add .
  git commit -m "Release v{{VERSION}}"
  git tag -a "v{{VERSION}}" -m "Release v{{VERSION}}"
  git push --follow-tags
