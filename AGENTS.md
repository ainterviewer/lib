# AGENTS.md

## Branching & Merging

History on `main` is linear: no merge commits.

- Smaller fixes and self-contained features, a commit that stands on its own,
  may be committed straight to `main`. So does the release commit
  `just publish` makes.
- Anything larger, or spread over several commits, goes on a branch
  (`feat/…`, `fix/…`, `refactor/…`).
- To land a branch, rebase onto `main` and fast-forward:
  `git rebase main <branch> && git switch main && git merge --ff-only <branch>`.
  Several branches land one after another, each rebased onto the result of
  the last.
- Run the checks again after a rebase that touched the same files as `main`;
  a clean rebase is not proof the two changes compose.
- On GitHub, PRs merge by rebase or squash; merge commits are disabled.
- Clones should set `git config pull.rebase true` and
  `git config merge.ff only`, so a stray `git pull` or `git merge` cannot
  create a merge commit.

### Work on a branch in its own worktree

Start a branch in a git worktree, not by switching branches in the main
checkout. Several features can then be in progress at once, each with its own
working tree, and moving between them is a `cd`, not a stash. It also keeps
out of the way of anyone (or any agent session) working in another checkout:
before you edit, check `git status` in the checkout you're about to touch, and
if it holds someone else's uncommitted work, start a worktree instead of
editing beside it.

Worktrees live outside the repos, under `stack/.worktrees/<topic>/`, with
one directory per repo named as the repo, all on the same branch name. The
same practice is written up in every repo of the stack that has an AGENTS.md.

```bash
# From this repo's main checkout:
git worktree add -b feat/my-change ../.worktrees/my-change/lib main
# A change that spans repos: the same branch in each, side by side.
git -C ../backend worktree add -b feat/my-change \
  ../.worktrees/my-change/backend main
```

- **Sibling layout matters.** Paths between repos (`../lib`, `../backend`)
  resolve inside the same topic directory, so the worktrees of one change see
  each other.
- **A worktree starts with tracked files only.** There is no `.venv`,
  `config.toml` or `.env`. Run `uv sync` in it for its own venv, or borrow the
  main checkout's: `PYTHONPATH=src ../../../lib/.venv/bin/python -m pytest`
  (the `PYTHONPATH` makes it import this worktree's code, not the main
  checkout's). Copy config from `config.example.toml` and `.env.example`.
- **Testing consumers against unreleased lib changes** without touching their
  pin or venv: from the backend's or proxy's worktree,
  `PYTHONPATH=../lib/src ../../../backend/.venv/bin/python -m pytest`, and
  `--extra-search-path ../lib/src` for `ty check`.
- **Landing.** `main` stays checked out in the main checkout, and git does not
  let two worktrees check out the same branch, so rebase in the worktree
  (`git rebase main`), then fast-forward from the main checkout
  (`git merge --ff-only <branch>`).
- **Cleaning up.** Once it has landed:
  `git worktree remove ../.worktrees/<topic>/<repo>` from the main checkout,
  then `git branch -d <branch>`. `git worktree list` shows what is open;
  `git worktree prune` forgets a worktree whose directory was deleted by hand.

## A model that is not there

`chat()` (`lpm/clients.py`) is the one place a provider's error is read. A
503 or 429 comes out as `ModelUnavailable` (`exceptions.py`), with a
`reason` and a `retry_after`, taken from the error's `code` and `Retry-After`
-- our EC2 proxy names each refusal in its body (`Refusal` in the proxy
repo), and the OpenAI SDK and any-llm carry the code through. A refusal the
proxy calls final (`capacity_unavailable`, `inference_disabled`) has no
`retry_after`: nothing is on its way up to wait for.

- **Every fallback re-raises it.** A fallback (`except Exception`, carry on
  with a default) is for a reply that went wrong. With no model it only
  hides the outage, and the next call spends its own time finding the same
  thing -- that is how a cold start once took three minutes of typing dots
  to reach the respondent. Put `except ModelUnavailable: raise` ahead of any
  new `except Exception` around an agent call, and test it.
- **Deciding to wait is the caller's.** The library raises at once and never
  sleeps on a refusal; the self-hosted model's client runs with no SDK
  retries for that reason. The backend owns the respondent's socket, so it
  tells them and waits.

## Commits

Commit messages follow Conventional Commits (enforced by the `commit-msg`
hook; run `just install-hooks` once per clone). They are the source of the
release notes users see in the app's "What's new", so the type decides who
hears about a change: `feat`, `fix` and `perf` reach users, everything else
(`refactor`, `docs`, `test`, `build`, `ci`, `chore`) stays internal. When
unsure, ask "would a researcher notice?"

### Preview features

Preview features are given to users one by one. Each is a `PreviewFeature`
(the backend's `app/types.py`), and a user holds a list of them
(`user.preview_features`, set by an admin or handed out by an invitation's
`user_preview_features`). The list rides in the auth token, and a route is
gated on one feature with `Depends(require_preview(PreviewFeature.X))` in
the backend; in the frontend, `hasPreview(user, 'x')` and a sidebar item's
`previewFeature`.

A change only users with a preview feature can see takes a `Preview:`
trailer naming that feature -- its `PreviewFeature` value -- in the footer
after a blank line:

```
feat(analysis): add code similarity search

Preview: codebook
```

Without the trailer the change is announced to every user. A `preview` scope
does **not** do this — `feat(preview): …` is an ordinary feature to the
tooling. The rules:

- **Fixes count.** A `fix` or `perf` to a preview feature takes the trailer too.
- **One name per feature, reused everywhere**, across lib, backend and
  frontend: it is the key that collects the feature's history and decides
  who is shown its release highlights (`Highlight.feature`). Use the
  `PreviewFeature` value; a new feature is a new member first.
- **Split mixed commits.** A change touching both a preview and a general path
  is two commits.
- **Squash merges** must keep the trailer in the squash message.

Opening a feature to everyone is its own commit, the one that removes the
gate and the `PreviewFeature` member, with a `Graduates:` trailer:

```
feat(analysis): make the codebook available to all users

Graduates: codebook
```

Removing the member needs no migration: user rows and tokens still naming it
are read through `PreviewFeatureList`, which drops names it does not know.
Its announcement is written from the feature's earlier `Preview:` entries, so
a feature that shipped without them graduates with nothing to describe.
Entries before the list existed used `explore` for what is now `codebook`.
