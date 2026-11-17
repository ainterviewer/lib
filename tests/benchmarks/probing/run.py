"""Benchmark how often the probing agent asks for what the respondent already said.

Each case in `cases/` (and the gitignored `cases/local/`) is an interview
stopped just before a probe. For every prompt variant, the runner samples that
probe many times and an LLM judge labels each one:

- REPEAT: asks for information the respondent has already given
- ELABORATION: goes deeper into something specific the respondent said
- NEW: asks about an aspect not yet addressed

A variant is the probing instruction template at a git revision, or the
working tree; everything else (system prompt, default slots, rendering code)
is the working tree's. Appending `:trim` drops the suggested probes the case
marks as already answered (`covered_suggestions`); `:classify` instead asks the
model under test which ones are answered before each probe, as a production
step would, and reports its agreement with the labels and its latency. Run from the lib directory:

    uv run python tests/benchmarks/probing/run.py \\
        --variant v0.7.11=6666190 --variant HEAD=HEAD --variant HEAD+trim=HEAD:trim

This makes paid OpenRouter calls (OPENROUTER_API_KEY): n x cases x variants
probes, each judged once. How to read the results:

- Requests from all variants are shuffled together, so drift during a run
  (load, provider changes) affects every variant equally.
- The model under test is pinned to one provider without fallbacks:
  OpenRouter's providers serve gpt-oss-120b differently, and fallbacks under
  load made earlier comparisons meaningless. The provider of each response is
  recorded.
- Before sampling, the judge must label each case's `calibration` examples
  correctly.
- With n=30, two identical prompts differed by 8 repeats; use n of about 80
  per variant and compare confidence intervals, not point estimates.
"""

import argparse
import asyncio
import json
import random
import re
import statistics
import subprocess
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from jinja2 import ChoiceLoader, DictLoader, PackageLoader

from ainterviewer.agents.prompts.agent_prompts import (
    ClassificationAgentPrompts,
    ProbingAgentPrompts,
)
from ainterviewer.settings import api_keys

HERE = Path(__file__).parent
TEMPLATE = "probing_agent/instruction_prompt.jinja"
TEMPLATE_PATH = Path("src/ainterviewer/agents/prompts/templates/EN") / TEMPLATE
URL = "https://openrouter.ai/api/v1/chat/completions"
LABELS = ("REPEAT", "ELABORATION", "NEW")


@dataclass
class Case:
    name: str
    description: str
    language: str
    framing: str
    section_description: str
    question_description: str
    suggested_probes: list[str]
    #: Suggested probes the respondent has already answered at this point
    #: (labelled by hand), dropped from the prompt by a `:trim` variant.
    covered_suggestions: list[str]
    max_probes: int | None
    turns: list[dict[str, str]]
    already_given: str
    calibration: dict[str, str]
    #: Interviewer messages before the question that take no answer (e.g. the
    #: introduction).
    preceding: list[str] | None = None
    #: Earlier questions and answers in the interview (survey items included),
    #: shown in the transcript before this question, as production does.
    earlier: list[dict[str, str]] | None = None

    @classmethod
    def load(cls, path: Path) -> "Case":
        return cls(name=path.stem, **json.loads(path.read_text()))

    @property
    def main_question(self) -> str:
        return self.turns[0]["question"]

    @property
    def previous_probes(self) -> list[str]:
        return [turn["question"] for turn in self.turns[1:]]

    @property
    def interview(self) -> str:
        """The question's turns, for the judge and the classifier."""
        return "\n".join(
            f"Interviewer: {turn['question']}\nRespondent: {turn['answer']}"
            for turn in self.turns
        )

    @property
    def transcript(self) -> str:
        """Formatted like `InterviewHistory.get_transcript`."""
        parts = ["Q: " + message for message in self.preceding or []]
        parts += [
            f"Q: {turn['question']}\nA: {turn['answer']}"
            for turn in [*(self.earlier or []), *self.turns]
        ]
        return "\n\n".join(parts)

    def probe_messages(self, template: str, drop: set[str]) -> list[dict[str, str]]:
        suggestions = [probe for probe in self.suggested_probes if probe not in drop]
        loader = ChoiceLoader(
            [
                DictLoader({TEMPLATE: template}),
                PackageLoader("ainterviewer.agents.prompts.templates", "EN"),
            ]
        )
        prompts = ProbingAgentPrompts(lang=self.language, template_loader=loader)
        user = prompts.generate_probing_prompt(
            interview_framing=self.framing,
            section_description=self.section_description,
            question_description=self.question_description,
            main_question=self.main_question,
            interview_transcript=self.transcript,
            suggested_probes="\n".join("- " + probe for probe in suggestions) or None,
            probe_number=len(self.turns),
            max_probes=self.max_probes,
            previous_probes=self.previous_probes,
        )
        return [
            {"role": "system", "content": prompts.system_prompt},
            {"role": "user", "content": user},
        ]


JUDGE_PROMPT = """You are evaluating the next follow-up question in a qualitative research interview.

Interview so far:
{interview}

What the respondent has already said, in short: {already_given}

Candidate next question:
{candidate}

Classify the candidate as exactly one of:
- REPEAT: it mainly asks for information the respondent has already given. This holds even if it is worded differently, asks for a "more specific" or "the most important" version of it, or lists examples.
- ELABORATION: it picks up one specific thing the respondent said and asks them to go deeper into that particular thing (explain it, give an example, say why about that specific thing).
- NEW: it asks about an aspect the respondent has not addressed yet.

Reply with only the label."""


class Runner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.headers = {
            "Authorization": "Bearer " + api_keys.openrouter_api_key.get_secret_value()
        }

    async def complete(
        self,
        client: httpx.AsyncClient,
        model: str,
        messages: list[dict[str, str]],
        temperature: float,
        effort: str,
        pin: bool,
    ) -> tuple[str, str]:
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "reasoning": {"effort": effort},
        }
        if pin:
            body["provider"] = {"order": [self.args.provider], "allow_fallbacks": False}

        data: dict[str, Any] = {}
        for attempt in range(8):
            try:
                response = await client.post(URL, headers=self.headers, json=body)
                data = response.json()
                if response.status_code == 200 and data.get("choices"):
                    content = data["choices"][0]["message"]["content"] or ""
                    return content.strip(), data.get("provider", "?")
            except (httpx.HTTPError, json.JSONDecodeError) as error:
                data = {"error": repr(error)}
            await asyncio.sleep(min(2**attempt, 30) + random.random())
        raise RuntimeError(f"gave up after retries: {data}")

    async def probe(self, client: httpx.AsyncClient, messages: list[dict[str, str]]):
        # Same temperature and reasoning effort as the probing agent's defaults.
        return await self.complete(
            client, self.args.model, messages, 0.7, "low", pin=True
        )

    async def classify(
        self, client: httpx.AsyncClient, case: Case
    ) -> tuple[set[str], float]:
        """Ask the model under test which suggested probes are already answered,
        as a production step before probing would. Returns them and the time
        the call took."""
        if not case.suggested_probes:
            return set(), 0.0
        # The production prompt (`ClassificationAgent.answered_suggestions`),
        # on the same transcript the probing agent gets.
        prompts = ClassificationAgentPrompts(lang=case.language)
        content = prompts.generate_answered_suggestions_prompt(
            transcript=case.transcript,
            main_question=case.main_question,
            suggestions=case.suggested_probes,
        )
        start = time.perf_counter()
        # Temperature 0, like the interview's classification agent.
        reply, _ = await self.complete(
            client,
            self.args.model,
            [
                {"role": "system", "content": prompts.system_prompt},
                {"role": "user", "content": content},
            ],
            0.0,
            "low",
            pin=True,
        )
        elapsed = time.perf_counter() - start
        match = re.search(r"\[([\d,\s]*)\]", reply)
        numbers = (
            {int(n) for n in re.findall(r"\d+", match.group(1))} if match else set()
        )
        answered = {
            probe for i, probe in enumerate(case.suggested_probes, 1) if i in numbers
        }
        return answered, elapsed

    async def judge(self, client: httpx.AsyncClient, case: Case, candidate: str) -> str:
        content = JUDGE_PROMPT.format(
            interview=case.interview,
            already_given=case.already_given,
            candidate=candidate,
        )
        reply, _ = await self.complete(
            client,
            self.args.judge_model,
            [{"role": "user", "content": content}],
            0.0,
            "medium",
            pin=False,
        )
        match = re.search("|".join(LABELS), reply.upper())
        return match.group(0) if match else "UNPARSED"

    async def calibrate(
        self, client: httpx.AsyncClient, cases: list[Case], votes: int = 3
    ) -> bool:
        """Check the judge on each case's calibration examples, by majority of
        `votes` labels, so one stray label does not abort a run."""
        checks = [
            (case, text, want)
            for case in cases
            for text, want in case.calibration.items()
        ]
        labels = await asyncio.gather(
            *(
                self.judge(client, c, text)
                for c, text, _ in checks
                for _ in range(votes)
            )
        )
        wrong, unstable = [], []
        for i, (case, text, want) in enumerate(checks):
            got = Counter(labels[i * votes : (i + 1) * votes])
            majority = got.most_common(1)[0][0]
            if majority != want:
                wrong.append((case, text, want, dict(got)))
            elif len(got) > 1:
                unstable.append((case, text, want, dict(got)))
        print(
            f"judge calibration: {len(checks) - len(wrong)}/{len(checks)} correct "
            f"(majority of {votes})"
        )
        for case, text, want, got in wrong:
            print(f"  [{case.name}] expected {want}, got {got}: {text}")
        for case, text, want, got in unstable:
            print(f"  unstable [{case.name}] {got}: {text}")
        return not wrong


def template_at(revision: str) -> str:
    if revision == "worktree":
        return TEMPLATE_PATH.read_text()
    if revision.endswith(".jinja"):
        return Path(revision).read_text()
    return subprocess.run(
        ["git", "show", f"{revision}:{TEMPLATE_PATH}"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for the proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return centre - half, centre + half


def report(
    results: list[dict[str, Any]], variants: list[str], cases: list[Case]
) -> None:
    def row(title: str, rows: list[dict[str, Any]]) -> str:
        labels = Counter(r["label"] for r in rows)
        judged = sum(labels[label] for label in LABELS)
        low, high = wilson(labels["REPEAT"], judged)
        other = len(rows) - judged
        return (
            f"  {title:20} {labels['REPEAT']:>4}/{judged:<4} {low:>4.0%}–{high:<4.0%} "
            f"{labels['ELABORATION']:>6} {labels['NEW']:>5} {other:>6}"
        )

    header = f"  {'variant':20} {'REPEAT':>9} {'95% CI':>9} {'ELAB':>6} {'NEW':>5} {'other':>6}"
    groups = [(case.name, [case.name]) for case in cases]
    if len(cases) > 1:
        groups.append(("all cases", [case.name for case in cases]))

    for title, names in groups:
        print(f"\n{title}\n{header}")
        for variant in variants:
            rows = [
                r for r in results if r["variant"] == variant and r["case"] in names
            ]
            print(row(variant, rows))


def report_classifier(results: list[dict[str, Any]], cases: list[Case]) -> None:
    """How the classifier's choices compare to the hand labels, and its latency."""
    rows = [r for r in results if "dropped" in r]
    if not rows:
        return
    print("\nclassifier vs hand labels (covered_suggestions)")
    print(f"  {'case':20} {'exact':>7} {'missed':>7} {'extra':>6}")
    for case in cases:
        own = [r for r in rows if r["case"] == case.name]
        if not own:
            continue
        truth = set(case.covered_suggestions)
        exact = sum(set(r["dropped"]) == truth for r in own)
        missed = sum(len(truth - set(r["dropped"])) for r in own)
        extra = sum(len(set(r["dropped"]) - truth) for r in own)
        print(f"  {case.name:20} {exact:>3}/{len(own):<3} {missed:>7} {extra:>6}")
    print("  (missed/extra: suggestions summed over runs)")

    seconds = sorted(r["classify_seconds"] for r in rows if r["classify_seconds"])
    if seconds:
        p90 = seconds[int(0.9 * (len(seconds) - 1))]
        print(
            f"\nclassifier latency: median {statistics.median(seconds):.1f}s, "
            f"p90 {p90:.1f}s (n={len(seconds)}, concurrent load included)"
        )


async def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument(
        "--variant",
        action="append",
        metavar="NAME=REV[:trim|:classify]",
        help=(
            "a git revision, 'worktree', or a path to a .jinja template; ':trim' drops the case's hand-labelled "
            "covered_suggestions, ':classify' drops those the model itself judges "
            "answered (default: HEAD and worktree)"
        ),
    )
    parser.add_argument(
        "--case", action="append", help="case name(s) to run (default: all)"
    )
    parser.add_argument(
        "--n", type=int, default=80, help="samples per case and variant"
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument(
        "--provider",
        default="together",
        help="OpenRouter provider to pin the model to (DeepInfra, production's preference, is often rate-limited)",
    )
    parser.add_argument("--judge-model", default="anthropic/claude-sonnet-5.5")
    parser.add_argument("--ignore-calibration", action="store_true")
    args = parser.parse_args()

    paths = sorted((HERE / "cases").glob("*.json")) + sorted(
        (HERE / "cases/local").glob("*.json")
    )
    cases = [Case.load(path) for path in paths]
    if args.case:
        cases = [case for case in cases if case.name in args.case]
    if not cases:
        raise SystemExit("no cases selected")

    variants = dict(
        v.split("=", 1) for v in args.variant or ["HEAD=HEAD", "worktree=worktree"]
    )
    for case in cases:
        if unknown := set(case.covered_suggestions) - set(case.suggested_probes):
            raise SystemExit(
                f"[{case.name}] covered_suggestions not suggested: {unknown}"
            )

    setups: dict[str, tuple[str, str]] = {}
    for name, spec in variants.items():
        revision, _, option = spec.partition(":")
        if option not in ("", "trim", "classify"):
            raise SystemExit(f"unknown variant option {option!r} in {spec!r}")
        setups[name] = (template_at(revision), option)

    jobs = [(case, name) for case in cases for name in variants for _ in range(args.n)]
    random.shuffle(jobs)

    runner = Runner(args)
    semaphore = asyncio.Semaphore(args.concurrency)

    async with httpx.AsyncClient(timeout=180) as client:
        if not await runner.calibrate(client, cases) and not args.ignore_calibration:
            raise SystemExit(
                "judge failed calibration; fix the cases or pass --ignore-calibration"
            )

        async def run(case: Case, variant: str) -> dict[str, Any]:
            result: dict[str, Any] = {"case": case.name, "variant": variant}
            template, option = setups[variant]
            async with semaphore:
                try:
                    drop: set[str] = set()
                    if option == "trim":
                        drop = set(case.covered_suggestions)
                    elif option == "classify":
                        drop, seconds = await runner.classify(client, case)
                        result |= {"dropped": sorted(drop), "classify_seconds": seconds}
                    messages = case.probe_messages(template, drop)
                    probe, provider = await runner.probe(client, messages)
                    label = await runner.judge(client, case, probe)
                except RuntimeError as error:
                    return result | {"label": "FAILED", "error": str(error)}
            return result | {"probe": probe, "provider": provider, "label": label}

        print(f"sampling {len(jobs)} probes ({args.model} on {args.provider})…")
        results = await asyncio.gather(*(run(case, variant) for case, variant in jobs))

    print(f"providers: {dict(Counter(r.get('provider') for r in results))}")
    report(results, list(variants), cases)
    report_classifier(results, cases)

    out = HERE / "results" / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(
        json.dumps(
            {"args": vars(args), "variants": variants, "results": results},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"\nprobes and labels: {out}")


if __name__ == "__main__":
    asyncio.run(main())
