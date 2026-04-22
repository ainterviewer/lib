import argparse
import importlib.util
import warnings

try:
    import torch  # ty:ignore[unresolved-import]
except ImportError:
    warnings.warn(
        "Torch not installed. Run `pip install torch` if you need to run a PromptGuard server."
    )
import uvicorn

from ainterviewer.safety.serve import app, get_prompt_guard


def parse_args():
    parser = argparse.ArgumentParser(description="PromptGuard utility")

    parser.add_argument(
        "--mode",
        type=str,
        choices=["server", "evaluate"],
        default="server",
        help="Run in server mode or evaluate mode.",
    )

    # Create subgroups for server and evaluate arguments
    server_group = parser.add_argument_group("Server Mode Arguments")
    evaluate_group = parser.add_argument_group("Evaluate Mode Arguments")

    # Arguments exclusive to server functionality
    server_group.add_argument(
        "--model", type=str, default="meta-llama/Prompt-Guard-86M"
    )
    server_group.add_argument("--device", type=str, default="cuda")
    server_group.add_argument("--port", type=int, default=8000)

    # Arguments exclusive to command line evaluation functionality
    evaluate_group.add_argument(
        "--evaluation-type",
        type=str,
        choices=["jailbreak", "injection"],
        default="jailbreak",
    )

    return parser.parse_args()


def run_server(args):
    app.state.prompt_guard = get_prompt_guard(model=args.model, device=args.device)

    uvicorn.run(app, host="0.0.0.0", port=args.port)


def run_evaluation(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt_guard = get_prompt_guard(model="meta-llama/Prompt-Guard-86M", device=device)

    print(f"Enter text to evaluate for {args.evaluation_type}:")
    while True:
        text = input("> ")
        if args.evaluation_type == "jailbreak":
            score = prompt_guard.get_jailbreak_score(text)
            print(f"Jailbreak score: {score}")
        else:
            score = prompt_guard.get_indirect_injection_score(text)
            print(f"Indirect injection score: {score}")


def main():
    args = parse_args()

    if args.mode == "server":
        if importlib.util.find_spec("transformers") is None:
            warnings.warn(
                "Transformers not installed. "
                "Run `pip install transformers` if you need to run a PromptGuard server."
            )
        run_server(args)
    elif args.mode == "evaluate":
        run_evaluation(args)


if __name__ == "__main__":
    main()
