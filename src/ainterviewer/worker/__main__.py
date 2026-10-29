import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "ainterviewer.worker.server:app",
        host=os.environ.get("WORKER_HOST", "0.0.0.0"),
        port=int(os.environ.get("WORKER_PORT", "8700")),
    )


if __name__ == "__main__":
    main()
