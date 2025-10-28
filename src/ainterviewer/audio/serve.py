# TODO: - implement streaming
# whisper streaming:
# - https://github.com/ufal/whisper_streaming
# - https://github.com/QuentinFuxa/WhisperLiveKit
# Dummy code for fastapi integration:
# - https://trinesis.com/blog/articles-1/real-time-audio-processing-with-fastapi-whisper-complete-guide-2024-70
# audio websockets:
# - https://stackoverflow.com/questions/65361686/websockets-bridge-for-audio-stream-in-fastapi
# client side:
# - https://github.com/Ivan-Feofanov/ws-audio-api/tree/master

from fastapi import APIRouter, FastAPI
from faster_whisper import WhisperModel
from pydantic import BaseModel

# from rich.progress import track, Progress
from tqdm import tqdm

app = FastAPI()
router = APIRouter()


class Audio(BaseModel):
    file: str
    language: str | None = None


@router.post("/stt")
async def predict(audio: Audio):
    segments, info = model.transcribe(
        audio.file, language=audio.language, beam_size=args.beam_size
    )

    text = ""

    total = round(info.duration)
    pbar = tqdm(total=total)

    for segment in segments:
        segment_duration = segment.end - segment.start
        pbar.update(segment_duration)

        text += segment.text + " "

    pbar.n = total
    pbar.refresh()

    text = text.strip().replace("  ", " ")

    return {"result": text}


app.include_router(router, prefix="/api", tags=["audio"])


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="large-v3", help="Model size (default: large-v3)"
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run on (default: cpu)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "int8_float16", "float16"],
        help="Compute type (default: int8)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="Number of workers for parallel decoding (default: 1)",
    )

    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="Beam size for decoding (default: 5)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
        num_workers=args.num_workers,
    )

    uvicorn.run(app)
