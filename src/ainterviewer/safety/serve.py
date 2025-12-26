from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ainterviewer.safety.prompt_guard import PromptGuard
from ainterviewer.types import Device

app = FastAPI()

# TODO:
# https://stackoverflow.com/questions/71298179/fastapi-how-to-get-app-instance-inside-a-router#comment135358797_71298949


def get_prompt_guard(model: str, device: Device) -> PromptGuard:
    return PromptGuard(model_name=model, device=device)


class TextInput(BaseModel):
    text: str


class TextsInput(BaseModel):
    texts: List[str]


@app.post("/class_probabilities")
async def get_class_probabilities(input: TextInput):
    try:
        probabilities = app.state.prompt_guard._get_class_probabilities(input.text)
        return {"probabilities": probabilities.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jailbreak_score")
async def get_jailbreak_score(input: TextInput):
    try:
        score = app.state.prompt_guard.get_jailbreak_score(input.text)
        return {"jailbreak_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/indirect_injection_score")
async def get_indirect_injection_score(input: TextInput):
    try:
        score = app.state.prompt_guard.get_indirect_injection_score(input.text)
        return {"indirect_injection_score": score}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jailbreak_scores_batch")
async def get_jailbreak_scores_batch(input: TextsInput):
    try:
        scores = app.state.prompt_guard.get_jailbreak_scores_for_texts(input.texts)
        return {"jailbreak_scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/indirect_injection_scores_batch")
async def get_indirect_injection_scores_batch(input: TextsInput):
    try:
        scores = app.state.prompt_guard.get_indirect_injection_scores_for_texts(
            input.texts
        )
        return {"indirect_injection_scores": scores}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
