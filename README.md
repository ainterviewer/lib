# AInterviewer

**Semi-structured AI interviewing**

> [!WARNING]
> The documentation below is outdated and in the process of being updated.

[Link to diagram](https://drive.google.com/file/d/1Wl3YAC2z2ADNl7SKaR-cCV4fD6mrPj4C/view?usp=sharing)

## Interview Structure

AInterviewer will conduct an interview based on the interview guide.
The workflow generally follows the steps below (to see specific prompts, see `src/ainterviewer/prompts/`):

```pseudo
if framing: framing is used as initial context by the LLM
if introduction: show introduction
for section in sections:
    for question in section.questions:
        ask question.main_question
        while True:
            if question.subquestions:
                probe subquestions
            else:
                probe
if outro: show introduction
```

## Install

To install the project in your current environment run:

```console
make setup
```

### Setup

In order to make the application function, you need to specify the following
environment variables

## Serving

```console
make serve
```

### External dependencies

#### Required

- [redis](https://redis.io/)

#### Dev

- [schemaspy](http://schemaspy.org/)
- [sqlitebrowser](https://sqlitebrowser.org/)
