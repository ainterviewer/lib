from abc import ABC, abstractmethod

# TODO:
# Replace with lpm.clients.chat
# from ollama import Client

QA_MAPPER = {
    "question": "Q: ",
    "answer": "A: ",
    "probe": "P: ",
    "meta": "",
}


class BaseAgent(ABC):
    def __init__(self):
        self.client = Client(host="http://localhost:8667")

    def generate_response(self, interview):
        transcript = self.generate_transcript(interview)
        prompt = self.generate_prompt(transcript)
        print(prompt)
        return self._generate_response(prompt=prompt)

    def generate_transcript(self, messages):
        transcript = "\n".join(
            [
                f"{role}{message['content']}"
                for message in messages
                if (role := QA_MAPPER.get(message["role"])) is not None
            ]
        )

        with open("transcript.txt", "w") as f:
            f.write(transcript)

        return transcript

    @abstractmethod
    def generate_prompt(self, transcript) -> str:
        pass

    def _generate_response(self, prompt):
        response = self.client.generate(
            prompt=prompt,
            model="llama3:70b-text",
            options=dict(
                stop=["Q:", "A:", "P:"],
                temperature=0.3,
            ),
        )
        response = response["response"].strip()

        return response


class ProbingAgent(BaseAgent):
    def generate_prompt(self, transcript):
        return INTERVIEW_TEMPLATE.format(transcript=transcript) + "\nQ: "


class AnsweringAgent(BaseAgent):
    def __init__(self, interviewee):
        super().__init__()
        self.interviewee = interviewee

    def generate_prompt(self, transcript):
        return (
            ANSWERING_TEMPLATE.format(
                interviewee=self.interviewee, transcript=transcript
            )
            + "\nA: "
        )


INTERVIEW_TEMPLATE = """
The following document contains information about and transcript of a qualitative social scientific interview. 

Relevant probes are asked based on the transcript and the information about the interview guide. 

Q = Question
P = Probe
A = Answer

{transcript}"""

ANSWERING_TEMPLATE = """
The following document contains information about and transcript of a qualitative social scientific interview. 

Relevant probes are asked based on the transcript and the information about the interview guide. 

The interviewee is:
{interviewee}

Q = Question
P = Probe
A = Answer

{transcript}"""

ACADEMIC_HELP_SEEKING_DESCRIPTION = """ 
In this interview, university students’ academic help seeking behavior will be investigated.
In the interview, we will be questioning students’ academic help seeking through two cases. 
The aim is to investigate social processes and mechanisms surrounding university students’ help seeking and receiving help in the academic context. 
More specifically, these cases will question why students needed help, who they sought help or support from, the social situation surrounding their need, their relation to the helper(s) and students’ feelings, attitudes and behaviors.
We also want to investigate how university students navigate their social network to access resources and cope with academic challenges. 
We want to shed light on social context, university students’ experiences and how their social ties and relations impact their academic help seeking and getting help, as well as their attitudes and resilience in the face of academic challenges.
In the general context, the following questions will be examined: How do university students decide to ask for help, from which resources do they seek help from, what kind of help is needed the most, and students’ attitudes and attributions regarding different sources of help will be examined. 
It is important to note that the focus is not the outcome of help seeking but rather on students’ experiences, attitudes, and thoughts around seeking support/help. 
"""
