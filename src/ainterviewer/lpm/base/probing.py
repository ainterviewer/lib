from ainterviewer.interview_guides.interview_guide import InterviewGuide
from ainterviewer.lpm.base.utils import ProbingAgent

probing_agent = ProbingAgent()

with open("data/interview_guides/green_transition.json") as f:
    interview_guide = InterviewGuide.model_validate_json(f.read())


interview = [
    {"role": "meta", "content": interview_guide.introduction},
]

transcript = probing_agent.generate_transcript(interview)
print(transcript)
for section in interview_guide.question_sections:
    interview.append(
        {
            "role": "meta",
            "content": f"\nSection description: {section.description}\n",
        }
    )
    for question in section.questions:
        n_probes = 0
        interview.append(
            {
                "role": "question",
                "content": question.main_question,
            }
        )
        print("Q:", question.main_question)
        answer = input("A: ")
        interview.append({"role": "answer", "content": answer})
        if question.probes:
            interview.append(
                {
                    "role": "meta",
                    "content": f"\nSuggested probes: {''.join(['\n\t- ' + probe for probe in question.probes])}\n",
                }
            )
            while n_probes < question.max_probes_n:
                response = probing_agent.generate_response(interview)
                interview.append({"role": "probe", "content": response})
                print("P:", response)
                answer = input("A: ")
                interview.append({"role": "answer", "content": answer})
                n_probes += 1
