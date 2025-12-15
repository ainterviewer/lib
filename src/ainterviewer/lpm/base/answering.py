from ainterviewer.interview_guides.interview_guide import InterviewGuide
from ainterviewer.lpm.base.utils import AnsweringAgent

interviewee = """
Name: Thomas Hansen
Occupation: Veterinarian
Age: 34
"""

answering_agent = AnsweringAgent(interviewee=interviewee)

with open("data/interview_guides/green_transition.json") as f:
    interview_guide = InterviewGuide.model_validate_json(f.read())


interview = [
    {"role": "meta", "content": interview_guide.introduction},
]

print(interview_guide.introduction)

while True:
    question = input("Q: ")
    interview.append({"role": "question", "content": question})
    response = answering_agent.generate_response(interview=interview)
    interview.append({"role": "answer", "content": response})
    print("A:", response)
