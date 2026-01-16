from typing import Self
from datasets import Dataset
import polars as pl

from pydantic import BaseModel


class QRA(BaseModel):
    question: str
    answer: str
    reasoning: str


class QRADataset(BaseModel):
    problems: list[QRA]

    def as_prompt_completion(self) -> Dataset:
        prompt = []
        completion = []
        for sample in self.problems:
            prompt.append(sample.question)
            completion.append(
                f"<think>\n{sample.reasoning}\n</think>\n\n{sample.answer}"
            )
        dataframe = pl.DataFrame({"prompt": prompt, "completion": completion})
        return Dataset.from_polars(dataframe)

    def as_conversational(self) -> Dataset:
        items = [
            {
                "messages": [
                    {
                        "role": "user",
                        "content": sample.question,
                    },
                    {
                        "role": "assistant",
                        "content": f"<think>{sample.reasoning}</think>{sample.answer}",
                    },
                ]
            }
            for sample in self.problems
        ]
        return Dataset.from_list(items)

    def sort(self) -> Self:
        return self.__class__(problems=sorted(self.problems, key=lambda x: x.question))

    def head(self, n: int) -> Self:
        return self.__class__(problems=self.problems[:n])
