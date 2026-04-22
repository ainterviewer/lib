# https://github.com/meta-llama/llama-cookbook/tree/main/getting-started/responsible_ai/prompt_guard

from typing import TYPE_CHECKING

if not TYPE_CHECKING:
    import torch
    from torch.nn.functional import softmax
    from transformers import AutoModelForSequenceClassification, AutoTokenizer


class PromptGuard:
    def __init__(self, model_name="meta-llama/Prompt-Guard-86M", device="cpu"):
        self.device = device
        self.model, self.tokenizer = self._load_model_and_tokenizer(model_name)
        self.model.to(self.device)

    def _load_model_and_tokenizer(self, model_name):
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def get_jailbreak_score(self, text, temperature=1.0):
        probabilities = self._get_class_probabilities(text, temperature)
        return probabilities[0, 2].item()

    def get_indirect_injection_score(self, text, temperature=1.0):
        probabilities = self._get_class_probabilities(text, temperature)
        return (probabilities[0, 1] + probabilities[0, 2]).item()

    def _get_class_probabilities(self, text, temperature=1.0):
        inputs = self.tokenizer(
            text, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = inputs.to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        scaled_logits = logits / temperature
        probabilities = softmax(scaled_logits, dim=-1)
        return probabilities

    def get_jailbreak_scores_for_texts(self, texts, temperature=1.0, max_batch_size=16):
        return self._get_scores_for_texts(texts, [2], temperature, max_batch_size)

    def get_indirect_injection_scores_for_texts(
        self, texts, temperature=1.0, max_batch_size=16
    ):
        return self._get_scores_for_texts(texts, [1, 2], temperature, max_batch_size)

    def _get_scores_for_texts(
        self, texts, score_indices, temperature=1.0, max_batch_size=16
    ):
        all_chunks = []
        text_indices = []
        for index, text in enumerate(texts):
            chunks = [text[i : i + 512] for i in range(0, len(text), 512)]
            all_chunks.extend(chunks)
            text_indices.extend([index] * len(chunks))

        all_scores = [0] * len(texts)
        for i in range(0, len(all_chunks), max_batch_size):
            batch_chunks = all_chunks[i : i + max_batch_size]
            batch_indices = text_indices[i : i + max_batch_size]
            probabilities = self._process_text_batch(batch_chunks, temperature)
            scores = probabilities[:, score_indices].sum(dim=1).tolist()

            for idx, score in zip(batch_indices, scores):
                all_scores[idx] = max(all_scores[idx], score)
        return all_scores

    def _process_text_batch(self, texts, temperature=1.0):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        )
        inputs = inputs.to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        scaled_logits = logits / temperature
        probabilities = softmax(scaled_logits, dim=-1)
        return probabilities
