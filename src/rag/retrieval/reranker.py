from typing import List, Optional

try:
    import torch
    from sentence_transformers import CrossEncoder
except Exception:  
    torch = None
    CrossEncoder = None


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2",
        device: Optional[str] = None,
        enabled: bool = True,
    ):
        self.model_name = model_name
        self.device = device
        self.enabled = enabled and CrossEncoder is not None
        self._model = None
        self._load_error = None

    def _minmax(self, scores: List[float]) -> List[float]:
        if not scores:
            return []

        lo = min(scores)
        hi = max(scores)

        if hi == lo:
            if hi == 0:
                return [0.0 for _ in scores]
            return [1.0 for _ in scores]

        return [(s - lo) / (hi - lo) for s in scores]

    def _load(self) -> None:
        if not self.enabled or self._model is not None or self._load_error is not None:
            return

        try:
            kwargs = {}
            if self.device:
                kwargs["device"] = self.device
            if torch is not None:
                kwargs["activation_fn"] = torch.nn.Sigmoid()

            self._model = CrossEncoder(self.model_name, **kwargs)
        except Exception as e:  # pragma: no cover
            self._load_error = e
            self.enabled = False
            self._model = None

    def available(self) -> bool:
        self._load()
        return self._model is not None

    def get_load_error(self) -> Optional[Exception]:
        return self._load_error

    def score(self, query: str, texts: List[str]) -> List[float]:
        """
        Returns reranker scores normalized to [0, 1].
        """
        if not texts or not self.enabled:
            return [0.0 for _ in texts]

        self._load()
        if self._model is None:
            return [0.0 for _ in texts]

        pairs = [(query, text) for text in texts]

        raw_scores = self._model.predict(
            pairs,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        scores = [float(x) for x in raw_scores]
        return self._minmax(scores)