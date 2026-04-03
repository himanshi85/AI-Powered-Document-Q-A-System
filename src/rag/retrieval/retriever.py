from typing import Dict, List, Optional

from src.rag.document_processing.models import RetrievalResult


class HybridRetriever:
    def __init__(
        self,
        vector_store,
        embedding_fn,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        max_chunks_per_document: int = 3,
        reranker: Optional[object] = None,
        rerank_top_n: int = 15,
        rerank_weight: float = 0.35,
    ):
        self.vector_store = vector_store
        self.embedding_fn = embedding_fn
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.max_chunks_per_document = max_chunks_per_document

        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.rerank_weight = rerank_weight

    def _normalize_weights(self):
        total = self.dense_weight + self.sparse_weight
        if total <= 0:
            self.dense_weight = 0.5
            self.sparse_weight = 0.5
            return

        self.dense_weight = self.dense_weight / total
        self.sparse_weight = self.sparse_weight / total

    def _merge_results(
        self,
        dense_results: List[RetrievalResult],
        sparse_results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        merged: Dict[str, RetrievalResult] = {}

        for result in dense_results:
            chunk_id = result.chunk.chunk_id
            if chunk_id not in merged:
                merged[chunk_id] = RetrievalResult(
                    chunk=result.chunk,
                    score=0.0,
                    dense_score=result.dense_score,
                    sparse_score=0.0,
                )
            else:
                merged[chunk_id].dense_score = max(
                    merged[chunk_id].dense_score,
                    result.dense_score,
                )

        for result in sparse_results:
            chunk_id = result.chunk.chunk_id
            if chunk_id not in merged:
                merged[chunk_id] = RetrievalResult(
                    chunk=result.chunk,
                    score=0.0,
                    dense_score=0.0,
                    sparse_score=result.sparse_score,
                )
            else:
                merged[chunk_id].sparse_score = max(
                    merged[chunk_id].sparse_score,
                    result.sparse_score,
                )

        fused_results = []
        for _, result in merged.items():
            hybrid_score = (
                self.dense_weight * result.dense_score
                + self.sparse_weight * result.sparse_score
            )
            result.score = hybrid_score
            fused_results.append(result)

        return fused_results

    def _apply_reranking(
        self,
        query: str,
        results: List[RetrievalResult],
    ) -> List[RetrievalResult]:
        if not results or self.reranker is None:
            return results

        if not getattr(self.reranker, "available", lambda: False)():
            return results

        top_n = min(self.rerank_top_n, len(results))
        head = results[:top_n]
        tail = results[top_n:]

        texts = [r.chunk.content for r in head]
        rerank_scores = self.reranker.score(query, texts)

        for result, rerank_score in zip(head, rerank_scores):
            result.score = (
                (1.0 - self.rerank_weight) * result.score
                + self.rerank_weight * rerank_score
            )
            setattr(result, "rerank_score", float(rerank_score))

        all_results = head + tail
        all_results.sort(
            key=lambda r: (
                r.score,
                getattr(r, "rerank_score", 0.0),
                r.dense_score,
                r.sparse_score,
            ),
            reverse=True,
        )
        return all_results

    def _apply_document_diversity(
        self,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        selected = []
        doc_counts: Dict[str, int] = {}

        for result in results:
            doc_id = result.chunk.document_id
            current_count = doc_counts.get(doc_id, 0)

            if current_count >= self.max_chunks_per_document:
                continue

            selected.append(result)
            doc_counts[doc_id] = current_count + 1

            if len(selected) >= top_k:
                break

        return selected

    def _assign_ranks(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        for i, result in enumerate(results, start=1):
            result.rank = i
        return results

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        self._normalize_weights()

        query_embedding = self.embedding_fn(query)

        candidate_pool = max(top_k * 4, 12)

        dense_results = self.vector_store.search(query_embedding, top_k=candidate_pool)
        sparse_results = self.vector_store.keyword_search(query, top_k=candidate_pool)

        merged_results = self._merge_results(dense_results, sparse_results)

        merged_results.sort(
            key=lambda r: (r.score, r.dense_score, r.sparse_score),
            reverse=True,
        )

        reranked_results = self._apply_reranking(query, merged_results)
        diverse_results = self._apply_document_diversity(reranked_results, top_k=top_k)
        ranked_results = self._assign_ranks(diverse_results)

        return ranked_results