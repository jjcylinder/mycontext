import sys
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from project_demo.local_embedding import DIMENSIONS, MAX_INPUT_TOKENS, MODEL_NAME, LocalEmbedding, last_token_pool


def fake_service(monkeypatch, tmp_path):
    torch = pytest.importorskip('torch')
    calls, options = [], []

    class Tokenizer:
        @staticmethod
        def from_pretrained(path, **kwargs):
            options.append((str(path), kwargs))
            return Tokenizer()

        def __call__(self, text, **kwargs):
            assert kwargs['truncation'] is False
            ids = torch.ones((1, len(text) + 2), dtype=torch.long)
            return {'input_ids': ids, 'attention_mask': ids}

    class Model:
        @staticmethod
        def from_pretrained(path, **kwargs):
            options.append((str(path), kwargs))
            return Model()

        def eval(self):
            return self

        def __call__(self, input_ids, attention_mask, use_cache):
            assert use_cache is False
            calls.append(input_ids.shape[-1])
            return SimpleNamespace(last_hidden_state=torch.ones((1, input_ids.shape[-1], DIMENSIONS)))

    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(AutoModel=Model, AutoTokenizer=Tokenizer))
    service = LocalEmbedding(tmp_path)
    service.model_path.mkdir()
    (service.model_path / 'config.json').write_text('{}')
    return service, calls, options


def test_local_embeddings_normalize_preserve_order_and_report_tokens(monkeypatch, tmp_path):
    service, calls, options = fake_service(monkeypatch, tmp_path)
    app = FastAPI()
    app.include_router(service.router('test-token'))
    with TestClient(app) as client:
        body = {'model': MODEL_NAME, 'input': ['项目', '会议记录'], 'encoding_format': 'float'}
        assert client.post('/v1/embeddings', json=body).status_code == 401
        response = client.post('/v1/embeddings', json=body, headers={'Authorization': 'Bearer test-token'})
        assert response.status_code == 200
        data = response.json()
        assert [item['index'] for item in data['data']] == [0, 1]
        assert all(len(item['embedding']) == 1024 for item in data['data'])
        assert all(np.isclose(np.linalg.norm(item['embedding']), 1) for item in data['data'])
        assert data['usage']['prompt_tokens'] == 10
        service.load()
        assert len(options) == 2
        assert all(o[1]['local_files_only'] for o in options)
        assert calls == [4, 6]


def test_overflow_rejects_whole_request_before_inference(monkeypatch, tmp_path):
    service, calls, _ = fake_service(monkeypatch, tmp_path)
    app = FastAPI()
    app.include_router(service.router('test-token'))
    with TestClient(app) as client:
        response = client.post('/v1/embeddings', headers={'Authorization': 'Bearer test-token'},
                               json={'model': MODEL_NAME, 'input': ['短文', '字' * MAX_INPUT_TOKENS]})
        assert response.status_code == 413
        assert '未截断' in response.json()['detail']
        assert calls == []


def test_last_token_pool_respects_padding():
    torch = pytest.importorskip('torch')
    hidden = torch.arange(12).reshape(2, 3, 2)
    assert torch.equal(last_token_pool(hidden, torch.tensor([[1, 1, 0], [1, 1, 1]])),
                       torch.stack([hidden[0, 1], hidden[1, 2]]))
    assert torch.equal(last_token_pool(hidden, torch.tensor([[0, 1, 1], [1, 1, 1]])), hidden[:, -1])


@pytest.mark.parametrize(('changes', 'status'), [
    ({'model': 'bad-model'}, 400), ({'dimensions': 512}, 400),
    ({'encoding_format': 'base64'}, 422), ({'input': [123]}, 422), ({'input': []}, 422),
])
def test_local_embeddings_validate_input(changes, status, tmp_path):
    app = FastAPI()
    app.include_router(LocalEmbedding(tmp_path).router('test-token'))
    with TestClient(app) as client:
        response = client.post('/v1/embeddings', json={'model': MODEL_NAME, 'input': '核查进度', **changes},
                               headers={'Authorization': 'Bearer test-token'})
        assert response.status_code == status


@pytest.mark.asyncio
async def test_query_instructions_do_not_modify_ingested_text(monkeypatch):
    from kl_graph.config import cfg
    from kl_graph.ingest.embedder import Embedder
    monkeypatch.setattr(cfg.services.embedding, 'query_instruction', 'Retrieve project evidence.')
    embedder = Embedder(concurrency=1)
    sent = []
    def sync(texts):
        sent.append(texts)
        return [[1.0] for _ in texts]
    async def asynchronous(texts):
        return sync(texts)
    monkeypatch.setattr(embedder, '_embed', sync)
    monkeypatch.setattr(embedder, '_aembed', asynchronous)
    embedder.embed_one('状态？')
    await embedder.aembed_one('状态？')
    embedder.embed_batch(['工作仍待确认。'])
    assert sent == [['Instruct: Retrieve project evidence.\nQuery:状态？']] * 2 + [['工作仍待确认。']]
