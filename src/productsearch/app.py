from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .service import SearchService
from .catalog_search import CatalogSearchService, SEARCH_MODELS

WEB = Path(__file__).parent / "web"


def create_app(service=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.search = service if service is not None else CatalogSearchService()
        app.state.legacy = None
        app.state.current = None
        yield

    def current():
        if isinstance(app.state.search, CatalogSearchService):
            return app.state.search
        if app.state.current is None:
            app.state.current = CatalogSearchService(app.state.search.root)
        return app.state.current

    def legacy():
        if isinstance(app.state.search, SearchService):
            return app.state.search
        if app.state.legacy is None:
            app.state.legacy = SearchService(app.state.search.root)
        return app.state.legacy

    app = FastAPI(title="MiniMatch", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=WEB), name="static")

    @app.get("/")
    def home():
        return FileResponse(WEB / "index.html")

    @app.get("/api/health")
    def health():
        return {"status": "ready", "products": len(current().catalog), "dense_available": True}

    @app.get("/api/application")
    def application():
        return current().info()

    @app.get("/api/products/{product_id}")
    def product_detail(product_id: str):
        try:
            return current().detail(product_id)
        except KeyError as exc:
            raise HTTPException(404, "商品不存在") from exc

    @app.get("/api/metrics", include_in_schema=False)
    def metrics():
        return legacy().report

    @app.get("/api/queries", include_in_schema=False)
    def queries(q: str = Query("", max_length=240)):
        frame = legacy().queries
        if q.strip():
            frame = frame[frame["query"].str.contains(q.strip(), case=False, regex=False)]
        return frame.to_dict("records")

    @app.get("/api/search")
    def search(q: str = Query(min_length=1, max_length=240), model: Literal["hybrid", "minilm", "keyword", "minilm_pretrained", "bm25", "linear", "lambdamart", "dense_pretrained", "dense_finetuned", "dense_hybrid_pretrained", "dense_hybrid", "dense_v2", "dense_v2_pretrained", "hybrid_v2"] = Query("hybrid", json_schema_extra={"enum": ["hybrid", "minilm", "keyword"]}), limit: int = Query(12, ge=1, le=48), offset: int = Query(0, ge=0, le=200)):
        try:
            if model in SEARCH_MODELS:
                return current().search(q.strip(), model, limit, offset)
            if offset:
                raise ValueError("历史模型不支持分页。")
            return legacy().search(q.strip(), model, limit)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/api/benchmark/{query_group}", include_in_schema=False)
    def benchmark(query_group: str):
        try:
            return legacy().benchmark(query_group)
        except KeyError as exc:
            raise HTTPException(404, "测试查询不存在") from exc

    @app.get("/api/dense/metrics", include_in_schema=False)
    def dense_metrics():
        if not legacy().dense_available():
            return {"available": False}
        import json
        from .dense_service import choose_serving_policy
        report = json.loads((legacy().root / "reports/dense_v1/metrics.json").read_text("utf-8"))
        return {"available": True, **report,"serving_policy":choose_serving_policy(report["selection"])}

    @app.get("/api/dense/queries", include_in_schema=False)
    def dense_queries(q: str = Query("", max_length=240)):
        try:
            frame = legacy().dense().queries
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        if q.strip():
            frame = frame[frame["query"].str.contains(q.strip(), case=False, regex=False)]
        return frame.to_dict("records")

    @app.get("/api/dense/benchmark/{query_group}", include_in_schema=False)
    def dense_benchmark(query_group: str):
        try:
            return legacy().dense().benchmark(query_group)
        except KeyError as exc:
            raise HTTPException(404, "测试查询不存在") from exc
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/dense-v2/metrics", include_in_schema=False)
    def dense_v2_metrics():
        import json
        directory = legacy().root / "reports/dense_v2"
        if not legacy().dense_v2_available():
            progress = json.loads((directory / "progress.json").read_text("utf-8")) if (directory / "progress.json").exists() else {"status": "preparing"}
            return {"available": False, "progress": progress}
        return {"available": True, **json.loads((directory / "metrics.json").read_text("utf-8"))}

    @app.get("/api/dense-v2/queries", include_in_schema=False)
    def dense_v2_queries(q: str = Query("", max_length=240)):
        try:
            frame = legacy().dense_v2().queries
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
        if q.strip():
            frame = frame[frame["query"].str.contains(q.strip(), case=False, regex=False)]
        return frame.to_dict("records")

    @app.get("/api/dense-v2/benchmark/{query_group}", include_in_schema=False)
    def dense_v2_benchmark(query_group: str):
        try:
            return legacy().dense_v2().benchmark(query_group)
        except KeyError as exc:
            raise HTTPException(404, "改进实验测试查询不存在") from exc
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc

    return app


app = create_app()
