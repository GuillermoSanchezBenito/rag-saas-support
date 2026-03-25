import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app
from api.dependencies import get_pipeline

@pytest.mark.asyncio
async def test_health_check():
    # test health endpoint
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_query_endpoint_empty_query():
    # mock pipeline dependency for testing
    app.dependency_overrides[get_pipeline] = lambda: None
    
    # test query endpoint with empty string
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/query", json={"query": "   "})
        
    app.dependency_overrides.clear()
    
    assert response.status_code == 400
    assert "Empty query" in response.json()["detail"]
