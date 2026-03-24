import pytest
from httpx import AsyncClient, ASGITransport
from api.main import app

@pytest.mark.asyncio
async def test_health_check():
    """Test the health endpoint."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "message": "Service is healthy"}

@pytest.mark.asyncio
async def test_query_endpoint_empty_query():
    """Test the query endpoint with an empty query string."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.post("/query", json={"query": "   "})
    assert response.status_code == 400
    assert "Query cannot be empty" in response.json()["detail"]
