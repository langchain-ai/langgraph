import os
import httpx
from typing import Optional
class NanoEmpireClient:
    def __init__(self, api_key: Optional[str] = None, base_url: str = "https://api.nanoempireai.com"):
        self.base_url = base_url
        self.api_key = api_key or os.getenv("NANOEMPIRE_API_KEY")
        self.client = httpx.AsyncClient(timeout=30)
    async def claim_faucet(self, agent_id: str) -> dict:
        resp = await self.client.post(f"{self.base_url}/faucet/claim", json={"agent_id": agent_id})
        return resp.json()
    async def call_tool(self, tool: str, params: dict, agent_id: str) -> dict:
        headers = {"X-Agent-ID": agent_id}
        if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"
        resp = await self.client.post(f"{self.base_url}/mcp/{tool}", json=params, headers=headers)
        return resp.json()
