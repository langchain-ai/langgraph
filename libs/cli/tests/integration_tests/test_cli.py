import pytest
import requests

from langgraph_cli.templates import TEMPLATE_ID_TO_CONFIG


@pytest.mark.parametrize("template_key", TEMPLATE_ID_TO_CONFIG.keys())
def test_template_urls_work(template_key: str) -> None:
    """Integration test to verify that all template URLs are reachable."""
    _, _, template_url = TEMPLATE_ID_TO_CONFIG[template_key]
    response = requests.head(template_url)
    # Returns 302 on a successful HEAD request
    assert response.status_code == 302, f"URL {template_url} is not reachable."
