# Test Fixtures

This directory contains test fixtures for the LangGraph SDK.

## OpenAPI Spec (openapi.json)

The `openapi.json` file contains the LangGraph Platform API OpenAPI specification used for testing. This spec ensures that the SDK's type definitions (like `AssistantSelectField`, `ThreadSelectField`, `RunSelectField`, and `CronSelectField`) match the actual API.

### Updating the OpenAPI Spec

The OpenAPI spec should be updated when:
- New fields are added to the API
- The API structure changes
- SDK type definitions are updated

To update the spec:

1. Start a local LangGraph API server or use a deployed instance
2. Fetch the OpenAPI spec from the `/openapi.json` endpoint:
   ```bash
   curl http://localhost:8000/openapi.json > libs/sdk-py/tests/fixtures/openapi.json
   ```
3. Run the tests to ensure compatibility:
   ```bash
   cd libs/sdk-py
   make test
   ```

### Test Coverage

The `test_select_fields_sync.py` tests verify that:
- `AssistantSelectField` matches `/assistants/search` POST endpoint
- `ThreadSelectField` matches `/threads/search` POST endpoint
- `RunSelectField` matches `/threads/{thread_id}/runs` GET endpoint
- `CronSelectField` matches `/runs/crons/search` POST endpoint

If tests fail after updating the spec, the SDK type definitions in `langgraph_sdk/schema.py` need to be updated accordingly.
