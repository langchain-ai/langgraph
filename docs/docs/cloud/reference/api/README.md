# Build API Reference Docs

1. Copy the `openapi.json` API specification file to this directory.
1. From the root directory of this repository, run the following command:

        poetry run oad gen-docs -s ./docs/docs/cloud/reference/api/openapi.json -d ./docs/docs/cloud/reference/api/api_ref.md
