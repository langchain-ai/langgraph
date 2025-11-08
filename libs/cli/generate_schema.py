#!/usr/bin/env python3
"""
Script to generate a JSON schema for the langgraph-cli Config class.

This script creates a schema.json file that can be referenced in langgraph.json files
to provide IDE autocompletion and validation.
"""

import inspect
import json
import textwrap
from pathlib import Path

import msgspec

from langgraph_cli.schemas import (
    AuthConfig,
    CheckpointerConfig,
    Config,
    ConfigurableHeaderConfig,
    CorsConfig,
    HttpConfig,
    IndexConfig,
    SecurityConfig,
    SerdeConfig,
    StoreConfig,
    ThreadTTLConfig,
    TTLConfig,
)


def add_descriptions_to_schema(schema, cls):
    """Add docstring descriptions to the schema properties."""
    if schema.get("description"):
        schema["description"] = inspect.cleandoc(schema["description"])
    elif class_doc := inspect.getdoc(cls):
        schema["description"] = inspect.cleandoc(class_doc)
    # Get attribute docstrings from the class
    attr_docs = {}

    # Also check class annotations for docstrings
    source_lines = inspect.getsourcelines(cls)[0]
    current_attr = None
    docstring_lines = []

    for line in source_lines:
        line = line.strip()

        # Check for attribute definition (TypedDict style)
        if ":" in line and not line.startswith("#") and not line.startswith('"""'):
            parts = line.split(":", 1)
            if len(parts) == 2 and parts[0].strip().isidentifier():
                # If we were collecting a docstring, save it for the previous attribute
                if current_attr and docstring_lines:
                    attr_docs[current_attr] = "\n".join(docstring_lines).strip('"')
                    docstring_lines = []

                current_attr = parts[0].strip()

        # Check for docstring after attribute
        elif line.startswith('"""') and current_attr:
            # Start or end of a docstring
            if len(line) > 3 and line.endswith('"""'):
                # Single line docstring
                attr_docs[current_attr] = line.strip('"')
                current_attr = None
            elif docstring_lines:
                # End of multi-line docstring
                docstring_lines.append(line.rstrip('"'))
                attr_docs[current_attr] = "\n".join(docstring_lines).strip('"')
                docstring_lines = []
                current_attr = None
            else:
                # Start of multi-line docstring
                docstring_lines.append(line.lstrip('"'))

        # Continue multi-line docstring
        elif docstring_lines and current_attr:
            docstring_lines.append(line.strip('"'))

    # Add the last docstring if there is one
    if current_attr and docstring_lines:
        attr_docs[current_attr] = "\n".join(docstring_lines).strip('"')

    # Add descriptions to properties
    if "properties" in schema:
        for prop_name, prop_schema in schema["properties"].items():
            # First try to get from attribute docstrings
            if prop_name in attr_docs and "description" not in prop_schema:
                prop_schema["description"] = textwrap.dedent(attr_docs[prop_name])
            # Fall back to class docstring parsing
            elif class_doc:
                for line in class_doc.split("\n"):
                    if line.strip().startswith(
                        f"{prop_name}:"
                    ) or line.strip().startswith(f'"{prop_name}"'):
                        description = line.split(":", 1)[1].strip()
                        if description and "description" not in prop_schema:
                            prop_schema["description"] = description
                            break

    # Recursively process nested definitions
    if "$defs" in schema:
        for def_name, def_schema in schema["$defs"].items():
            # Find the class that corresponds to this definition
            for potential_cls in [
                Config,
                StoreConfig,
                IndexConfig,
                AuthConfig,
                SecurityConfig,
                HttpConfig,
                CorsConfig,
                ThreadTTLConfig,
                CheckpointerConfig,
                SerdeConfig,
                TTLConfig,
                ConfigurableHeaderConfig,
            ]:
                if potential_cls.__name__ == def_name:
                    add_descriptions_to_schema(def_schema, potential_cls)
                    break

    return schema


def generate_schema():
    """Generate a JSON schema for the Config class using msgspec."""
    # Generate the basic schema
    schema = msgspec.json.schema(Config)

    # Add title and description
    schema["title"] = "LangGraph CLI Configuration"
    schema["description"] = "Configuration schema for langgraph-cli"

    # Add docstring descriptions
    schema = add_descriptions_to_schema(schema, Config)

    # Add constraint that only one of python_version or node_version should be specified
    config_schema = schema["$defs"]["Config"]

    # Create two subschemas: one with python_version and one with node_version
    # Define properties specific to Python projects
    python_specific_props = ["python_version", "pip_config_file"]
    # Define properties specific to Node.js projects
    node_specific_props = ["node_version"]
    # Define properties common to both project types
    common_props = [
        k
        for k in config_schema["properties"]
        if k not in python_specific_props and k not in node_specific_props
    ]

    # Create Python schema with python_version and pip_config_file
    python_schema = {
        "type": "object",
        "properties": {
            # Include Python-specific properties
            **{k: config_schema["properties"][k].copy() for k in python_specific_props},
            # Include common properties
            **{k: config_schema["properties"][k].copy() for k in common_props},
        },
        "required": ["dependencies", "graphs"],
    }

    # Add enum constraint for python_version
    if "python_version" in python_schema["properties"]:
        python_schema["properties"]["python_version"]["enum"] = ["3.11", "3.12", "3.13"]

    # Create Node.js schema with node_version
    node_schema = {
        "type": "object",
        "properties": {
            # Include Node-specific properties
            **{k: config_schema["properties"][k].copy() for k in node_specific_props},
            # Include common properties
            **{k: config_schema["properties"][k].copy() for k in common_props},
        },
        "required": ["node_version", "graphs"],
    }

    # Add enum constraint for node_version
    if "node_version" in node_schema["properties"]:
        node_schema["properties"]["node_version"]["anyOf"] = [
            {"type": "string", "enum": ["20"]},
            {"type": "null"},
        ]

    # Add enum constraint for image_distro
    if "image_distro" in node_schema["properties"]:
        node_schema["properties"]["image_distro"]["anyOf"] = [
            {"type": "string", "enum": ["debian", "wolfi"]},
            {"type": "null"},
        ]

    # Replace the Config schema with a oneOf constraint
    config_schema["oneOf"] = [python_schema, node_schema]

    # Remove the properties field as it's now defined in the oneOf subschemas
    if "properties" in config_schema:
        del config_schema["properties"]

    return schema


def main():
    """Generate the schema and write it to a file."""
    schema = generate_schema()

    # Add versioning to the schema
    import importlib.metadata

    try:
        version = importlib.metadata.version("langgraph_cli").split(".")
        schema_version = f"v{version[0]}"
    except importlib.metadata.PackageNotFoundError:
        schema_version = "v1"

    # Add version to schema
    schema["version"] = schema_version

    config_dir = Path(__file__).parent / "schemas"

    # Create versioned schema file
    versioned_path = config_dir / f"schema.{schema_version}.json"
    with open(versioned_path, "w") as f:
        json.dump(schema, f, indent=2)

    # Also create a latest version
    latest_path = config_dir / "schema.json"
    with open(latest_path, "w") as f:
        json.dump(schema, f, indent=2)

    print(f"Schema written to {versioned_path} and {latest_path}")
    print(
        f"You can now add '$schema: https://raw.githubusercontent.com/langchain-ai/langgraph/refs/heads/main/libs/cli/schemas/schema.json'"
        f" or '$schema: https://raw.githubusercontent.com/langchain-ai/langgraph/refs/heads/main/libs/cli/schemas/schema.{schema_version}.json'"
        " to your langgraph.json files"
    )


if __name__ == "__main__":
    main()
