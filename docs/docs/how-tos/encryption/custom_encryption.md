# Add custom at-rest encryption

!!! tip "Prerequisites"

    This guide assumes familiarity with the following concepts:

      *  [**LangGraph Platform**](../../concepts/langgraph_platform.md)

???+ note "Support by deployment type"

    Custom at-rest encryption is supported for **self-hosted** LangGraph Platform deployments only (Python graphs only).

This guide shows how to add custom at-rest encryption to your LangGraph Platform application for self-hosted deployments. This allows you to encrypt sensitive data before it's stored in the database, using your own encryption keys and services.

!!! warning "Python only"

    Custom encryption is currently only supported for Python graphs. JavaScript/TypeScript support is not yet available.

## Overview

Custom at-rest encryption allows you to:

- **Encrypt JSON fields** - Selectively encrypt metadata, values, and other JSON data on assistants, threads, runs, and crons
- **Encrypt checkpoint blobs** - Encrypt opaque checkpoint data stored during graph execution
- **Use your own encryption service** - Integrate with AWS KMS, Google Cloud KMS, HashiCorp Vault, or any other encryption service
- **Multi-tenant isolation** - Use different encryption keys per tenant or customer

The encryption system provides a decorator-based API similar to the Auth system, allowing you to define custom encryption and decryption handlers that are executed server-side.

## How it works

1. **Define handlers** - Create encryption and decryption functions decorated with `@encrypt.blob`, `@decrypt.blob`, `@encrypt.json`, and `@decrypt.json`
2. **Configure** - Add the path to your encryption module in `langgraph.json`
3. **Pass context** - Send encryption context (like tenant ID and key ID) via the `X-Encryption-Context` header
4. **Automatic encryption** - LangGraph automatically encrypts data before storing and decrypts on retrieval

## Add custom encryption to your deployment

### 1. Create an encryption module

Create a Python file (e.g., `encrypt.py`) with your encryption handlers:

```python
from langgraph_sdk import Encrypt, EncryptionContext
import boto3

# Create the Encrypt instance
encrypt = Encrypt()

# Initialize your encryption service (example: AWS KMS)
kms_client = boto3.client('kms')


@encrypt.blob
async def encrypt_checkpoint(ctx: EncryptionContext, blob: bytes) -> bytes:
    """Encrypt checkpoint blob data.

    Args:
        ctx: Encryption context with metadata containing:
             - tenant_id: Tenant identifier for multi-tenant isolation
             - key_id: KMS key identifier to use
        blob: The raw checkpoint bytes to encrypt

    Returns:
        Encrypted checkpoint bytes
    """
    # Extract context from X-Encryption-Context header
    tenant_id = ctx.metadata.get("tenant_id", "default")
    key_id = ctx.metadata.get("key_id")

    # Encrypt using your KMS service
    response = kms_client.encrypt(
        KeyId=key_id,
        Plaintext=blob,
        EncryptionContext={'tenant_id': tenant_id}
    )
    return response['CiphertextBlob']


@decrypt.blob
async def decrypt_checkpoint(ctx: EncryptionContext, blob: bytes) -> bytes:
    """Decrypt checkpoint blob data.

    The context (tenant_id, key_id) is automatically retrieved from storage,
    so callers don't need to re-pass the X-Encryption-Context header.

    Args:
        ctx: Encryption context with stored metadata
        blob: The encrypted checkpoint bytes to decrypt

    Returns:
        Decrypted checkpoint bytes
    """
    # Context is automatically retrieved from storage
    tenant_id = ctx.metadata.get("tenant_id", "default")

    # Decrypt using your KMS service
    response = kms_client.decrypt(
        CiphertextBlob=blob,
        EncryptionContext={'tenant_id': tenant_id}
    )
    return response['Plaintext']


@encrypt.json
async def encrypt_json_data(ctx: EncryptionContext, data: dict) -> dict:
    """Encrypt JSON data fields.

    This example demonstrates a practical encryption strategy:
    - "owner" field: Left unencrypted for search/filtering (indexed field)
    - Fields with "my.customer.org/" prefix: Encrypt VALUES only (sensitive data)
    - All other fields: Pass through unencrypted for search/filtering

    Note: Searching on encrypted fields will not work with non-deterministic
    encryption. Only unencrypted fields can be reliably searched.

    Args:
        ctx: Encryption context with metadata containing tenant_id and key_id
        data: The plaintext data dictionary

    Returns:
        Encrypted data dictionary
    """
    tenant_id = ctx.metadata.get("tenant_id", "default")
    key_id = ctx.metadata.get("key_id")

    encrypted_data = {}
    for key, value in data.items():
        if key.startswith("my.customer.org/"):
            # Encrypt the VALUE for customer-specific sensitive fields
            response = kms_client.encrypt(
                KeyId=key_id,
                Plaintext=str(value).encode(),
                EncryptionContext={'tenant_id': tenant_id}
            )
            import base64
            encrypted_data[key] = base64.b64encode(
                response['CiphertextBlob']
            ).decode()
        else:
            # Pass through unencrypted (including "owner" for search)
            encrypted_data[key] = value

    return encrypted_data


@decrypt.json
async def decrypt_json_data(ctx: EncryptionContext, data: dict) -> dict:
    """Decrypt JSON data fields.

    Inverse of `encrypt_json_data`. Decrypts VALUES for fields with
    "my.customer.org/" prefix, passes through all other fields.

    The encryption context (tenant_id, key_id) is automatically retrieved
    from storage, so callers don't need to re-pass the X-Encryption-Context header.

    Args:
        ctx: Encryption context with stored metadata
        data: The encrypted data dictionary

    Returns:
        Decrypted data dictionary
    """
    tenant_id = ctx.metadata.get("tenant_id", "default")

    decrypted_data = {}
    for key, value in data.items():
        if key.startswith("my.customer.org/") and isinstance(value, str):
            # Decrypt the VALUE for customer-specific fields
            import base64
            ciphertext = base64.b64decode(value.encode())
            response = kms_client.decrypt(
                CiphertextBlob=ciphertext,
                EncryptionContext={'tenant_id': tenant_id}
            )
            decrypted_data[key] = response['Plaintext'].decode()
        else:
            # Pass through unencrypted fields
            decrypted_data[key] = value

    return decrypted_data
```

### 2. Configure in langgraph.json

Add the path to your encryption module in your `langgraph.json`:

```json hl_lines="7-9"
{
  "dependencies": ["."],
  "graphs": {
    "agent": "./agent.py:graph"
  },
  "env": ".env",
  "encrypt": {
    "path": "./encrypt.py:encrypt"
  }
}
```

### 3. Pass encryption context in requests

When making requests to your deployment, include the `X-Encryption-Context` header with a base64-encoded JSON object containing your encryption parameters:

=== "Python Client"

    ```python
    import base64
    import json
    from langgraph_sdk import get_client

    # Define encryption context
    encryption_context = {
        "tenant_id": "customer-123",
        "key_id": "arn:aws:kms:us-east-1:123456789:key/abc-def-123"
    }

    # Encode as base64
    encoded_context = base64.b64encode(
        json.dumps(encryption_context).encode()
    ).decode()

    # Create client with encryption context header
    client = get_client(
        url="http://localhost:2024",
        headers={"X-Encryption-Context": encoded_context}
    )

    # Create a thread with encrypted metadata
    thread = await client.threads.create(
        metadata={
            "owner": "user-456",  # Unencrypted (for search)
            "my.customer.org/email": "john@example.com",  # Encrypted
            "my.customer.org/phone": "+1-555-0123",  # Encrypted
        }
    )
    ```

=== "Python RemoteGraph"

    ```python
    import base64
    import json
    from langgraph.pregel.remote import RemoteGraph

    # Define encryption context
    encryption_context = {
        "tenant_id": "customer-123",
        "key_id": "arn:aws:kms:us-east-1:123456789:key/abc-def-123"
    }

    # Encode as base64
    encoded_context = base64.b64encode(
        json.dumps(encryption_context).encode()
    ).decode()

    # Create remote graph with encryption context
    remote_graph = RemoteGraph(
        "agent",
        url="http://localhost:2024",
        headers={"X-Encryption-Context": encoded_context}
    )

    # The encryption context is automatically used for all operations
    result = await remote_graph.ainvoke(
        {"messages": [{"role": "user", "content": "Hello"}]},
        config={"configurable": {"thread_id": "thread-1"}}
    )
    ```

=== "CURL"

    ```bash
    # Encode encryption context
    ENCRYPTION_CONTEXT=$(echo -n '{"tenant_id":"customer-123","key_id":"arn:aws:kms:us-east-1:123456789:key/abc-def-123"}' | base64)

    # Create thread with encrypted metadata
    curl -X POST http://localhost:2024/threads \
      -H "Content-Type: application/json" \
      -H "X-Encryption-Context: $ENCRYPTION_CONTEXT" \
      -d '{
        "metadata": {
          "owner": "user-456",
          "my.customer.org/email": "john@example.com",
          "my.customer.org/phone": "+1-555-0123"
        }
      }'
    ```

## Important considerations

### Searchability

Fields with encrypted **values** cannot be reliably searched when using non-deterministic encryption (which most real-world encryption provides). In the example above:

- ✅ **Can search**: `owner` field (unencrypted)
- ❌ **Cannot search**: `my.customer.org/email` field (encrypted value)

Design your metadata schema carefully:
- Put searchable/filterable fields in unencrypted metadata
- Put sensitive data that doesn't need to be searched in encrypted fields
- Consider using prefixes to denote which fields should be encrypted

### Decryption doesn't require re-passing context

Once data is encrypted with a context, that context is stored alongside the encrypted data. When retrieving data, you **do not** need to pass the `X-Encryption-Context` header again - the stored context is automatically used for decryption.

```python
# Encrypt: pass context
client = get_client(
    url="http://localhost:2024",
    headers={"X-Encryption-Context": encoded_context}
)
thread = await client.threads.create(metadata={"my.customer.org/secret": "value"})

# Decrypt: no context needed
client2 = get_client(url="http://localhost:2024")
retrieved = await client2.threads.get(thread["thread_id"])
# The secret field is automatically decrypted using the stored context
```

### What gets encrypted

The encryption handlers are called for:

**JSON encryption** (`@encrypt.json` / `@decrypt.json`):
- `thread.metadata`
- `thread.values`
- `assistant.metadata`
- `assistant.context`
- `run.metadata`
- `run.kwargs`
- `cron.metadata`
- `cron.payload`

**Blob encryption** (`@encrypt.blob` / `@decrypt.blob`):
- Checkpoint blobs (complex state data)

### Model-specific handlers

You can register different encryption handlers for different model types using `@encrypt.json.thread`, `@encrypt.json.assistant`, etc.:

```python
from langgraph_sdk import Encrypt, EncryptionContext

encrypt = Encrypt()

# Default handler for models without specific handlers
@encrypt.json
async def default_encrypt(ctx: EncryptionContext, data: dict) -> dict:
    return standard_encrypt(data)

# Thread-specific handler (uses different KMS key)
@encrypt.json.thread
async def encrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
    return encrypt_with_thread_key(data)

# Assistant-specific handler
@encrypt.json.assistant
async def encrypt_assistant(ctx: EncryptionContext, data: dict) -> dict:
    return encrypt_with_assistant_key(data)

# Same pattern for decryption
@decrypt.json
async def default_decrypt(ctx: EncryptionContext, data: dict) -> dict:
    return standard_decrypt(data)

@decrypt.json.thread
async def decrypt_thread(ctx: EncryptionContext, data: dict) -> dict:
    return decrypt_with_thread_key(data)
```

### Security best practices

1. **Never hardcode keys** - Use environment variables or secret managers
2. **Use KMS services** - Don't implement your own encryption algorithms
3. **Audit logging** - Log encryption/decryption operations for compliance
4. **Key rotation** - Plan for periodic key rotation
5. **Access control** - Restrict access to encryption keys using IAM policies

## Example: Multi-tenant encryption

Here's a complete example showing multi-tenant encryption where each customer has their own encryption key:

```python
from langgraph_sdk import Encrypt, EncryptionContext
import boto3
import os

encrypt = Encrypt()
kms_client = boto3.client('kms', region_name=os.getenv('AWS_REGION'))

# Map of tenant IDs to KMS key ARNs (in production, fetch from a database)
TENANT_KEYS = {
    "customer-123": "arn:aws:kms:us-east-1:123456789:key/abc-123",
    "customer-456": "arn:aws:kms:us-east-1:123456789:key/def-456",
}

@encrypt.json
async def encrypt_json_data(ctx: EncryptionContext, data: dict) -> dict:
    tenant_id = ctx.metadata.get("tenant_id")
    if not tenant_id:
        raise ValueError("tenant_id is required in encryption context")

    key_id = TENANT_KEYS.get(tenant_id)
    if not key_id:
        raise ValueError(f"No encryption key found for tenant: {tenant_id}")

    encrypted_data = {}
    for key, value in data.items():
        if key.startswith("my.customer.org/"):
            response = kms_client.encrypt(
                KeyId=key_id,
                Plaintext=str(value).encode(),
                EncryptionContext={'tenant_id': tenant_id, 'field': key}
            )
            import base64
            encrypted_data[key] = base64.b64encode(
                response['CiphertextBlob']
            ).decode()
        else:
            encrypted_data[key] = value

    return encrypted_data

@decrypt.json
async def decrypt_json_data(ctx: EncryptionContext, data: dict) -> dict:
    tenant_id = ctx.metadata.get("tenant_id")
    if not tenant_id:
        raise ValueError("tenant_id is required in encryption context")

    decrypted_data = {}
    for key, value in data.items():
        if key.startswith("my.customer.org/") and isinstance(value, str):
            import base64
            ciphertext = base64.b64decode(value.encode())
            response = kms_client.decrypt(
                CiphertextBlob=ciphertext,
                EncryptionContext={'tenant_id': tenant_id, 'field': key}
            )
            decrypted_data[key] = response['Plaintext'].decode()
        else:
            decrypted_data[key] = value

    return decrypted_data
```

## Related resources

- [Authentication & Access Control](../../concepts/auth.md) - Similar decorator-based system for custom auth
- [LangGraph Platform Concepts](../../concepts/langgraph_platform.md) - Overall platform architecture