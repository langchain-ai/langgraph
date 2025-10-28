# Azure OpenAI Setup Guide

Quick reference for setting up Azure OpenAI with the tutorial.

## 📋 Required Environment Variables

```powershell
# PowerShell (Windows)
$env:AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY = "your-key-from-azure-portal"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"  # Or your deployment name
```

```bash
# Bash (Linux/Mac)
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-key-from-azure-portal"
export AZURE_OPENAI_DEPLOYMENT="gpt-4o-mini"
```

## 🔍 Finding Your Credentials

1. **Go to Azure Portal**: <https://portal.azure.com>
2. **Navigate to your OpenAI resource**
3. **Click "Keys and Endpoint"** in the left sidebar
4. **Copy the values:**
   - **Endpoint**: `https://your-resource.openai.azure.com/`
   - **Key**: Copy "KEY 1" or "KEY 2"
5. **Get deployment name:**
   - Click "Model deployments" (or go to Azure OpenAI Studio)
   - Copy your deployment name (e.g., `gpt-4o-mini`, `gpt-35-turbo`)

## 🎯 Example Configuration

If your resource is named `mycompany-openai` in `eastus` region:

```powershell
$env:AZURE_OPENAI_ENDPOINT = "https://mycompany-openai.openai.azure.com/"
$env:AZURE_OPENAI_API_KEY = "abc123...xyz789"
$env:AZURE_OPENAI_DEPLOYMENT = "gpt-4o-mini"
```

## ✅ Verify Setup

Test your configuration:

```powershell
# PowerShell
Write-Host "Endpoint: $env:AZURE_OPENAI_ENDPOINT"
Write-Host "Key set: $($env:AZURE_OPENAI_API_KEY -ne $null)"
Write-Host "Deployment: $env:AZURE_OPENAI_DEPLOYMENT"
```

```bash
# Bash
echo "Endpoint: $AZURE_OPENAI_ENDPOINT"
echo "Key set: $([ -n "$AZURE_OPENAI_API_KEY" ] && echo 'Yes' || echo 'No')"
echo "Deployment: $AZURE_OPENAI_DEPLOYMENT"
```

## 🚀 Run Tutorial

```bash
cd examples
python tutorial_04_openai_chatbot.py
```

The tutorial will automatically detect Azure OpenAI configuration and use `AzureChatOpenAI` instead of regular `ChatOpenAI`.

## 🔄 Switch Between OpenAI and Azure OpenAI

The tutorial checks for `AZURE_OPENAI_ENDPOINT`:

- **If set** → Uses Azure OpenAI
- **If not set** → Uses regular OpenAI (requires `OPENAI_API_KEY`)

To switch back to regular OpenAI:

```powershell
# Clear Azure OpenAI variables
Remove-Item Env:AZURE_OPENAI_ENDPOINT
Remove-Item Env:AZURE_OPENAI_API_KEY
Remove-Item Env:AZURE_OPENAI_DEPLOYMENT

# Set OpenAI key
$env:OPENAI_API_KEY = "sk-proj-..."
```

## 💡 Common Issues

### "Authentication failed"

- **Check your API key** is correct
- **Verify endpoint URL** ends with `/`
- **Ensure resource is active** in Azure Portal

### "Deployment not found"

- **Check deployment name** matches exactly (case-sensitive)
- **Verify deployment is active** in Azure OpenAI Studio
- **Check model is deployed** to your resource

### "Resource not found"

- **Verify endpoint URL** is correct
- **Check resource region** matches your subscription
- **Ensure resource exists** in Azure Portal

## 📊 Available Models in Azure OpenAI

Common deployment names (yours may differ):

- `gpt-4o` - Latest GPT-4o model
- `gpt-4o-mini` - Efficient, cost-effective
- `gpt-35-turbo` - GPT-3.5 Turbo (legacy naming)
- `gpt-4-turbo` - GPT-4 Turbo
- `gpt-4` - Standard GPT-4

Check your deployments in: Azure OpenAI Studio → Deployments

## 🔗 Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/azure/ai-services/openai/)
- [LangChain Azure OpenAI](https://python.langchain.com/docs/integrations/chat/azure_chat_openai)
- [Azure OpenAI Studio](https://oai.azure.com/)
