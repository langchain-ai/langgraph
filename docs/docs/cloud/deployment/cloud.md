# How to Deploy to LangGraph Cloud

LangGraph Cloud is available within <a href="https://www.langchain.com/langsmith" target="_blank">LangSmith</a>. To deploy a LangGraph Cloud API, navigate to the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>.

## Setup GitHub Repository

LangGraph Cloud applications are deployed from GitHub repositories. Configure and upload a LangGraph Cloud application to a GitHub repository in order to deploy it to LangGraph Cloud.

## Create New Deployment

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `Deployments`. The `Deployments` view contains a list of existing LangGraph Cloud deployments.
1. In the top-right corner, select `+ New Deployment` to create a new deployment.
1. In the `Create New Deployment` panel, fill out the required fields.
    1. `Deployment details`
        1. Select `Import from GitHub` and follow the GitHub OAuth workflow to install and authorize LangChain's `hosted-langserve` GitHub app to  access the selected repositories. After installation is complete, return to the `Create New Deployment` panel and select the GitHub repository to deploy from the dropdown menu.
        1. Specify a name for the deployment.
        1. Specify the full path to the [LangGraph API config file](../reference/cli.md#configuration-file) including the file name. For example, if the file `langgraph.json` is in the root of the repository, simply specify `langgraph.json`.
        1. Specify the desired `git` reference (e.g. branch name). For example, different branches of the repository can be deployed.
    1. Select the desired `Deployment Type`.
        1. `Development` deployments are meant for non-production use cases and are provisioned with minimal resources.
        1. `Production` deployments can serve up to 500 requests/second and are provisioned with highly available storage with automatic backups.
    1. Specify `Environment Variables` and secrets. See the [Environment Variables reference](../reference/env_var.md) to configure additional variables for the deployment.
        1. Sensitive values such as API keys (e.g. `OPENAI_API_KEY`) should be specified as secrets.
        1. Additional non-secret environment variables can be specified as well.
    1. A new LangSmith `Tracing Project` is automatically created with the same name as the deployment.
1. In the top-right corner, select `Submit`. After a few seconds, the `Deployment` view appears and the new deployment will be queued for provisioning.

## Create New Revision

When [creating a new deployment](#create-a-new-deployment), a new revision is created by default. Subsequent revisions can be created to deploy new code changes.

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `Deployments`. The `Deployments` view contains a list of existing LangGraph Cloud deployments.
1. Select an existing deployment to create a new revision for.
1. In the `Deployment` view, in the top-right corner, select `+ New Revision`.
1. In the `New Revision` modal, fill out the required fields.
    1. Specify the full path to the [LangGraph API config file](../reference/cli.md#configuration-file) including the file name. For example, if the file `langgraph.json` is in the root of the repository, simply specify `langgraph.json`.
    1. Specify the desired `git` reference (e.g. branch name). For example, different branches of the repository can be deployed.
    1. Specify `Environment Variables` and secrets. Existing secrets and environment variables are prepopulated. See the [Environment Variables reference](../reference/env_var.md) to configure additional variables for the revision.
        1. Add new secrets or environment variables.
        1. Remove existing secrets or environment variables.
        1. Update the value of existing secrets or environment variables.
1. Select `Submit`. After a few seconds, the `New Revision` modal will close and the new revision will be queued for deployment.

## Asynchronous Deployment

New [deployments](#create-new-deployment) and [revisions](#create-new-revision) are provisioned and deployed asynchronously. They are not deployed immediately after submission. Currently, deployment can take up to several minutes.

The `Deployment` view continually updates the status of pending revisions.
