# How to Deploy to LangGraph Cloud

LangGraph Cloud is available within <a href="https://www.langchain.com/langsmith" target="_blank">LangSmith</a>. To deploy a LangGraph Cloud API, navigate to the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>.

## Prerequisites

1. LangGraph Cloud applications are deployed from GitHub repositories. Configure and upload a LangGraph Cloud application to a GitHub repository in order to deploy it to LangGraph Cloud.
1. [Verify that the LangGraph API runs locally](test_locally.md). If the API does not build and run successfully (i.e. `langgraph up`), deploying to LangGraph Cloud will fail as well.

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

When [creating a new deployment](#create-new-deployment), a new revision is created by default. Subsequent revisions can be created to deploy new code changes.

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

## View Build and Deployment Logs

Build and deployment logs are available for each revision.

Starting from the `Deployment` view...

1. Select the desired revision from the `Revisions` table. A panel slides open from the right-hand side and the `Build` tab is selected by default, which displays build logs for the revision.
1. In the panel, select the `Deploy` tab to view deployment logs for the revision.
1. Within the `Deploy` tab, adjust the date/time range picker as needed. By default, the date/time range picker is set to the `Last 15 minutes`.

## Interrupt Revision

Interrupting a revision will stop deployment of the revision.

!!! warning "Undefined Behavior"
    Interrupted revisions have undefined behavior. This is only useful if you need to deploy a new revision and you already have a revision "stuck" in progress. In the future, this feature may be removed.

Starting from the `Deployment` view...

1. Select the menu icon (three dots) on the right-hand side of the row for the desired revision from the `Revisions` table.
1. Select `Interrupt` from the menu.
1. A modal will appear. Review the confirmation message. Select `Interrupt revision`.

## Delete Deployment

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `Deployments`. The `Deployments` view contains a list of existing LangGraph Cloud deployments.
1. Select the menu icon (three dots) on the right-hand side of the row for the desired deployment and select `Delete`.
1. A `Confirmation` modal will appear. Select `Delete`.
