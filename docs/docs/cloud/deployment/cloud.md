# How to Deploy to Cloud SaaS

Before deploying, review the [conceptual guide for the Cloud SaaS](../../concepts/langgraph_cloud.md) deployment option.

## Prerequisites

1. LangGraph Platform applications are deployed from GitHub repositories. Configure and upload a LangGraph Platform application to a GitHub repository in order to deploy it to LangGraph Platform.
1. [Verify that the LangGraph API runs locally](../../tutorials/langgraph-platform/local-server.md). If the API does not run successfully (i.e. `langgraph dev`), deploying to LangGraph Platform will fail as well.

## Create New Deployment

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `LangGraph Platform`. The `LangGraph Platform` view contains a list of existing LangGraph Platform deployments.
1. In the top-right corner, select `+ New Deployment` to create a new deployment.
1. In the `Create New Deployment` panel, fill out the required fields.
    1. `Deployment details`
        1. Select `Import from GitHub` and follow the GitHub OAuth workflow to install and authorize LangChain's `hosted-langserve` GitHub app to access the selected repositories. After installation is complete, return to the `Create New Deployment` panel and select the GitHub repository to deploy from the dropdown menu. **Note**: The GitHub user installing LangChain's `hosted-langserve` GitHub app must be an [owner](https://docs.github.com/en/organizations/managing-peoples-access-to-your-organization-with-roles/roles-in-an-organization#organization-owners) of the organization or account.
        1. Specify a name for the deployment.
        1. Specify the desired `Git Branch`. A deployment is linked to a branch. When a new revision is created, code for the linked branch will be deployed. The branch can be updated later in the [Deployment Settings](#deployment-settings).
        1. Specify the full path to the [LangGraph API config file](../reference/cli.md#configuration-file) including the file name. For example, if the file `langgraph.json` is in the root of the repository, simply specify `langgraph.json`.
        1. Check/uncheck checkbox to `Automatically update deployment on push to branch`. If checked, the deployment will automatically be updated when changes are pushed to the specified `Git Branch`. This setting can be enabled/disabled later in the [Deployment Settings](#deployment-settings).
    1. Select the desired `Deployment Type`.
        1. `Development` deployments are meant for non-production use cases and are provisioned with minimal resources.
        1. `Production` deployments can serve up to 500 requests/second and are provisioned with highly available storage with automatic backups.
    1. Determine if the deployment should be `Shareable through LangGraph Studio`.
        1. If unchecked, the deployment will only be accessible with a valid LangSmith API key for the workspace.
        1. If checked, the deployment will be accessible through LangGraph Studio to any LangSmith user. A direct URL to LangGraph Studio for the deployment will be provided to share with other LangSmith users.
    1. Specify `Environment Variables` and secrets. See the [Environment Variables reference](../reference/env_var.md) to configure additional variables for the deployment.
        1. Sensitive values such as API keys (e.g. `OPENAI_API_KEY`) should be specified as secrets.
        1. Additional non-secret environment variables can be specified as well.
    1. A new LangSmith `Tracing Project` is automatically created with the same name as the deployment.
1. In the top-right corner, select `Submit`. After a few seconds, the `Deployment` view appears and the new deployment will be queued for provisioning.

## Create New Revision

When [creating a new deployment](#create-new-deployment), a new revision is created by default. Subsequent revisions can be created to deploy new code changes.

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `LangGraph Platform`. The `LangGraph Platform` view contains a list of existing LangGraph Platform deployments.
1. Select an existing deployment to create a new revision for.
1. In the `Deployment` view, in the top-right corner, select `+ New Revision`.
1. In the `New Revision` modal, fill out the required fields.
    1. Specify the full path to the [LangGraph API config file](../reference/cli.md#configuration-file) including the file name. For example, if the file `langgraph.json` is in the root of the repository, simply specify `langgraph.json`.
    1. Determine if the deployment should be `Shareable through LangGraph Studio`.
        1. If unchecked, the deployment will only be accessible with a valid LangSmith API key for the workspace.
        1. If checked, the deployment will be accessible through LangGraph Studio to any LangSmith user. A direct URL to LangGraph Studio for the deployment will be provided to share with other LangSmith users.
    1. Specify `Environment Variables` and secrets. Existing secrets and environment variables are prepopulated. See the [Environment Variables reference](../reference/env_var.md) to configure additional variables for the revision.
        1. Add new secrets or environment variables.
        1. Remove existing secrets or environment variables.
        1. Update the value of existing secrets or environment variables.
1. Select `Submit`. After a few seconds, the `New Revision` modal will close and the new revision will be queued for deployment.

## View Build and Server Logs

Build and server logs are available for each revision.

Starting from the `LangGraph Platform` view...

1. Select the desired revision from the `Revisions` table. A panel slides open from the right-hand side and the `Build` tab is selected by default, which displays build logs for the revision.
1. In the panel, select the `Server` tab to view server logs for the revision. Server logs are only available after a revision has been deployed.
1. Within the `Server` tab, adjust the date/time range picker as needed. By default, the date/time range picker is set to the `Last 7 days`.

## View Deployment Metrics

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `LangGraph Platform`. The `LangGraph Platform` view contains a list of existing LangGraph Platform deployments.
1. Select an existing deployment to monitor.
1. Select the `Monitoring` tab to view the deployment metrics. See a list of [all available metrics](../../concepts/langgraph_control_plane.md#monitoring).
1. Within the `Monitoring` tab, use the date/time range picker as needed. By default, the date/time range picker is set to the `Last 15 minutes`.

## Interrupt Revision

Interrupting a revision will stop deployment of the revision.

!!! warning "Undefined Behavior"
    Interrupted revisions have undefined behavior. This is only useful if you need to deploy a new revision and you already have a revision "stuck" in progress. In the future, this feature may be removed.

Starting from the `LangGraph Platform` view...

1. Select the menu icon (three dots) on the right-hand side of the row for the desired revision from the `Revisions` table.
1. Select `Interrupt` from the menu.
1. A modal will appear. Review the confirmation message. Select `Interrupt revision`.

## Delete Deployment

Starting from the <a href="https://smith.langchain.com/" target="_blank">LangSmith UI</a>...

1. In the left-hand navigation panel, select `LangGraph Platform`. The `LangGraph Platform` view contains a list of existing LangGraph Platform deployments.
1. Select the menu icon (three dots) on the right-hand side of the row for the desired deployment and select `Delete`.
1. A `Confirmation` modal will appear. Select `Delete`.

## Deployment Settings

Starting from the `LangGraph Platform` view...

1. In the top-right corner, select the gear icon (`Deployment Settings`).
1. Update the `Git Branch` to the desired branch.
1. Check/uncheck checkbox to `Automatically update deployment on push to branch`.
1. Branch creation/deletion and tag creation/deletion events will not trigger an update. Only pushes to an existing branch will trigger an update.
1. Pushes in quick succession to a branch will queue subsequent updates. Once a build completes, the most recent commit will begin building and the other queued builds will be skipped.

## Add or Remove GitHub Repositories

After installing and authorizing LangChain's `hosted-langserve` GitHub app, repository access for the app can be modified to add new repositories or remove existing repositories. If a new repository is created, it may need to be added explicitly.

1. From the GitHub profile, navigate to `Settings` > `Applications` > `hosted-langserve` > click `Configure`.
1. Under `Repository access`, select `All repositories` or `Only select repositories`. If `Only select repositories` is selected, new repositories must be explicitly added.
1. Click `Save`.
1. When creating a new deployment, the list of GitHub repositories in the dropdown menu will be updated to reflect the repository access changes.

## Whitelisting IP Addresses

All traffic from `LangGraph Platform` deployments created after January 6th 2025 will come through a NAT gateway.
This NAT gateway will have several static ip addresses depending on the region you are deploying in. Refer to the table below for the list of IP addresses to whitelist:

| US             | EU              |
|----------------|-----------------|
| 35.197.29.146  | 34.90.213.236   |
| 34.145.102.123 | 34.13.244.114   |
| 34.169.45.153  | 34.32.180.189   |
| 34.82.222.17   | 34.34.69.108    |
| 35.227.171.135 | 34.32.145.240   | 
| 34.169.88.30   | 34.90.157.44    |
| 34.19.93.202   | 34.141.242.180  |
| 34.19.34.50    | 34.32.141.108   |
