# Manage assistants

!!! info "Prerequisites"

    - [Assistants Overview](../../../concepts/assistants.md)

LangGraph Studio lets you view, edit, and update your assistants, and allows you to run your graph using these assistant configurations.

## Graph mode

To view your assistants, click the "Manage Assistants" button in the bottom left corner.

This opens a modal for you to view all the assistants for the selected graph. Specify the assistant and its version you would like to mark as "Active", and this assistant will be used when submitting runs.

By default, the "Default configuration" option will be active. This option reflects the default configuration defined in your graph. Edits made to this configuration will be used to update the run-time configuration, but will not update or create a new assistant unless you click "Create new assistant".

## Chat mode

Chat mode enables you to switch through the different assistants in your graph via the dropdown selector at the top of the page. To create, edit, or delete assistants, use Graph mode.