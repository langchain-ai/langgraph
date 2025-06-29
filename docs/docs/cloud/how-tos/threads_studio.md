# Manage threads

Studio allows you to view [threads](../../concepts/persistence.md#threads) from the server and edit their state.

## View threads

### Graph mode

1. In the top of the right-hand pane, select the dropdown menu to view existing threads.
1. Select the desired thread, and the thread history will populate in the right-hand side of the page.
1. To create a new thread, click `+ New Thread` and [submit a run](../how-tos/invoke_studio.md#graph-mode).

To view more granular information in the thread, drag the slider at the top of the page to the right. To view less information, drag the slider to the left. Additionally, collapse or expand individual turns, nodes, and keys of the state.

Switch between `Pretty` and `JSON` mode for different rendering formats.

### Chat mode

1. View all threads in the right-hand pane of the page.
2. Select the desired thread and the thread history will populate in the center panel.
3. To create a new thread, click the plus button and [submit a run](../how-tos/invoke_studio.md#chat-mode).

## Edit thread history

### Graph mode

To edit the state of the thread, select "edit node state" next to the desired node. Edit the node's output as desired and click "fork" to confirm. This will create a new forked run from the checkpoint of the selected node.

If you instead want to re-run the thread from a given checkpoint without editing the state, click the "Re-run from here". This will again create a new forked run from the selected checkpoint. This is useful for re-running with changes that are not specific to the state, such as the selected assistant.

### Chat mode

To edit a human message in the thread, click the edit button below the human message. Edit the message as desired and submit. This will create a new fork of the conversation history. To re-generate an AI message, click the retry icon below the AI message.

## Learn more

For more information about time travel, [see here](../../concepts/time-travel.md).
