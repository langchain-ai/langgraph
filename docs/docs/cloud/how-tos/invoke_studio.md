# Run application

!!!info  "Prerequisites"
    - [Running agents](../../agents/run_agents.md#running-agents)

This guide shows how to submit a [run](../../concepts/assistants.md#execution) to your application.

## Graph mode

### Specify input
First define the input to your graph with in the "Input" section on the left side of the page, below the graph interface.

Studio will attempt to render a form for your input based on the graph's defined [state schema](../../concepts/low_level.md/#schema). To disable this, click the "View Raw" button, which will present you with a JSON editor.

Click the up/down arrows at the top of the "Input" section to toggle through and use previously submitted inputs.

### Run settings

#### Assistant

To specify the [assistant](../../concepts/assistants.md) that is used for the run click the settings button in the bottom left corner. If an assistant is currently selected the button will also list the assistant name. If no assistant is selected it will say "Manage Assistants".

Select the assistant to run and click the "Active" toggle at the top of the modal to activate it. [See here](./studio/manage_assistants.md) for more information on managing assistants.

#### Streaming
Click the dropdown next to "Submit" and click the toggle to enable/disable streaming.

#### Breakpoints
To run your graph with breakpoints, click the "Interrupt" button. Select a node and whether to pause before and/or after that node has executed. Click "Continue" in the thread log to resume execution.


For more information on breakpoints see [here](../../concepts/human_in_the_loop.md).

### Submit run

To submit the run with the specified input and run settings, click the "Submit" button. This will add a [run](../../concepts/assistants.md#execution) to the existing selected [thread](../../concepts/persistence.md#threads). If no thread is currently selected, a new one will be created.

To cancel the ongoing run, click the "Cancel" button.


## Chat mode
Specify the input to your chat application in the bottom of the conversation panel. Click the "Send message" button to submit the input as a Human message and have the response streamed back.

To cancel the ongoing run, click the "Cancel" button. Click the "Show tool calls" toggle to hide/show tool calls in the conversation.

## Learn more

To run your application from a specific checkpoint in an existing thread, see [this guide](./threads_studio.md#edit-thread-history).