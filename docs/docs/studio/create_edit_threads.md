# How to create and edit threads

## Create a thread

When you open LangGraph Studio you will automatically be in a new thread window. If you have an existing thread open, follow these steps to create a new thread:

1. In the top-right corner of the right-hand pane, press `+` to open a new thread menu.
1. Choose between `Empty thread` or `Clone thread`. If you choose `Clone thread`, the state from the currently selected (existing) thread will be copied into a new thread. The original and copied thread are completely independent.

The following video shows how to create a thread:

<video controls="true" allowfullscreen="true" poster="../img/graph_video_poster.png">
    <source src="../img/create_thread.mp4" type="video/mp4">
</video>

## Select a thread

To select a thread:

1. Click on `New Thread` / `Thread <thread-id>` label at the top of the right-hand pane to open a thread list dropdown.
1. Select a thread that you wish to view / edit.

The following video shows how to select a thread:

<video controls="true" allowfullscreen="true" poster="../img/graph_video_poster.png">
    <source src="../img/select_thread.mp4" type="video/mp4">
</video>

## Edit thread state

LangGraph Studio allows you to edit the thread state and fork to create alternative graph execution with the updated state. To do it:

1. Select a thread you wish to edit.
1. In the right-hand pane hover over the step you wish to edit and click on "pencil" icon to edit.
1. Make your edits.
1. Click `Fork` to update the state and create a new graph execution with the updated state.

The following video shows how to edit a thread in the studio:

<video controls allowfullscreen="true" poster="../img/graph_video_poster.png">
    <source src="../img/fork_thread.mp4" type="video/mp4">
</video>
