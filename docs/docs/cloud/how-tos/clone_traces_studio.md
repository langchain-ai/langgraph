# Testing local agents with remote traces

## Overview

A common workflow when debugging production-deployed agents is to test the same thread against a local version of the same agent, which may have modifications.

To support this, LangGraph Studio, in combination with LangSmith, allows you to clone remote threads traced in LangSmith into your locally running agent. This cloned thread can then be used to re-run specific nodes within Studio.

## Requirements

!!! info "Prerequisites"

    - langgraph>=0.3.18
    - langgraph-api>=0.0.32

- A thread traced in LangSmith.
- A locally running agent. See [here](../../how-tos/local-studio.md) for setup instructions.
  - Note that your local agent must be using the above specified `langgraph` and `langgraph-api` versions.
  - The nodes present in the remote trace must exist in at least one of the graphs in your local agent.

## Cloning Thread

First navigate to the LangSmith trace. Here you should see a button to "Run in Studio".

![Run in Studio](../img/run_in_studio.png){width=1200}

This will prompt you to enter the url that your locally running agent is accessible at. Once provided, select "Clone thread locally". If you have multiple graphs in your agent, you will also be prompted to select a graph to clone this thread under.

Once selected, a will a new thread in your local agent will be created and the thread history will be reconstruced to reflect the original trace.

Alternatively, if your trace originates from an agent deployed on LangGraph Platform, you can "View original thread" to open Studio with the actual deployed thread.
