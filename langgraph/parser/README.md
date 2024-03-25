# LangGraph StateGraph Parser
This module parses python files which have a LangGraph StateGraph definition and
generates a PNG using a mermaid-cli docker container.

## Overview
While developing in LangGraph it can be helpful for the developer to visualize the 
flowchart representing the logic flow of their StateGraph.This parser reads a python
file and creates an abstract syntax tree (AST) of the StateGraph. The AST is then
translated into Mermaid charting language and saved as a PNG. 

## Prerequisites
- Docker must be running
- A python file containing a LangGraph workflow that defines a StateGraph

## Usage
1. cd into parser directory: `cd langgraph/parser`
2. run parser
    Syntax: `python3 {name_of_parser_file} {name_of_python_file_with_StateGraph}`

Example: `python3 graph_parser.py simple_agent.py`