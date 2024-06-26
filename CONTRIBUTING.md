# General guidelines

Here are some things to keep in mind for all types of contributions:

- Follow the ["fork and pull request"](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) workflow.
- Fill out the checked-in pull request template when opening pull requests. Note related issues and tag relevant maintainers.
- Ensure your PR passes formatting, linting, and testing checks before requesting a review.
  - If you would like comments or feedback on your current progress, please open an issue or discussion and tag a maintainer.
  - See the sections on [Testing](/docs/contributing/code/setup#testing) and [Formatting and Linting](/docs/contributing/code/setup#formatting-and-linting) for how to run these checks locally.
- Backwards compatibility is key. Your changes must not be breaking, except in case of critical bug and security fixes.
- Look for duplicate PRs or issues that have already been opened before opening a new one.
- Keep scope as isolated as possible. As a general rule, your changes should not affect more than one package at a time.

# Contribute Documentation

Documentation is a vital part of LangGraph. We welcome both new documentation for new features and
community improvements to our current documentation. Please read the resources below before getting started:

- [Documentation style guide](#documentation-style-guide)
- [Documentation Setup](/docs/contributing/documentation/setup/)

# Documentation Style Guide

As LangGraph continues to grow, the surface area of documentation required to cover it continues to grow too.
This page provides guidelines for anyone writing documentation for LangGraph, as well as some of our philosophies around organization and structure.

## Philosophy

LangGraph's documentation follows the [Diataxis framework](https://diataxis.fr).
Under this framework, all documentation falls under one of four categories: [Tutorials](#tutorials),
[How-to guides](#how-to-guides),
[References](#references), and [Explanations](#conceptual-guide).

### Tutorials

Tutorials are lessons that take the reader through a practical activity. Their purpose is to help the user
gain understanding of concepts and how they interact by showing one way to achieve some goal in a hands-on way. They should not cover
multiple permutations of ways to achieve that goal in-depth, and the end result of a tutorial does not need to
be completely production-ready against all cases. Information on how to address additional scenarios
can occur in how-to guides.

To quote the Diataxis website:

> A tutorial serves the user’s *acquisition* of skills and knowledge - their study. Its purpose is not to help the user get something done, but to help them learn.

In LangGraph, these are often higher level guides that show off end-to-end use cases.

Some examples include:

- [Build a Customer Support Bot](https://langchain-ai.github.io/langgraph/tutorials/customer-support/customer-support/)
- [Build a SQL Agent](https://langchain-ai.github.io/langgraph/tutorials/sql-agent/)

Here are some high-level tips on writing a good tutorial:

- Focus on guiding the user to get something done, but keep in mind the end-goal is more to impart principles than to create a perfect production system
- Be specific, not abstract and follow one path
  - No need to go deeply into alternative approaches, but it’s ok to reference them, ideally with a link to an appropriate how-to guide
- Get "a point on the board" as soon as possible - something the user can run that outputs something
  - You can iterate and expand afterwards
  - Try to frequently checkpoint at given steps where the user can run code and see progress
- Focus on results, not technical explanation
  - Crosslink heavily to appropriate conceptual/reference pages
- The first time you mention a LangGraph concept, use its full name (e.g. "human-in-the-loop"), and link to its conceptual/other documentation page
  - It's also helpful to add a prerequisite callout that links to any pages with necessary background information
- End with a recap/next steps section summarizing what the tutorial covered and future reading, such as related how-to guides

### How-to guides

A how-to guide, as the name implies, demonstrates how to do something discrete and specific.
It should assume that the user is already familiar with underlying concepts, and is trying to solve an immediate problem, but
should still give some background or list the scenarios where the information contained within can be relevant.
They can and should discuss alternatives if one approach may be better than another in certain cases.

To quote the Diataxis website:

> A how-to guide serves the work of the already-competent user, whom you can assume to know what they want to do, and to be able to follow your instructions correctly.

Some examples include:

- [How to add persistence to your graph](https://langchain-ai.github.io/langgraph/how-tos/persistence/)
- [How to view and update past graph state](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/time-travel/)

Here are some high-level tips on writing a good how-to guide:

- Clearly explain what you are guiding the user through at the start
- Assume higher intent than a tutorial and show what the user needs to do to get that task done
- Assume familiarity of concepts, but explain why suggested actions are helpful
  - Crosslink heavily to conceptual/reference pages
- Discuss alternatives and responses to real-world tradeoffs that may arise when solving a problem
- Use lots of example code
- End with a recap/next steps section summarizing what the tutorial covered and future reading, such as other related how-to guides

### Conceptual guide

LangGraph's conceptual guide falls under the **Explanation** quadrant of Diataxis. They should cover LangChain terms and concepts
in a more abstract way than how-to guides or tutorials, and should be geared towards curious users interested in
gaining a deeper understanding of the framework. There should be few, if any, concrete code examples. The goal here is to
impart perspective to the user rather than to finish a practical project.

This guide on documentation style is meant to fall under this category.

To quote the Diataxis website:

> The perspective of explanation is higher and wider than that of the other types. It does not take the user’s eye-level view, as in a how-to guide, or a close-up view of the machinery, like reference material. Its scope in each case is a topic - “an area of knowledge”, that somehow has to be bounded in a reasonable, meaningful way.

Some examples include:

- [What does it mean to be agentic?](https://langchain-ai.github.io/langgraph/concepts/high_level/)
- [Tool calling](https://langchain-ai.github.io/langgraph/concepts/agentic_concepts/#tool-calling)

Here are some high-level tips on writing a good conceptual guide:

- Explain design decisions. Why does concept X exist and why was it designed this way?
- Use analogies and reference other concepts and alternatives
- Avoid blending in too much reference content
- You can and should reference content covered in other guides, but make sure to link to them

### References

References contain detailed, low-level information that describes exactly what functionality exists and how to use it.
In LangGraph, this is mainly our API reference pages, which are populated from docstrings within code.
References pages are generally not read end-to-end, but are consulted as necessary when a user needs to know
how to use something specific.

To quote the Diataxis website:

> The only purpose of a reference guide is to describe, as succinctly as possible, and in an orderly way. Whereas the content of tutorials and how-to guides are led by needs of the user, reference material is led by the product it describes.

Many of the reference pages in LangChain are automatically generated from code,
but here are some high-level tips on writing a good docstring:

- Be concise
- Discuss special cases and deviations from a user's expectations
- Go into detail on required inputs and outputs
- Light details on when one might use the feature are fine, but in-depth details belong in other sections.

Each category serves a distinct purpose and requires a specific approach to writing and structuring the content.

## General guidelines

Here are some other guidelines you should think about when writing and organizing documentation.

We generally do not merge new tutorials from outside contributors without an actue need.
We welcome updates as well as new integration docs, how-tos, and references.

### Avoid duplication

Multiple pages that cover the same material in depth are difficult to maintain and cause confusion. There should
be only one (very rarely two), canonical pages for a given concept or feature. Instead, you should link to other guides.

### Link to other sections

Because sections of the docs do not exist in a vacuum, it is important to link to other sections as often as possible
to allow a developer to learn more about an unfamiliar topic inline.

This includes linking to the API references as well as conceptual sections!

### Be concise

In general, take a less-is-more approach. If a section with a good explanation of a concept already exists, you should link to it rather than
re-explain it, unless the concept you are documenting presents some new wrinkle.

Be concise, including in code samples.

### General style

- Use active voice and present tense whenever possible
- Use examples and code snippets to illustrate concepts and usage
- Use appropriate header levels (`#`, `##`, `###`, etc.) to organize the content hierarchically
- Use fewer cells with more code to make copy/paste easier
- Use bullet points and numbered lists to break down information into easily digestible chunks
- Use tables (especially for **Reference** sections) and diagrams often to present information visually
- Include the table of contents for longer documentation pages to help readers navigate the content, but hide it for shorter pages

# Setup

LangChain documentation consists of two components:

1. Main Documentation: Hosted at [https://langchain-ai.github.io](https://langchain-ai.github.io/langgraph/),
this comprehensive resource serves as the primary user-facing documentation.
It covers a wide array of topics, including tutorials, use cases, integrations,
and more, offering extensive guidance on building with LangGraph.
The content for this documentation lives in the `/docs` directory of the monorepo.
2. In-code Documentation: This is documentation of the codebase itself, which is also
used to generate the externally facing [API Reference](https://langchain-ai.github.io/langgraph/reference/graphs/).
The content for the API reference is autogenerated by scanning the docstrings in the codebase. For this reason we ask that developers document their code well.

We appreciate all contributions to the documentation, whether it be fixing a typo,
adding a new tutorial or example and whether it be in the main documentation or the API Reference.

## 📜 Main Documentation

The content for the main documentation is located in the `/docs` directory of the monorepo.

The documentation is written using a combination of ipython notebooks (`.ipynb` files)
and markdown (`.md` files). The notebooks are converted to markdown
and then built using [MkDocs](https://www.mkdocs.org/).

Feel free to make contributions to the main documentation! 🥰

After modifying the documentation:

1. Run the linting and formatting commands (see below) to ensure that the documentation is well-formatted and free of errors.
2. Optionally build the documentation locally to verify that the changes look good.
3. Make a pull request with the changes.
4. You can preview and verify that the changes are what you wanted by clicking the `View deployment` or `Visit Preview` buttons on the pull request `Conversation` page. This will take you to a preview of the documentation changes.

## ⚒️ Linting and Building Documentation Locally

After writing up the documentation, you may want to lint and build the documentation
locally to ensure that it looks good and is free of errors.

If you're unable to build it locally that's okay as well, as you will be able to
see a preview of the documentation on the pull request page.

From the **monorepo root**, run the following command to install the dependencies:

```bash
poetry install --with docs --no-root
````

### Building

The code that builds the documentation is located in the `/docs` directory of the monorepo.

Before building the documentation, it is always a good idea to clean the build directory:

```bash
make clean-docs
```

You can build and preview the documentation as outlined below:

```bash
make serve-docs
```

### Linting

The Main Documentation is linted from the **monorepo root**. To lint the main documentation, run the following from there:

```bash
make spellcheck
```

## ️In-code Documentation

The in-code documentation is autogenerated from docstrings.

For the API reference to be useful, the codebase must be well-documented. This means that all functions, classes, and methods should have a docstring that explains what they do, what the arguments are, and what the return value is. This is a good practice in general, but it is especially important for LangChain because the API reference is the primary resource for developers to understand how to use the codebase.

We generally follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for docstrings.

Here is an example of a well-documented function:

```python

def my_function(arg1: int, arg2: str) -> float:
    """This is a short description of the function. (It should be a single sentence.)

    This is a longer description of the function. It should explain what
    the function does, what the arguments are, and what the return value is.
    It should wrap at 88 characters.

    Examples:
        This is a section for examples of how to use the function.

        .. code-block:: python

            my_function(1, "hello")

    Args:
        arg1: This is a description of arg1. We do not need to specify the type since
            it is already specified in the function signature.
        arg2: This is a description of arg2.

    Returns:
        This is a description of the return value.
    """
    return 3.14
```