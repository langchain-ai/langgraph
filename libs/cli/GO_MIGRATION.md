# Go Migration Plan

## Goal

Move the full `langgraph` CLI implementation to Go while preserving existing
Python distribution and invocation flows.

Users should continue to be able to run:

- `langgraph ...`
- `uv run langgraph ...`
- `uvx langgraph ...`

During phase 1, the Go path is gated behind a feature flag. The Python package
remains the public entrypoint and launcher.

## Non-Goals

This phase does not include:

- JS migration
- `langsmith-cli` integration
- user-facing command renames
- intentional CLI behavior changes
- a long-lived dual implementation

## Source Of Truth

There will be one implementation of CLI behavior:

- shared Go implementation lives in the `langgraph` repo
- standalone Go `langgraph` binary uses that implementation
- Python `langgraph-cli` package is a thin launcher around that binary
- legacy Python implementation exists only temporarily as fallback during rollout

## Phase 1 Artifacts

Phase 1 ships these artifacts:

- shared Go package(s) in `langgraph`
- standalone `langgraph` Go binary
- Python wheel `langgraph-cli` that bundles the platform-specific Go binary
- Python launcher entrypoint that can route to legacy Python or Go

JS is explicitly out of scope for phase 1.

## User-Facing Command Scope

Phase 1 scope is the whole `langgraph` CLI, not just deploy.

Target command coverage:

- `langgraph deploy ...`
- `langgraph build ...`
- `langgraph up ...`
- `langgraph dockerfile ...`
- `langgraph dev ...`
- `langgraph new ...`

The goal is full parity with the current Python CLI command surface.

## Compatibility Contract

Behavior must not regress.

Required parity:

- exact JSON output where commands emit JSON
- exact or equivalent error semantics
- same exit codes
- same argument and flag behavior
- same generated artifacts for:
  - Dockerfile output
  - docker compose / inline compose output
- same API request semantics where mocked in tests

Human-readable output should be same or better, but not worse.

## Repo Ownership

Shared implementation lives in the current `langgraph` repo.

Reasons:

- current CLI spec and tests already live here
- rollout is initially only for `langgraph-cli`
- command compatibility should be driven by existing behavior in this repo

## Architecture

Use process boundaries, not language FFI.

Python should not call a Go shared library directly. Instead:

- Python launcher locates bundled `langgraph` Go binary
- Python launcher `exec`s or subprocesses into the Go binary
- Go handles all command execution
- for `dev`, Go subprocesses back into Python

This keeps the boundary simple and cross-platform.

## Go Package Structure

Recommended structure:

- `pkg/cli/config`
  - parse and validate `langgraph.json`
  - normalize config model
- `pkg/cli/docker`
  - docker capability detection
  - compose generation
  - Dockerfile/build plan generation
- `pkg/cli/deploy`
  - deployment flows
  - host backend client
  - polling, logs, revision logic
- `pkg/cli/dev`
  - `dev` command orchestration
  - Python subprocess handoff
- `pkg/cli/cmds`
  - command runner functions with typed options/results
  - no Cobra-specific code here
- `cmd/langgraph`
  - standalone Go binary wrapping shared packages

Business logic should live in shared packages, not directly in CLI adapter code.

## Python Wrapper Model

The Python package remains installed as `langgraph-cli`, with entrypoint
`langgraph`.

During migration, the wrapper decides whether to route to legacy Python or Go.

Wrapper behavior:

1. inspect feature flags
2. resolve Go binary path
3. if Go path is active, `exec` into Go binary
4. otherwise fall back to legacy Python implementation

Long-term target:

- remove fallback
- Python wrapper always launches bundled Go binary

## Feature Flags

Temporary rollout env vars:

- `LANGGRAPH_USE_GO_CLI=1`
  - route the Python wrapper to the Go binary instead of legacy Python
- `LANGGRAPH_GO_CLI_PATH=/path/to/langgraph`
  - internal/dev/CI override for binary path resolution
  - not intended as a long-term public interface
- `LANGGRAPH_CALLING_PYTHON=/path/to/python`
  - set by the Python wrapper before invoking Go
  - used by Go for `dev`

`LANGGRAPH_GO_CLI_PATH` is mainly for local development and CI and can be
removed later.

## `dev` Invocation Contract

`dev` is the main tricky area.

Design rule:

- Go owns CLI parsing and routing
- Python owns the actual in-process local dev server runtime

Flow for `uv run langgraph dev`:

1. `uv` selects the Python interpreter/environment
2. Python wrapper starts
3. Python wrapper sets `LANGGRAPH_CALLING_PYTHON=sys.executable`
4. Python wrapper launches Go binary
5. Go receives `dev`
6. Go shells out to that exact Python interpreter for the actual Python runtime behavior

This preserves the current selected Python environment.

Go Python resolution order for `dev`:

1. `LANGGRAPH_CALLING_PYTHON`
2. optional explicit override if added later
3. environment-derived interpreter / active venv
4. fallback detection
5. clear failure

The critical constraint is: if the user entered through Python, `dev` should
use that exact Python when possible.

## Why Not FFI

Do not use:

- cgo shared libs
- Python-Go FFI bindings
- embedded Python in Go
- RPC unless absolutely necessary

Reasons:

- packaging complexity
- cross-platform pain
- no advantage for a CLI architecture
- much worse release/debug story

Process-level boundaries are the right choice here.

## Packaging Constraints

The Go binary should be bundled inside Python wheels.

Preferred distribution model:

- build platform-specific `langgraph-cli` wheels
- each wheel includes the matching `langgraph` Go binary
- Python launcher resolves and executes the bundled binary

Do not rely on runtime download of the binary for normal operation.

Support matrix target:

- all OS/arch targets that are currently expected to be supported
- at minimum, align with the practical support matrix desired for the CLI,
  using `orjson` support as a rough proxy if needed

If a platform is unsupported, fail clearly rather than silently falling back
forever.

## Release Constraints

Phase 1 versioning applies to:

- shared Go implementation
- standalone Go `langgraph` binary
- PyPI `langgraph-cli` wrapper

They should stay on one version line.

Constraint:

- bundled Go binary version must exactly match the Python wrapper version for
  the migrated surface

The wrapper should detect obvious mismatch and fail clearly if it occurs.

## Migration Strategy

Use a big-bang hidden implementation change with gradual activation.

Phase 1 rollout:

1. implement full Go path behind `LANGGRAPH_USE_GO_CLI`
2. keep default behavior on legacy Python
3. run dual CI for legacy and Go-backed paths
4. dogfood with feature flag
5. flip default to Go
6. keep fallback briefly
7. remove fallback in about two weeks

This is a big internal rewrite with gradual external activation.

## CI Strategy

Dual CI is required during migration.

Run both variants:

- legacy Python implementation
- Python wrapper -> Go binary implementation

Required parity checks:

- help output
- exit code
- stdout
- stderr
- generated Dockerfile output
- generated compose output
- mocked deployment API request semantics
- validation errors / usage errors

Goal is not merely "both tests pass". Goal is "both implementations behave
identically enough to swap by default safely".

## Parity Test Philosophy

Use the current Python CLI tests as the behavioral spec.

Priority test areas:

- config validation
- compose/Dockerfile generation
- deployment flows
- error and prompt behavior
- command help / command surface

Where practical, add golden comparisons so regressions are obvious.

## Implementation Order Inside Phase 1

Even though rollout is one hidden phase, implementation should proceed in this
order:

1. wrapper contract and env contract
2. Go command scaffolding and package boundaries
3. config + docker/build/compose logic
4. deploy flows
5. remaining commands
6. `dev` subprocess orchestration
7. parity hardening in CI

This reduces risk because `dev` is the highest-uncertainty area.

## Command Ownership Constraint

All command behavior should live in Go once ported.

Do not allow:

- some flags parsed in Python and others in Go
- duplicated command logic across Python and Go
- separate behavior definitions for legacy and migrated commands

The wrapper should be thin only.

## Fallback Constraint

Fallback is temporary, not a product feature.

Policy:

- use feature flag during migration
- flip default after parity confidence
- remove legacy Python implementation roughly two weeks later

Do not normalize to permanent dual execution paths.

## Documentation Constraint

During migration, documentation should stay conservative:

- existing Python install flow remains primary
- feature flag is acceptable for internal/dogfood docs
- avoid broad external messaging about the Go implementation until default is flipped

## Open Issues To Track

These are not blockers, but they need explicit implementation decisions:

- exact bundled wheel layout for binaries
- exact list of supported OS/arch targets
- whether to expose a public `--python` override for `dev`
- whether some pretty output is allowed to improve while keeping parsed output stable

## Summary

Phase 1 plan:

- move the entire `langgraph` CLI implementation into shared Go code in the
  `langgraph` repo
- ship a standalone `langgraph` Go binary
- keep `langgraph-cli` on PyPI as a thin launcher that bundles and executes
  that binary
- preserve `uv run` / `uvx` behavior
- handle `dev` by passing the calling Python path through the wrapper and
  having Go subprocess back into Python
- gate everything behind `LANGGRAPH_USE_GO_CLI`
- run dual CI until parity is proven
- flip default
- remove legacy fallback quickly
