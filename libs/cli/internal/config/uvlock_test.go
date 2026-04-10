package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Helper: write a uv-lock workspace fixture
// ---------------------------------------------------------------------------

type workspaceOpts struct {
	// Relative directory (within project_root) that holds langgraph.json.
	// Default: "deploy/agent"
	configRelativeDir string
	// Extra TOML appended to the root pyproject.toml (e.g. [tool.uv.sources]).
	rootSources string
	// Extra TOML appended to the agent pyproject.toml (e.g. [tool.uv.sources]).
	agentSources string
	// Extra TOML appended to the shared lib pyproject.toml (e.g. [tool.uv] section).
	sharedUvConfig string
	// Custom agent dependencies list.  When nil the default is used.
	agentDependencies []string
	// Additional files to create: relative-path (from project_root) -> content.
	extraFiles map[string]string
}

// writeUvLockWorkspace creates the standard multi-package uv workspace used
// by the Python test_config_to_docker_uv_lock* test suite and returns
// (projectRoot, configPath).
//
// Layout:
//
//	workspace/
//	  pyproject.toml          [project] name="workspace-root" + [tool.uv.workspace]
//	  uv.lock
//	  apps/agent/             package "agent"
//	    pyproject.toml
//	    src/agent/graph.py
//	  libs/shared/            package "shared"
//	    pyproject.toml
//	    src/shared/auth.py
//	  libs/extra/             package "extra" (not a dep of agent)
//	    pyproject.toml
//	    src/extra/graph.py
//	  deploy/agent/           config directory (configRelativeDir)
//	    langgraph.json
func writeUvLockWorkspace(t *testing.T, opts workspaceOpts) (string, string) {
	t.Helper()
	base := t.TempDir()
	projectRoot := filepath.Join(base, "workspace")

	configRelDir := opts.configRelativeDir
	if configRelDir == "" {
		configRelDir = "deploy/agent"
	}

	configDir := filepath.Join(projectRoot, configRelDir)
	sharedDir := filepath.Join(projectRoot, "libs", "shared")
	extraDir := filepath.Join(projectRoot, "libs", "extra")
	deployDir := filepath.Join(projectRoot, "deploy", "agent")

	for _, d := range []string{configDir, sharedDir, extraDir, deployDir} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatalf("MkdirAll(%q): %v", d, err)
		}
	}

	// -- root pyproject.toml -------------------------------------------------
	rootSources := opts.rootSources
	writeFile(t, filepath.Join(projectRoot, "pyproject.toml"),
		"[project]\n"+
			"name = \"workspace-root\"\n"+
			"version = \"0.1.0\"\n"+
			"\n"+
			"[tool.uv.workspace]\n"+
			"members = [\"apps/*\", \"libs/*\"]\n"+
			"\n"+
			rootSources+"\n"+
			"\n"+
			"[build-system]\n"+
			"requires = [\"setuptools>=61\"]\n"+
			"build-backend = \"setuptools.build_meta\"\n")

	// -- uv.lock -------------------------------------------------------------
	writeFile(t, filepath.Join(projectRoot, "uv.lock"), "# uv lock file\n")

	// -- agent pyproject.toml ------------------------------------------------
	agentDir := filepath.Join(projectRoot, "apps", "agent")
	if err := os.MkdirAll(agentDir, 0o755); err != nil {
		t.Fatalf("MkdirAll: %v", err)
	}

	agentDeps := opts.agentDependencies
	if agentDeps == nil {
		agentDeps = []string{"shared", "httpx>=0.28"}
	}
	depsList := "[\"" + strings.Join(agentDeps, "\", \"") + "\"]"

	agentSources := opts.agentSources
	writeFile(t, filepath.Join(agentDir, "pyproject.toml"),
		"[project]\n"+
			"name = \"agent\"\n"+
			"version = \"0.1.0\"\n"+
			"dependencies = "+depsList+"\n"+
			"\n"+
			agentSources+"\n"+
			"\n"+
			"[build-system]\n"+
			"requires = [\"setuptools>=61\"]\n"+
			"build-backend = \"setuptools.build_meta\"\n")

	// -- shared pyproject.toml -----------------------------------------------
	sharedUvConfig := opts.sharedUvConfig
	writeFile(t, filepath.Join(sharedDir, "pyproject.toml"),
		"[project]\n"+
			"name = \"shared\"\n"+
			"version = \"0.1.0\"\n"+
			"dependencies = [\"anyio>=4\"]\n"+
			"\n"+
			sharedUvConfig+"\n"+
			"\n"+
			"[build-system]\n"+
			"requires = [\"setuptools>=61\"]\n"+
			"build-backend = \"setuptools.build_meta\"\n")

	// -- extra pyproject.toml ------------------------------------------------
	writeFile(t, filepath.Join(extraDir, "pyproject.toml"),
		"[project]\n"+
			"name = \"extra\"\n"+
			"version = \"0.1.0\"\n"+
			"\n"+
			"[build-system]\n"+
			"requires = [\"setuptools>=61\"]\n"+
			"build-backend = \"setuptools.build_meta\"\n")

	// -- source files --------------------------------------------------------
	writeFile(t, filepath.Join(agentDir, "src", "agent", "graph.py"), "")
	writeFile(t, filepath.Join(sharedDir, "src", "shared", "auth.py"), "")
	writeFile(t, filepath.Join(extraDir, "src", "extra", "graph.py"), "")

	// -- config file ---------------------------------------------------------
	configPath := filepath.Join(deployDir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	// If configRelativeDir is non-default, also place langgraph.json there.
	if configRelDir != "deploy/agent" {
		altConfigPath := filepath.Join(configDir, "langgraph.json")
		writeFile(t, altConfigPath, "{}\n")
		configPath = altConfigPath
	}

	// -- extra files ---------------------------------------------------------
	for relPath, content := range opts.extraFiles {
		writeFile(t, filepath.Join(projectRoot, relPath), content)
	}

	return projectRoot, configPath
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

func TestUvLockBasic(t *testing.T) {
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{
		agentSources: "[tool.uv.sources]\nshared = { workspace = true }",
	})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
		"auth":   map[string]any{"path": "../../libs/shared/src/shared/auth.py:create_auth"},
	})

	docker, contexts, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker: %v", err)
	}

	// Core commands.
	assertContains(t, docker, "uv pip install --system")
	assertContains(t, docker,
		"uv export --package 'agent' --frozen --no-hashes --no-emit-project --no-emit-workspace")

	// Copy project metadata for uv export.
	assertContains(t, docker,
		"COPY --from=uv-workspace-root pyproject.toml /tmp/uv_export/project/pyproject.toml")
	assertContains(t, docker,
		"COPY --from=uv-workspace-root uv.lock /tmp/uv_export/project/uv.lock")

	// Additional build contexts.
	if _, ok := contexts["uv-workspace-root"]; !ok {
		t.Fatal("expected 'uv-workspace-root' in additional contexts")
	}

	// Workspace packages copied.
	assertContains(t, docker,
		"COPY --from=uv-workspace-root apps/agent /deps/workspace/apps/agent")
	assertContains(t, docker,
		"COPY --from=uv-workspace-root libs/shared /deps/workspace/libs/shared")
	// Unrelated member NOT copied.
	assertNotContains(t, docker,
		"libs/extra /deps/workspace/libs/extra")

	// Install order: shared before agent.
	assertContains(t, docker, "WORKDIR /deps/workspace/libs/shared")
	assertContains(t, docker, "WORKDIR /deps/workspace/apps/agent")
	sharedInstall := "uv pip install --system --no-cache-dir -c /api/constraints.txt --no-deps -e ."
	assertContains(t, docker, sharedInstall)

	// Ordering: shared COPY < shared WORKDIR < agent COPY < agent WORKDIR.
	sharedCopy := "COPY --from=uv-workspace-root libs/shared /deps/workspace/libs/shared"
	sharedWD := "WORKDIR /deps/workspace/libs/shared"
	agentCopy := "COPY --from=uv-workspace-root apps/agent /deps/workspace/apps/agent"
	agentWD := "WORKDIR /deps/workspace/apps/agent"
	if strings.Index(docker, sharedCopy) >= strings.Index(docker, sharedWD) {
		t.Error("shared COPY should appear before shared WORKDIR")
	}
	if strings.Index(docker, sharedWD) >= strings.Index(docker, agentCopy) {
		t.Error("shared WORKDIR should appear before agent COPY")
	}
	if strings.Index(docker, sharedWD) >= strings.Index(docker, agentWD) {
		t.Error("shared WORKDIR should appear before agent WORKDIR")
	}

	// No legacy dep-loop patterns.
	assertNotContains(t, docker, "for dep in /deps/*")
	assertNotContains(t, docker, "# -- Installing workspace packages --")
	assertContains(t, docker, "WORKDIR /tmp/uv_export/project")
	assertNotContains(t, docker, "RUN cd ")

	// Rewritten paths in env vars (Go JSON uses compact format without spaces).
	assertContains(t, docker,
		`"/deps/workspace/libs/shared/src/shared/auth.py:create_auth"`)
	assertContains(t, docker,
		`"/deps/workspace/apps/agent/src/agent/graph.py:graph"`)

	// Cleanup.
	assertContains(t, docker, "rm /usr/bin/uv /usr/bin/uvx")
}

func TestUvLockHonorsRootWorkspaceSources(t *testing.T) {
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{
		rootSources: "[tool.uv.sources]\nshared = { workspace = true }",
	})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
		"auth":   map[string]any{"path": "../../libs/shared/src/shared/auth.py:create_auth"},
	})

	docker, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker: %v", err)
	}

	assertContains(t, docker,
		"COPY --from=uv-workspace-root libs/shared /deps/workspace/libs/shared")
	assertContains(t, docker,
		`"/deps/workspace/libs/shared/src/shared/auth.py:create_auth"`)
}

func TestUvLockIgnoresUnrelatedWorkspacePackageSources(t *testing.T) {
	projectRoot, configPath := writeUvLockWorkspace(t, workspaceOpts{
		agentSources: "[tool.uv.sources]\nshared = { workspace = true }",
	})

	// Add badlib with [tool.uv.sources] pointing outside (an unrelated member).
	badlibDir := filepath.Join(projectRoot, "libs", "badlib")
	writeFile(t, filepath.Join(badlibDir, "pyproject.toml"),
		"[project]\n"+
			"name = \"badlib\"\n"+
			"version = \"0.1.0\"\n"+
			"\n"+
			"[tool.uv.sources]\n"+
			"outside = { path = \"../outside\" }\n"+
			"\n"+
			"[build-system]\n"+
			"requires = [\"setuptools>=61\"]\n"+
			"build-backend = \"setuptools.build_meta\"\n")

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
		"auth":   map[string]any{"path": "../../libs/shared/src/shared/auth.py:create_auth"},
	})

	docker, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker: %v", err)
	}

	assertContains(t, docker,
		"COPY --from=uv-workspace-root libs/shared /deps/workspace/libs/shared")
	assertNotContains(t, docker,
		"COPY --from=uv-workspace-root libs/badlib /deps/workspace/libs/badlib")
}

func TestUvLockIgnoresUnrelatedRootSources(t *testing.T) {
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{
		rootSources: "[tool.uv.sources]\nshared = { workspace = true }\nunused = { path = \"libs/extra\", editable = true }",
	})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
		"auth":   map[string]any{"path": "../../libs/shared/src/shared/auth.py:create_auth"},
	})

	docker, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker: %v", err)
	}

	assertContains(t, docker,
		"COPY --from=uv-workspace-root libs/shared /deps/workspace/libs/shared")
	assertNotContains(t, docker,
		"COPY --from=uv-workspace-root libs/extra /deps/workspace/libs/extra")
}

func TestUvLockValidatesRootPathSourcesRelativeToProjectRoot(t *testing.T) {
	projectRoot, configPath := writeUvLockWorkspace(t, workspaceOpts{
		rootSources: "[tool.uv.sources]\nshared = { path = \"../outside\", editable = true }",
	})

	// Create the outside directory the source points to.
	outsideDir := filepath.Join(filepath.Dir(projectRoot), "outside")
	writeFile(t, filepath.Join(outsideDir, "pyproject.toml"),
		"[project]\n"+
			"name = \"shared\"\n"+
			"version = \"0.1.0\"\n")

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})

	_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err == nil {
		t.Fatal("expected error for path outside project_root")
	}
	assertContains(t, err.Error(), "outside project_root")
}

func TestUvLockRequiresExplicitWorkspaceSources(t *testing.T) {
	// No agent_sources or root_sources => shared is NOT a workspace dep.
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
		"auth":   map[string]any{"path": "../../libs/shared/src/shared/auth.py:create_auth"},
	})

	_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err == nil {
		t.Fatal("expected error because shared is not an explicit workspace source")
	}
	assertContains(t, err.Error(), "not inside the target package 'agent'")
}

func TestUvLockAcceptsPathWorkspaceSources(t *testing.T) {
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{
		agentSources: "[tool.uv.sources]\nshared = { path = \"../../libs/shared\", editable = true }",
	})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})

	docker, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker: %v", err)
	}

	assertContains(t, docker,
		"COPY --from=uv-workspace-root libs/shared /deps/workspace/libs/shared")
}

func TestUvLockRejectsPackageFalseWorkspaceDependency(t *testing.T) {
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{
		agentSources:   "[tool.uv.sources]\nshared = { workspace = true }",
		sharedUvConfig: "[tool.uv]\npackage = false",
	})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})

	_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err == nil {
		t.Fatal("expected error for package = false workspace dependency")
	}
	assertContains(t, err.Error(), "tool.uv.package = false")
}

func TestUvLockAcceptsRootPathWorkspaceSources(t *testing.T) {
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{
		rootSources: "[tool.uv.sources]\nshared = { path = \"libs/shared\", editable = true }",
	})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
		"auth":   map[string]any{"path": "../../libs/shared/src/shared/auth.py:create_auth"},
	})

	docker, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker: %v", err)
	}

	assertContains(t, docker,
		"COPY --from=uv-workspace-root libs/shared /deps/workspace/libs/shared")
	assertContains(t, docker,
		`"/deps/workspace/libs/shared/src/shared/auth.py:create_auth"`)
}

func TestUvLockRejectsMismatchedPathWorkspaceSources(t *testing.T) {
	// agent sources point to libs/extra with the name "shared" -> mismatch.
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{
		agentSources: "[tool.uv.sources]\nshared = { path = \"../../libs/extra\", editable = true }",
	})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})

	_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err == nil {
		t.Fatal("expected error for mismatched path workspace sources")
	}
	assertContains(t, err.Error(), "dependency name and the workspace package name must match")
}

func TestUvLockDetectsJsPmFromTargetPackageRoot(t *testing.T) {
	projectRoot, configPath := writeUvLockWorkspace(t, workspaceOpts{
		agentSources: "[tool.uv.sources]\nshared = { workspace = true }",
	})

	agentDir := filepath.Join(projectRoot, "apps", "agent")
	writeFile(t, filepath.Join(agentDir, "package.json"), "{\"packageManager\":\"pnpm@9.0.0\"}\n")
	writeFile(t, filepath.Join(agentDir, "pnpm-lock.yaml"), "lockfileVersion: 9.0\n")
	writeFile(t, filepath.Join(agentDir, "ui.tsx"), "export const ui = null;\n")

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
		"ui":     map[string]any{"agent": "../../apps/agent/ui.tsx"},
	})

	docker, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker: %v", err)
	}

	assertContains(t, docker, "WORKDIR /deps/workspace/apps/agent")
	assertContains(t, docker,
		"RUN pnpm i --frozen-lockfile && tsx /api/langgraph_api/js/build.mts")
	assertContains(t, docker, `/deps/workspace/apps/agent/ui.tsx`)
}

func TestUvLockUsesWorkdirForJsInstallWithSpecialChars(t *testing.T) {
	projectRoot, _ := writeUvLockWorkspace(t, workspaceOpts{
		configRelativeDir: "apps/agent;echo pwned",
	})

	// The custom configRelativeDir creates its own langgraph.json.
	// We need to place the agent pyproject.toml at this directory.
	agentDir := filepath.Join(projectRoot, "apps", "agent;echo pwned")
	writeFile(t, filepath.Join(agentDir, "package.json"), "{\"packageManager\":\"pnpm@9.0.0\"}\n")
	writeFile(t, filepath.Join(agentDir, "pnpm-lock.yaml"), "lockfileVersion: 9.0\n")
	writeFile(t, filepath.Join(agentDir, "ui.tsx"), "export const ui = null;\n")
	writeFile(t, filepath.Join(agentDir, "pyproject.toml"),
		"[project]\n"+
			"name = \"agent\"\n"+
			"version = \"0.1.0\"\n"+
			"dependencies = [\"shared\", \"httpx>=0.28\"]\n"+
			"\n"+
			"[build-system]\n"+
			"requires = [\"setuptools>=61\"]\n"+
			"build-backend = \"setuptools.build_meta\"\n")
	writeFile(t, filepath.Join(agentDir, "src", "agent", "graph.py"), "")

	// Remove the default apps/agent so there's no duplicate package name.
	if err := os.RemoveAll(filepath.Join(projectRoot, "apps", "agent")); err != nil {
		t.Fatal(err)
	}

	// Update workspace members to include the special-char directory.
	writeFile(t, filepath.Join(projectRoot, "pyproject.toml"),
		"[project]\n"+
			"name = \"workspace-root\"\n"+
			"version = \"0.1.0\"\n"+
			"\n"+
			"[tool.uv.workspace]\n"+
			"members = [\"apps/*\", \"libs/*\"]\n"+
			"\n"+
			"[build-system]\n"+
			"requires = [\"setuptools>=61\"]\n"+
			"build-backend = \"setuptools.build_meta\"\n")

	configPath := filepath.Join(agentDir, "langgraph.json")

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "./src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
		"ui":     map[string]any{"agent": "./ui.tsx"},
	})

	docker, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker: %v", err)
	}

	assertContains(t, docker, "WORKDIR /deps/workspace/apps/agent;echo pwned")
	assertContains(t, docker,
		"RUN pnpm i --frozen-lockfile && tsx /api/langgraph_api/js/build.mts")
	assertNotContains(t, docker, "RUN cd /deps/workspace/apps/agent;echo pwned")
}

func TestUvLockSupportsSingleUvProjectRoot(t *testing.T) {
	base := t.TempDir()
	projectRoot := filepath.Join(base, "single")
	if err := os.MkdirAll(projectRoot, 0o755); err != nil {
		t.Fatal(err)
	}

	writeFile(t, filepath.Join(projectRoot, "uv.lock"), "# uv lock file\n")
	writeFile(t, filepath.Join(projectRoot, "pyproject.toml"),
		"[project]\n"+
			"name = \"single-app\"\n"+
			"version = \"0.1.0\"\n"+
			"dependencies = [\"httpx>=0.28\"]\n"+
			"\n"+
			"[build-system]\n"+
			"requires = [\"setuptools>=61\"]\n"+
			"build-backend = \"setuptools.build_meta\"\n")
	configPath := filepath.Join(projectRoot, "langgraph.json")
	writeFile(t, configPath, "{}\n")
	writeFile(t, filepath.Join(projectRoot, "src", "agent.py"), "graph = object()\n")

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs":         map[string]any{"agent": "./src/agent.py:graph"},
		"source":         map[string]any{"kind": "uv"},
	})

	docker, contexts, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker: %v", err)
	}

	assertContains(t, docker,
		"uv export --package 'single-app' --frozen --no-hashes --no-emit-project --no-emit-workspace")
	assertContains(t, docker, `"/deps/workspace/src/agent.py:graph"`)
	if len(contexts) != 0 {
		t.Fatalf("expected no additional contexts for single-project root, got %d: %v",
			len(contexts), contexts)
	}
}

func TestUvLockRejectsInvalidSourcePackageType(t *testing.T) {
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})
	// Override the validated source.package to an integer.
	cfg["source"].(map[string]any)["package"] = 123

	_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err == nil {
		t.Fatal("expected error for non-string source.package")
	}
	assertContains(t, err.Error(), "`source.package` must be a non-empty string")
}

func TestUvLockRejectsPathsOutsideTargetClosure(t *testing.T) {
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{
		agentSources: "[tool.uv.sources]\nshared = { workspace = true }",
	})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			// Graph path points into libs/extra which is NOT a dependency of agent.
			"agent": "../../libs/extra/src/extra/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})

	_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err == nil {
		t.Fatal("expected error for graph path outside target closure")
	}
	assertContains(t, err.Error(), "not inside the target package 'agent'")
}

func TestUvLockRejectsUnrelatedMemberWhenRootInClosure(t *testing.T) {
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{
		agentDependencies: []string{"workspace-root", "shared", "httpx>=0.28"},
		rootSources:       "[tool.uv.sources]\nshared = { workspace = true }\nworkspace-root = { workspace = true }",
		agentSources:      "[tool.uv.sources]\nshared = { workspace = true }\nworkspace-root = { workspace = true }",
	})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			// extra is a workspace member but NOT a dependency of agent.
			"agent": "../../libs/extra/src/extra/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})

	_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err == nil {
		t.Fatal("expected error for unrelated member when root is in closure")
	}
	assertContains(t, err.Error(), "not inside the target package 'agent'")
}

func TestUvLockRootPackageCopySkipsUnrelatedMembers(t *testing.T) {
	projectRoot, configPath := writeUvLockWorkspace(t, workspaceOpts{
		agentDependencies: []string{"workspace-root", "shared", "httpx>=0.28"},
		rootSources:       "[tool.uv.sources]\nshared = { workspace = true }\nworkspace-root = { workspace = true }",
		agentSources:      "[tool.uv.sources]\nshared = { workspace = true }\nworkspace-root = { workspace = true }",
	})

	// Add files in the workspace root package itself.
	writeFile(t, filepath.Join(projectRoot, "src", "workspace_root", "__init__.py"), "__all__ = []\n")
	writeFile(t, filepath.Join(projectRoot, "README.md"), "workspace root package\n")

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs": map[string]any{
			"agent": "../../apps/agent/src/agent/graph.py:graph",
		},
		"source": map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})

	docker, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker: %v", err)
	}

	// Should NOT do a blanket COPY of the whole workspace root.
	assertNotContains(t, docker, "COPY --from=uv-workspace-root . /deps/workspace")
	// Should copy specific entries.
	assertContains(t, docker, "COPY --from=uv-workspace-root src /deps/workspace/src")
	assertContains(t, docker, "COPY --from=uv-workspace-root README.md /deps/workspace/README.md")
	// Should NOT copy unrelated member dirs.
	assertNotContains(t, docker,
		"COPY --from=uv-workspace-root libs/extra /deps/workspace/libs/extra")
	// Should install workspace root from /deps/workspace.
	assertContains(t, docker, "WORKDIR /deps/workspace")
	assertContains(t, docker,
		"uv pip install --system --no-cache-dir -c /api/constraints.txt --no-deps -e .")
}

func TestUvLockMissingLockfile(t *testing.T) {
	projectRoot, configPath := writeUvLockWorkspace(t, workspaceOpts{})

	// Remove uv.lock.
	if err := os.Remove(filepath.Join(projectRoot, "uv.lock")); err != nil {
		t.Fatal(err)
	}

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs":         map[string]any{"agent": "../../apps/agent/src/agent/graph.py:graph"},
		"source":         map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})

	_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err == nil {
		t.Fatal("expected error for missing uv.lock")
	}
	assertContains(t, err.Error(), "No uv.lock found")
}

func TestUvLockMissingPyproject(t *testing.T) {
	projectRoot, configPath := writeUvLockWorkspace(t, workspaceOpts{})

	// Remove root pyproject.toml.
	if err := os.Remove(filepath.Join(projectRoot, "pyproject.toml")); err != nil {
		t.Fatal(err)
	}

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs":         map[string]any{"agent": "../../apps/agent/src/agent/graph.py:graph"},
		"source":         map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})

	_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.47",
	})
	if err == nil {
		t.Fatal("expected error for missing pyproject.toml")
	}
	assertContains(t, err.Error(), "No pyproject.toml found")
}

func TestUvLockOldImage(t *testing.T) {
	_, configPath := writeUvLockWorkspace(t, workspaceOpts{})

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs":         map[string]any{"agent": "../../apps/agent/src/agent/graph.py:graph"},
		"source":         map[string]any{"kind": "uv", "root": "../..", "package": "agent"},
	})

	_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api:0.2.46",
	})
	if err == nil {
		t.Fatal("expected error for old image without uv support")
	}
	assertContains(t, err.Error(), "requires a base image with uv support")
}
