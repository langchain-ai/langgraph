package config

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// mustValidate validates a config map and fatals on error.
func mustValidate(t *testing.T, raw map[string]any) map[string]any {
	t.Helper()
	cfg, err := ValidateConfig(raw)
	if err != nil {
		t.Fatalf("ValidateConfig failed: %v", err)
	}
	return cfg
}

// writeFile is a test helper to create a file with the given content.
func writeFile(t *testing.T, path, content string) {
	t.Helper()
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", dir, err)
	}
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("WriteFile(%q): %v", path, err)
	}
}

// assertContains checks that got contains the substring want.
func assertContains(t *testing.T, got, want string) {
	t.Helper()
	if !strings.Contains(got, want) {
		t.Errorf("expected output to contain %q, but it does not.\nGot:\n%s", want, got)
	}
}

// assertNotContains checks that got does NOT contain the substring want.
func assertNotContains(t *testing.T, got, want string) {
	t.Helper()
	if strings.Contains(got, want) {
		t.Errorf("expected output to NOT contain %q, but it does.\nGot:\n%s", want, got)
	}
}

// extractEnvJSON extracts and parses a JSON value from an ENV line in a Dockerfile.
func extractEnvJSON(t *testing.T, dockerfile, varName string) map[string]any {
	t.Helper()
	prefix := "ENV " + varName + "='"
	for _, line := range strings.Split(dockerfile, "\n") {
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, prefix) && strings.HasSuffix(line, "'") {
			jsonStr := line[len(prefix) : len(line)-1]
			var result map[string]any
			if err := json.Unmarshal([]byte(jsonStr), &result); err != nil {
				t.Fatalf("failed to parse JSON from %s: %v\njsonStr=%s", varName, err, jsonStr)
			}
			return result
		}
	}
	t.Fatalf("%s not found in Dockerfile env lines", varName)
	return nil
}

// setupSimplePythonProject creates a minimal Python project directory with a graph file
// and returns the config path.
func setupSimplePythonProject(t *testing.T, dir string) string {
	t.Helper()
	writeFile(t, filepath.Join(dir, "agent.py"), "graph = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")
	return configPath
}

// setupPythonProjectWithGraphs creates a project with a graphs subdirectory.
func setupPythonProjectWithGraphs(t *testing.T, dir string) string {
	t.Helper()
	writeFile(t, filepath.Join(dir, "graphs", "agent.py"), "graph = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")
	return configPath
}

// setupNodeProject creates a minimal Node.js project directory.
func setupNodeProject(t *testing.T, dir string) string {
	t.Helper()
	writeFile(t, filepath.Join(dir, "graphs", "agent.js"), "export const graph = {};\n")
	writeFile(t, filepath.Join(dir, "graphs", "auth.mts"), "export const auth = {};\n")
	writeFile(t, filepath.Join(dir, "package.json"), `{"name":"test"}`+"\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")
	return configPath
}

// ---------------------------------------------------------------------------
// DefaultBaseImage tests
// ---------------------------------------------------------------------------

func TestDefaultBaseImage(t *testing.T) {
	t.Run("python_combined_mode", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"dependencies": []any{"."},
			"graphs":       map[string]any{"agent": "./agent.py:graph"},
		})
		got := DefaultBaseImage(cfg, "combined_queue_worker")
		if got != "langchain/langgraph-api" {
			t.Fatalf("expected langchain/langgraph-api, got %q", got)
		}
	})

	t.Run("python_default_mode", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"dependencies": []any{"."},
			"graphs":       map[string]any{"agent": "./agent.py:graph"},
		})
		got := DefaultBaseImage(cfg, "")
		if got != "langchain/langgraph-api" {
			t.Fatalf("expected langchain/langgraph-api, got %q", got)
		}
	})

	t.Run("python_distributed_mode", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"dependencies": []any{"."},
			"graphs":       map[string]any{"agent": "./agent.py:graph"},
		})
		got := DefaultBaseImage(cfg, "distributed")
		if got != "langchain/langgraph-executor" {
			t.Fatalf("expected langchain/langgraph-executor, got %q", got)
		}
	})

	t.Run("distributed_with_explicit_base_image", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"dependencies": []any{"."},
			"graphs":       map[string]any{"agent": "./agent.py:graph"},
			"base_image":   "my-custom-image:latest",
		})
		got := DefaultBaseImage(cfg, "distributed")
		if got != "my-custom-image:latest" {
			t.Fatalf("expected my-custom-image:latest, got %q", got)
		}
	})

	t.Run("nodejs", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"node_version": "20",
			"graphs":       map[string]any{"agent": "./agent.js:graph"},
		})
		got := DefaultBaseImage(cfg, "")
		if got != "langchain/langgraphjs-api" {
			t.Fatalf("expected langchain/langgraphjs-api, got %q", got)
		}
	})
}

// ---------------------------------------------------------------------------
// DockerTag tests
// ---------------------------------------------------------------------------

func TestDockerTag(t *testing.T) {
	t.Run("python_debian", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"python_version": "3.11",
			"dependencies":   []any{"."},
			"graphs":         map[string]any{"agent": "./agent.py:graph"},
		})
		got := DockerTag(cfg, "langchain/langgraph-api", "")
		if got != "langchain/langgraph-api:3.11" {
			t.Fatalf("expected langchain/langgraph-api:3.11, got %q", got)
		}
	})

	t.Run("python_explicit_debian", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"python_version": "3.11",
			"dependencies":   []any{"."},
			"graphs":         map[string]any{"agent": "./agent.py:graph"},
			"image_distro":   "debian",
		})
		got := DockerTag(cfg, "langchain/langgraph-api", "")
		if got != "langchain/langgraph-api:3.11" {
			t.Fatalf("expected langchain/langgraph-api:3.11, got %q", got)
		}
	})

	t.Run("python_wolfi", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"python_version": "3.11",
			"dependencies":   []any{"."},
			"graphs":         map[string]any{"agent": "./agent.py:graph"},
			"image_distro":   "wolfi",
		})
		got := DockerTag(cfg, "langchain/langgraph-api", "")
		if got != "langchain/langgraph-api:3.11-wolfi" {
			t.Fatalf("expected langchain/langgraph-api:3.11-wolfi, got %q", got)
		}
	})

	t.Run("node_debian", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"node_version": "20",
			"graphs":       map[string]any{"agent": "./agent.js:graph"},
		})
		got := DockerTag(cfg, "langchain/langgraphjs-api", "")
		if got != "langchain/langgraphjs-api:20" {
			t.Fatalf("expected langchain/langgraphjs-api:20, got %q", got)
		}
	})

	t.Run("node_wolfi", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"node_version": "20",
			"graphs":       map[string]any{"agent": "./agent.js:graph"},
			"image_distro": "wolfi",
		})
		got := DockerTag(cfg, "langchain/langgraphjs-api", "")
		if got != "langchain/langgraphjs-api:20-wolfi" {
			t.Fatalf("expected langchain/langgraphjs-api:20-wolfi, got %q", got)
		}
	})

	t.Run("custom_base_image_wolfi", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"python_version": "3.12",
			"dependencies":   []any{"."},
			"graphs":         map[string]any{"agent": "./agent.py:graph"},
			"image_distro":   "wolfi",
			"base_image":     "my-registry/custom-image",
		})
		got := DockerTag(cfg, "my-registry/custom-image", "")
		if got != "my-registry/custom-image:3.12-wolfi" {
			t.Fatalf("expected my-registry/custom-image:3.12-wolfi, got %q", got)
		}
	})

	t.Run("multiplatform_python_node_wolfi", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"python_version": "3.11",
			"node_version":   "20",
			"dependencies":   []any{"."},
			"graphs":         map[string]any{"python": "./agent.py:graph", "js": "./agent.js:graph"},
			"image_distro":   "wolfi",
		})
		got := DockerTag(cfg, "", "")
		// Should default to Python when both are present
		if got != "langchain/langgraph-api:3.11-wolfi" {
			t.Fatalf("expected langchain/langgraph-api:3.11-wolfi, got %q", got)
		}
	})

	t.Run("python_versions_with_wolfi", func(t *testing.T) {
		for _, version := range []string{"3.11", "3.12", "3.13"} {
			cfg := mustValidate(t, map[string]any{
				"python_version": version,
				"dependencies":   []any{"."},
				"graphs":         map[string]any{"agent": "./agent.py:graph"},
				"image_distro":   "wolfi",
			})
			expected := "langchain/langgraph-api:" + version + "-wolfi"
			got := DockerTag(cfg, "", "")
			if got != expected {
				t.Fatalf("Python %s: expected %q, got %q", version, expected, got)
			}
		}
	})

	t.Run("node_versions_with_wolfi", func(t *testing.T) {
		for _, version := range []string{"20", "21", "22"} {
			cfg := mustValidate(t, map[string]any{
				"node_version": version,
				"graphs":       map[string]any{"agent": "./agent.js:graph"},
				"image_distro": "wolfi",
			})
			expected := "langchain/langgraphjs-api:" + version + "-wolfi"
			got := DockerTag(cfg, "", "")
			if got != expected {
				t.Fatalf("Node %s: expected %q, got %q", version, expected, got)
			}
		}
	})

	t.Run("internal_docker_tag_overrides", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"python_version":       "3.11",
			"dependencies":         []any{"."},
			"graphs":               map[string]any{"agent": "./agent.py:graph"},
			"_INTERNAL_docker_tag": "internal-tag",
		})
		got := DockerTag(cfg, "", "0.2.74")
		if got != "langchain/langgraph-api:internal-tag" {
			t.Fatalf("expected langchain/langgraph-api:internal-tag, got %q", got)
		}
	})
}

func TestDockerTagWithAPIVersion(t *testing.T) {
	apiVersion := "0.2.74"

	for _, inConfig := range []bool{false, true} {
		label := "param"
		if inConfig {
			label = "in_config"
		}

		t.Run("python_default_distro_"+label, func(t *testing.T) {
			raw := map[string]any{
				"python_version": "3.11",
				"dependencies":   []any{"."},
				"graphs":         map[string]any{"agent": "./agent.py:graph"},
			}
			passedVersion := apiVersion
			if inConfig {
				raw["api_version"] = apiVersion
				passedVersion = ""
			}
			cfg := mustValidate(t, raw)
			got := DockerTag(cfg, "", passedVersion)
			if got != "langchain/langgraph-api:0.2.74-py3.11" {
				t.Fatalf("expected langchain/langgraph-api:0.2.74-py3.11, got %q", got)
			}
		})

		t.Run("python_wolfi_distro_"+label, func(t *testing.T) {
			raw := map[string]any{
				"python_version": "3.12",
				"dependencies":   []any{"."},
				"graphs":         map[string]any{"agent": "./agent.py:graph"},
				"image_distro":   "wolfi",
			}
			passedVersion := apiVersion
			if inConfig {
				raw["api_version"] = apiVersion
				passedVersion = ""
			}
			cfg := mustValidate(t, raw)
			got := DockerTag(cfg, "", passedVersion)
			if got != "langchain/langgraph-api:0.2.74-py3.12-wolfi" {
				t.Fatalf("expected langchain/langgraph-api:0.2.74-py3.12-wolfi, got %q", got)
			}
		})

		t.Run("node_default_distro_"+label, func(t *testing.T) {
			raw := map[string]any{
				"node_version": "20",
				"graphs":       map[string]any{"agent": "./agent.js:graph"},
			}
			passedVersion := apiVersion
			if inConfig {
				raw["api_version"] = apiVersion
				passedVersion = ""
			}
			cfg := mustValidate(t, raw)
			got := DockerTag(cfg, "", passedVersion)
			if got != "langchain/langgraphjs-api:0.2.74-node20" {
				t.Fatalf("expected langchain/langgraphjs-api:0.2.74-node20, got %q", got)
			}
		})

		t.Run("node_wolfi_distro_"+label, func(t *testing.T) {
			raw := map[string]any{
				"node_version": "20",
				"graphs":       map[string]any{"agent": "./agent.js:graph"},
				"image_distro": "wolfi",
			}
			passedVersion := apiVersion
			if inConfig {
				raw["api_version"] = apiVersion
				passedVersion = ""
			}
			cfg := mustValidate(t, raw)
			got := DockerTag(cfg, "", passedVersion)
			if got != "langchain/langgraphjs-api:0.2.74-node20-wolfi" {
				t.Fatalf("expected langchain/langgraphjs-api:0.2.74-node20-wolfi, got %q", got)
			}
		})

		t.Run("custom_base_image_"+label, func(t *testing.T) {
			raw := map[string]any{
				"python_version": "3.11",
				"dependencies":   []any{"."},
				"graphs":         map[string]any{"agent": "./agent.py:graph"},
				"base_image":     "my-registry/custom-image",
			}
			passedVersion := apiVersion
			if inConfig {
				raw["api_version"] = apiVersion
				passedVersion = ""
			}
			cfg := mustValidate(t, raw)
			got := DockerTag(cfg, "my-registry/custom-image", passedVersion)
			if got != "my-registry/custom-image:0.2.74-py3.11" {
				t.Fatalf("expected my-registry/custom-image:0.2.74-py3.11, got %q", got)
			}
		})

		t.Run("python_versions_"+label, func(t *testing.T) {
			for _, pyVer := range []string{"3.11", "3.12", "3.13"} {
				raw := map[string]any{
					"python_version": pyVer,
					"dependencies":   []any{"."},
					"graphs":         map[string]any{"agent": "./agent.py:graph"},
				}
				passedVersion := apiVersion
				if inConfig {
					raw["api_version"] = apiVersion
					passedVersion = ""
				}
				cfg := mustValidate(t, raw)
				expected := "langchain/langgraph-api:" + apiVersion + "-py" + pyVer
				got := DockerTag(cfg, "", passedVersion)
				if got != expected {
					t.Fatalf("Python %s: expected %q, got %q", pyVer, expected, got)
				}
			}
		})

		t.Run("multiplatform_"+label, func(t *testing.T) {
			raw := map[string]any{
				"python_version": "3.11",
				"node_version":   "20",
				"dependencies":   []any{"."},
				"graphs":         map[string]any{"python": "./agent.py:graph", "js": "./agent.js:graph"},
			}
			passedVersion := apiVersion
			if inConfig {
				raw["api_version"] = apiVersion
				passedVersion = ""
			}
			cfg := mustValidate(t, raw)
			got := DockerTag(cfg, "", passedVersion)
			if got != "langchain/langgraph-api:0.2.74-py3.11" {
				t.Fatalf("expected langchain/langgraph-api:0.2.74-py3.11, got %q", got)
			}
		})

		t.Run("langgraph_server_base_"+label, func(t *testing.T) {
			raw := map[string]any{
				"python_version": "3.11",
				"dependencies":   []any{"."},
				"graphs":         map[string]any{"agent": "./agent.py:graph"},
			}
			passedVersion := apiVersion
			if inConfig {
				raw["api_version"] = apiVersion
				passedVersion = ""
			}
			cfg := mustValidate(t, raw)
			got := DockerTag(cfg, "langchain/langgraph-server", passedVersion)
			if got != "langchain/langgraph-server:0.2.74-py3.11" {
				t.Fatalf("expected langchain/langgraph-server:0.2.74-py3.11, got %q", got)
			}
		})
	}

	t.Run("without_api_version", func(t *testing.T) {
		cfg := mustValidate(t, map[string]any{
			"python_version": "3.11",
			"dependencies":   []any{"."},
			"graphs":         map[string]any{"agent": "./agent.py:graph"},
		})
		got := DockerTag(cfg, "", "")
		if got != "langchain/langgraph-api:3.11" {
			t.Fatalf("expected langchain/langgraph-api:3.11, got %q", got)
		}
	})
}

// ---------------------------------------------------------------------------
// ImageSupportsUV tests
// ---------------------------------------------------------------------------

func TestImageSupportsUV(t *testing.T) {
	t.Run("modern_image", func(t *testing.T) {
		if !ImageSupportsUV("langchain/langgraph-api:0.2.47") {
			t.Fatal("expected ImageSupportsUV to return true for 0.2.47")
		}
	})

	t.Run("old_image", func(t *testing.T) {
		if ImageSupportsUV("langchain/langgraph-api:0.2.46") {
			t.Fatal("expected ImageSupportsUV to return false for 0.2.46")
		}
	})

	t.Run("trial_image", func(t *testing.T) {
		if ImageSupportsUV("langchain/langgraph-trial") {
			t.Fatal("expected ImageSupportsUV to return false for trial image")
		}
	})

	t.Run("no_version_tag", func(t *testing.T) {
		if !ImageSupportsUV("langchain/langgraph-api") {
			t.Fatal("expected ImageSupportsUV to return true for image without version")
		}
	})

	t.Run("version_3.11", func(t *testing.T) {
		if !ImageSupportsUV("langchain/langgraph-api:3.11") {
			t.Fatal("expected ImageSupportsUV to return true for 3.11")
		}
	})
}

// ---------------------------------------------------------------------------
// GetBuildToolsToUninstall tests
// ---------------------------------------------------------------------------

func TestGetBuildToolsToUninstall(t *testing.T) {
	t.Run("nil_keep_pkg_tools", func(t *testing.T) {
		cfg := map[string]any{}
		tools, err := GetBuildToolsToUninstall(cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tools) != 3 {
			t.Fatalf("expected 3 tools, got %d: %v", len(tools), tools)
		}
	})

	t.Run("keep_pkg_tools_true", func(t *testing.T) {
		cfg := map[string]any{"keep_pkg_tools": true}
		tools, err := GetBuildToolsToUninstall(cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if tools != nil {
			t.Fatalf("expected nil, got %v", tools)
		}
	})

	t.Run("keep_pkg_tools_false", func(t *testing.T) {
		cfg := map[string]any{"keep_pkg_tools": false}
		tools, err := GetBuildToolsToUninstall(cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tools) != 3 {
			t.Fatalf("expected 3 tools, got %d", len(tools))
		}
	})

	t.Run("keep_pkg_tools_list", func(t *testing.T) {
		cfg := map[string]any{"keep_pkg_tools": []any{"pip", "setuptools"}}
		tools, err := GetBuildToolsToUninstall(cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if len(tools) != 1 || tools[0] != "wheel" {
			t.Fatalf("expected [wheel], got %v", tools)
		}
	})
}

// ---------------------------------------------------------------------------
// BuildRuntimeEnvVars tests
// ---------------------------------------------------------------------------

func TestBuildRuntimeEnvVars(t *testing.T) {
	t.Run("graphs_only", func(t *testing.T) {
		cfg := map[string]any{
			"graphs": map[string]any{"agent": "./agent.py:graph"},
		}
		vars := BuildRuntimeEnvVars(cfg)
		found := false
		for _, v := range vars {
			if strings.Contains(v, "LANGSERVE_GRAPHS=") {
				found = true
			}
		}
		if !found {
			t.Fatal("expected LANGSERVE_GRAPHS in env vars")
		}
	})

	t.Run("with_webhooks", func(t *testing.T) {
		cfg := map[string]any{
			"graphs":   map[string]any{"agent": "./agent.py:graph"},
			"webhooks": map[string]any{"env_prefix": "LG_"},
		}
		vars := BuildRuntimeEnvVars(cfg)
		found := false
		for _, v := range vars {
			if strings.Contains(v, "LANGGRAPH_WEBHOOKS=") {
				found = true
			}
		}
		if !found {
			t.Fatal("expected LANGGRAPH_WEBHOOKS in env vars")
		}
	})

	t.Run("without_webhooks", func(t *testing.T) {
		cfg := map[string]any{
			"graphs": map[string]any{"agent": "./agent.py:graph"},
		}
		vars := BuildRuntimeEnvVars(cfg)
		for _, v := range vars {
			if strings.Contains(v, "LANGGRAPH_WEBHOOKS=") {
				t.Fatal("did not expect LANGGRAPH_WEBHOOKS in env vars")
			}
		}
	})

	t.Run("with_encryption", func(t *testing.T) {
		cfg := map[string]any{
			"graphs":     map[string]any{"agent": "./agent.py:graph"},
			"encryption": map[string]any{"path": "./enc.py:enc"},
		}
		vars := BuildRuntimeEnvVars(cfg)
		found := false
		for _, v := range vars {
			if strings.Contains(v, "LANGGRAPH_ENCRYPTION=") {
				found = true
			}
		}
		if !found {
			t.Fatal("expected LANGGRAPH_ENCRYPTION in env vars")
		}
	})
}

// ---------------------------------------------------------------------------
// GetNodePMInstallCmd tests
// ---------------------------------------------------------------------------

func TestGetNodePMInstallCmd(t *testing.T) {
	t.Run("npm_default", func(t *testing.T) {
		dir := t.TempDir()
		got := GetNodePMInstallCmd(dir)
		if got != "npm i" {
			t.Fatalf("expected 'npm i', got %q", got)
		}
	})

	t.Run("yarn_lock", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, filepath.Join(dir, "yarn.lock"), "")
		got := GetNodePMInstallCmd(dir)
		if got != "yarn install --frozen-lockfile" {
			t.Fatalf("expected 'yarn install --frozen-lockfile', got %q", got)
		}
	})

	t.Run("pnpm_lock", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, filepath.Join(dir, "pnpm-lock.yaml"), "")
		got := GetNodePMInstallCmd(dir)
		if got != "pnpm i --frozen-lockfile" {
			t.Fatalf("expected 'pnpm i --frozen-lockfile', got %q", got)
		}
	})

	t.Run("package_lock_json", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, filepath.Join(dir, "package-lock.json"), "{}")
		got := GetNodePMInstallCmd(dir)
		if got != "npm ci" {
			t.Fatalf("expected 'npm ci', got %q", got)
		}
	})

	t.Run("bun_lock", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, filepath.Join(dir, "bun.lockb"), "")
		got := GetNodePMInstallCmd(dir)
		if got != "bun i" {
			t.Fatalf("expected 'bun i', got %q", got)
		}
	})

	t.Run("packageManager_pnpm", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, filepath.Join(dir, "package.json"), `{"packageManager":"pnpm@9.0.0"}`)
		got := GetNodePMInstallCmd(dir)
		if got != "pnpm i" {
			t.Fatalf("expected 'pnpm i', got %q", got)
		}
	})

	t.Run("packageManager_yarn", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, filepath.Join(dir, "package.json"), `{"packageManager":"yarn@4.0.0"}`)
		got := GetNodePMInstallCmd(dir)
		if got != "yarn install" {
			t.Fatalf("expected 'yarn install', got %q", got)
		}
	})
}

// ---------------------------------------------------------------------------
// GetPipCleanupLines tests
// ---------------------------------------------------------------------------

func TestGetPipCleanupLines(t *testing.T) {
	t.Run("uv_with_all_tools", func(t *testing.T) {
		result := GetPipCleanupLines("uv pip install --system", []string{"pip", "setuptools", "wheel"}, "uv")
		assertContains(t, result, "RUN pip uninstall -y pip setuptools wheel")
		assertContains(t, result, "rm /usr/bin/uv /usr/bin/uvx")
		assertContains(t, result, "/usr/local/lib/python*/site-packages/pip*")
	})

	t.Run("pip_with_all_tools", func(t *testing.T) {
		result := GetPipCleanupLines("pip install", []string{"pip", "setuptools", "wheel"}, "pip")
		assertContains(t, result, "RUN pip uninstall -y pip setuptools wheel")
		assertNotContains(t, result, "rm /usr/bin/uv")
	})

	t.Run("no_tools_to_uninstall_uv", func(t *testing.T) {
		result := GetPipCleanupLines("uv pip install --system", nil, "uv")
		assertNotContains(t, result, "RUN pip uninstall")
		assertContains(t, result, "rm /usr/bin/uv /usr/bin/uvx")
	})
}

// ---------------------------------------------------------------------------
// ConfigToDocker tests -- Python
// ---------------------------------------------------------------------------

func TestConfigToDockerSimple(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "FROM langchain/langgraph-api:3.11")
	assertContains(t, dockerfile, "LANGSERVE_GRAPHS=")
	assertContains(t, dockerfile, "uv pip install --system")
	// Should contain working directory reference
	baseName := filepath.Base(dir)
	assertContains(t, dockerfile, "/deps/outer-"+baseName)
}

func TestConfigToDockerPipConfig(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)
	writeFile(t, filepath.Join(dir, "pipconfig.txt"), "[global]\nindex-url = https://pypi.org/simple\n")

	cfg := mustValidate(t, map[string]any{
		"dependencies":    []any{"."},
		"graphs":          map[string]any{"agent": "./agent.py:graph"},
		"pip_config_file": "pipconfig.txt",
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "FROM langchain/langgraph-api:3.11")
	assertContains(t, dockerfile, "ADD pipconfig.txt /pipconfig.txt")
	assertContains(t, dockerfile, "PIP_CONFIG_FILE=/pipconfig.txt")
}

func TestConfigToDockerLocalDeps(t *testing.T) {
	dir := t.TempDir()
	// Create graphs directory with a Python file (src-layout faux package)
	writeFile(t, filepath.Join(dir, "graphs", "subpkg", "agent.py"), "graph = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"./graphs"},
		"graphs":       map[string]any{"agent": "./graphs/subpkg/agent.py:graph"},
	})

	dockerfile, contexts, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api-custom",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "FROM langchain/langgraph-api-custom:3.11")
	assertContains(t, dockerfile, "Adding non-package dependency graphs")
	assertContains(t, dockerfile, "/deps/outer-graphs")
	// No additional contexts needed for child directories
	if len(contexts) != 0 {
		t.Fatalf("expected 0 additional contexts, got %d: %v", len(contexts), contexts)
	}
}

func TestConfigToDockerPyproject(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "pyproject.toml"), `[project]
name = "custom"
version = "0.1"
dependencies = ["langchain"]`)
	writeFile(t, filepath.Join(dir, "graphs", "agent.py"), "graph = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./graphs/agent.py:graph"},
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	baseName := filepath.Base(dir)
	assertContains(t, dockerfile, "FROM langchain/langgraph-api:3.11")
	assertContains(t, dockerfile, "Adding local package .")
	assertContains(t, dockerfile, "ADD . /deps/"+baseName)
	assertContains(t, dockerfile, "WORKDIR /deps/"+baseName)
}

func TestConfigToDockerNodeJS(t *testing.T) {
	dir := t.TempDir()
	configPath := setupNodeProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"node_version":     "20",
		"graphs":           map[string]any{"agent": "./graphs/agent.js:graph"},
		"dockerfile_lines": []any{"ARG meow", "ARG foo"},
		"auth":             map[string]any{"path": "./graphs/auth.mts:auth"},
		"ui":               map[string]any{"agent": "./graphs/agent.ui.jsx"},
		"ui_config":        map[string]any{"shared": []any{"nuqs"}},
	})

	dockerfile, contexts, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraphjs-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "FROM langchain/langgraphjs-api:20")
	assertContains(t, dockerfile, "ARG meow")
	assertContains(t, dockerfile, "ARG foo")
	assertContains(t, dockerfile, "RUN npm i")
	assertContains(t, dockerfile, "LANGGRAPH_AUTH=")
	assertContains(t, dockerfile, "LANGGRAPH_UI=")
	assertContains(t, dockerfile, "LANGGRAPH_UI_CONFIG=")
	assertContains(t, dockerfile, "LANGSERVE_GRAPHS=")
	assertContains(t, dockerfile, "tsx /api/langgraph_api/js/build.mts")

	if len(contexts) != 0 {
		t.Fatalf("expected 0 additional contexts, got %d", len(contexts))
	}
}

func TestConfigToDockerNodeJSInternalTag(t *testing.T) {
	dir := t.TempDir()
	configPath := setupNodeProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"node_version":         "20",
		"graphs":               map[string]any{"agent": "./graphs/agent.js:graph"},
		"_INTERNAL_docker_tag": "my-tag",
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraphjs-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "FROM langchain/langgraphjs-api:my-tag")
}

func TestConfigToDockerMultiplatform(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "multiplatform", "python.py"), "graph = None\n")
	writeFile(t, filepath.Join(dir, "multiplatform", "js.mts"), "export const graph = {};\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	cfg := mustValidate(t, map[string]any{
		"node_version": "22",
		"dependencies": []any{"."},
		"graphs": map[string]any{
			"python": "./multiplatform/python.py:graph",
			"js":     "./multiplatform/js.mts:graph",
		},
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	// Multiplatform with both Python and Node -> Python base image
	assertContains(t, dockerfile, "FROM langchain/langgraph-api:3.11")
	// Should install node for JS
	assertContains(t, dockerfile, "RUN /storage/install-node.sh")
	assertContains(t, dockerfile, "ENV NODE_VERSION=22")
	assertContains(t, dockerfile, "npm i && tsx /api/langgraph_api/js/build.mts")
	assertContains(t, dockerfile, "LANGSERVE_GRAPHS=")
}

func TestConfigToDockerPipInstaller(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "graphs", "agent.py"), "graph = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	baseCfg := func() map[string]any {
		return map[string]any{
			"python_version": "3.11",
			"dependencies":   []any{"."},
			"graphs":         map[string]any{"agent": "./graphs/agent.py:graph"},
		}
	}

	t.Run("auto_with_uv_supporting_image", func(t *testing.T) {
		raw := baseCfg()
		raw["pip_installer"] = "auto"
		cfg := mustValidate(t, raw)
		dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage: "langchain/langgraph-api:0.2.47",
		})
		if err != nil {
			t.Fatalf("ConfigToDocker failed: %v", err)
		}
		assertContains(t, dockerfile, "uv pip install --system")
		assertContains(t, dockerfile, "rm /usr/bin/uv /usr/bin/uvx")
	})

	t.Run("explicit_pip", func(t *testing.T) {
		raw := baseCfg()
		raw["pip_installer"] = "pip"
		cfg := mustValidate(t, raw)
		dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage: "langchain/langgraph-api:0.2.47",
		})
		if err != nil {
			t.Fatalf("ConfigToDocker failed: %v", err)
		}
		assertNotContains(t, dockerfile, "uv pip install --system")
		assertContains(t, dockerfile, "pip install")
		assertNotContains(t, dockerfile, "rm /usr/bin/uv")
	})

	t.Run("explicit_uv", func(t *testing.T) {
		raw := baseCfg()
		raw["pip_installer"] = "uv"
		cfg := mustValidate(t, raw)
		dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage: "langchain/langgraph-api:0.2.47",
		})
		if err != nil {
			t.Fatalf("ConfigToDocker failed: %v", err)
		}
		assertContains(t, dockerfile, "uv pip install --system")
		assertContains(t, dockerfile, "rm /usr/bin/uv /usr/bin/uvx")
	})

	t.Run("auto_with_old_image_uses_pip", func(t *testing.T) {
		raw := baseCfg()
		raw["pip_installer"] = "auto"
		cfg := mustValidate(t, raw)
		dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage: "langchain/langgraph-api:0.2.46",
		})
		if err != nil {
			t.Fatalf("ConfigToDocker failed: %v", err)
		}
		assertNotContains(t, dockerfile, "uv pip install --system")
		assertContains(t, dockerfile, "pip install")
		assertNotContains(t, dockerfile, "rm /usr/bin/uv")
	})

	t.Run("default_auto_with_uv_image", func(t *testing.T) {
		cfg := mustValidate(t, baseCfg())
		dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage: "langchain/langgraph-api:0.2.47",
		})
		if err != nil {
			t.Fatalf("ConfigToDocker failed: %v", err)
		}
		assertContains(t, dockerfile, "uv pip install --system")
	})
}

func TestConfigToDockerWebhooksPython(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	webhooks := map[string]any{
		"env_prefix": "LG_WEBHOOK_",
		"url": map[string]any{
			"require_https":    true,
			"allowed_domains":  []any{"hooks.example.com", "*.example.org"},
			"allowed_ports":    []any{float64(443)},
			"max_url_length":   float64(1024),
			"disable_loopback": false,
		},
		"headers": map[string]any{
			"x-auth":  "${{ env.LG_WEBHOOK_TOKEN }}",
			"x-mixed": "Bearer ${{ env.LG_WEBHOOK_TOKEN }}-suffix",
		},
	}

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
		"webhooks":     webhooks,
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	parsed := extractEnvJSON(t, dockerfile, "LANGGRAPH_WEBHOOKS")
	if parsed["env_prefix"] != "LG_WEBHOOK_" {
		t.Fatalf("expected env_prefix LG_WEBHOOK_, got %v", parsed["env_prefix"])
	}
}

func TestConfigToDockerNoWebhooks(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertNotContains(t, dockerfile, "ENV LANGGRAPH_WEBHOOKS=")
}

func TestConfigToDockerEncryption(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "agent.py"), "graph = None\n")
	writeFile(t, filepath.Join(dir, "encryption.py"), "encryption = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	// Test encryption config is preserved after validation
	validated := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"graphs":         map[string]any{"agent": "./agent.py:graph"},
		"dependencies":   []any{"."},
		"encryption":     map[string]any{"path": "./encryption.py:encryption"},
	})

	enc, ok := validated["encryption"].(map[string]any)
	if !ok || enc == nil {
		t.Fatal("encryption config should be preserved after validation")
	}
	if enc["path"] != "./encryption.py:encryption" {
		t.Fatalf("expected encryption path ./encryption.py:encryption, got %v", enc["path"])
	}
}

func TestConfigToDockerEncryptionFormatted(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "agent.py"), "my_encryption = None\n")
	writeFile(t, filepath.Join(dir, "graphs", "agent.py"), "graph = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.11",
		"dependencies":   []any{"."},
		"graphs":         map[string]any{"agent": "./graphs/agent.py:graph"},
		"encryption":     map[string]any{"path": "./agent.py:my_encryption"},
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "LANGGRAPH_ENCRYPTION=")
	assertContains(t, dockerfile, "agent.py:my_encryption")
}

func TestConfigToDockerWithAPIVersion(t *testing.T) {
	t.Run("python", func(t *testing.T) {
		dir := t.TempDir()
		configPath := setupSimplePythonProject(t, dir)

		cfg := mustValidate(t, map[string]any{
			"dependencies": []any{"."},
			"graphs":       map[string]any{"agent": "./agent.py:graph"},
		})

		dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage:  "langchain/langgraph-api",
			APIVersion: "0.2.74",
		})
		if err != nil {
			t.Fatalf("ConfigToDocker failed: %v", err)
		}

		lines := strings.Split(dockerfile, "\n")
		if !strings.Contains(lines[0], "FROM langchain/langgraph-api:0.2.74-py3.11") {
			t.Fatalf("expected FROM line with api version, got: %s", lines[0])
		}
	})

	t.Run("nodejs", func(t *testing.T) {
		dir := t.TempDir()
		configPath := setupNodeProject(t, dir)

		cfg := mustValidate(t, map[string]any{
			"node_version": "20",
			"graphs":       map[string]any{"agent": "./graphs/agent.js:graph"},
		})

		dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage:  "langchain/langgraphjs-api",
			APIVersion: "0.2.74",
		})
		if err != nil {
			t.Fatalf("ConfigToDocker failed: %v", err)
		}

		lines := strings.Split(dockerfile, "\n")
		fromLine := ""
		for _, l := range lines {
			if strings.HasPrefix(strings.TrimSpace(l), "FROM ") {
				fromLine = strings.TrimSpace(l)
				break
			}
		}
		if fromLine != "FROM langchain/langgraphjs-api:0.2.74-node20" {
			t.Fatalf("expected FROM langchain/langgraphjs-api:0.2.74-node20, got %q", fromLine)
		}
	})
}

func TestConfigToDockerExecutorBaseImage(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-executor",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "FROM langchain/langgraph-executor:3.11")
	assertContains(t, dockerfile, "LANGSERVE_GRAPHS=")
}

// ---------------------------------------------------------------------------
// ConfigToDocker -- retain/remove build tools
// ---------------------------------------------------------------------------

func TestConfigRetainBuildTools(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "graphs", "agent.py"), "graph = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	baseCfg := func() map[string]any {
		return map[string]any{
			"python_version": "3.11",
			"dependencies":   []any{"."},
			"graphs":         map[string]any{"agent": "./graphs/agent.py:graph"},
		}
	}

	t.Run("keep_pkg_tools_true", func(t *testing.T) {
		raw := baseCfg()
		raw["keep_pkg_tools"] = true
		cfg := mustValidate(t, raw)
		dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage: "langchain/langgraph-api:0.2.47",
		})
		if err != nil {
			t.Fatalf("ConfigToDocker failed: %v", err)
		}

		for _, pkg := range []string{"pip", "setuptools", "wheel"} {
			assertNotContains(t, dockerfile, "/usr/local/lib/python*/site-packages/"+pkg+"*")
		}
		assertNotContains(t, dockerfile, "RUN pip uninstall -y pip setuptools wheel")
	})

	t.Run("keep_pkg_tools_false", func(t *testing.T) {
		raw := baseCfg()
		raw["keep_pkg_tools"] = false
		cfg := mustValidate(t, raw)
		dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage: "langchain/langgraph-api:0.2.47",
		})
		if err != nil {
			t.Fatalf("ConfigToDocker failed: %v", err)
		}

		for _, pkg := range []string{"pip", "setuptools", "wheel"} {
			assertContains(t, dockerfile, "/usr/local/lib/python*/site-packages/"+pkg+"*")
		}
		assertContains(t, dockerfile, "RUN pip uninstall -y pip setuptools wheel")
	})

	t.Run("keep_pkg_tools_list", func(t *testing.T) {
		raw := baseCfg()
		raw["keep_pkg_tools"] = []any{"pip", "setuptools"}
		cfg := mustValidate(t, raw)
		dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage: "langchain/langgraph-api:0.2.47",
		})
		if err != nil {
			t.Fatalf("ConfigToDocker failed: %v", err)
		}

		assertContains(t, dockerfile, "/usr/local/lib/python*/site-packages/wheel*")
		assertNotContains(t, dockerfile, "/usr/local/lib/python*/site-packages/pip*")
		assertNotContains(t, dockerfile, "/usr/local/lib/python*/site-packages/setuptools*")
		assertContains(t, dockerfile, "RUN pip uninstall -y wheel")
		assertNotContains(t, dockerfile, "RUN pip uninstall -y pip setuptools")
	})
}

// ---------------------------------------------------------------------------
// ConfigToDocker -- PyPI dependencies
// ---------------------------------------------------------------------------

func TestConfigToDockerPyPIDeps(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "graphs", "agent.py"), "graph = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	cfg := mustValidate(t, map[string]any{
		"python_version": "3.12",
		"dependencies":   []any{"./graphs/", "langchain", "langchain_openai"},
		"graphs":         map[string]any{"agent": "./graphs/agent.py:graph"},
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "FROM langchain/langgraph-api:3.12")
	assertContains(t, dockerfile, "langchain langchain_openai")
}

// ---------------------------------------------------------------------------
// ConfigToDocker -- dockerfile_lines
// ---------------------------------------------------------------------------

func TestConfigToDockerDockerfileLines(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "graphs", "agent.py"), "graph = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	cfg := mustValidate(t, map[string]any{
		"python_version":   "3.12",
		"dependencies":     []any{"./graphs/"},
		"graphs":           map[string]any{"agent": "./graphs/agent.py:graph"},
		"dockerfile_lines": []any{"ARG meow", "ARG foo"},
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "ARG meow")
	assertContains(t, dockerfile, "ARG foo")
}

// ---------------------------------------------------------------------------
// ConfigToDocker -- gen UI with Python (installs node)
// ---------------------------------------------------------------------------

func TestConfigToDockerGenUIPython(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "agent.py"), "graph = None\n")
	writeFile(t, filepath.Join(dir, "graphs", "agent.ui.jsx"), "export default null;\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
		"ui":           map[string]any{"agent": "./graphs/agent.ui.jsx"},
		"ui_config":    map[string]any{"shared": []any{"nuqs"}},
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "FROM langchain/langgraph-api:3.11")
	assertContains(t, dockerfile, "RUN /storage/install-node.sh")
	assertContains(t, dockerfile, "LANGGRAPH_UI=")
	assertContains(t, dockerfile, "LANGGRAPH_UI_CONFIG=")
	assertContains(t, dockerfile, "npm i && tsx /api/langgraph_api/js/build.mts")
	assertContains(t, dockerfile, "ENV NODE_VERSION=20")
}

// ---------------------------------------------------------------------------
// ConfigToCompose tests
// ---------------------------------------------------------------------------

func TestConfigToComposeSimple(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	})

	compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToCompose failed: %v", err)
	}

	assertContains(t, compose, "pull_policy: build")
	assertContains(t, compose, "dockerfile_inline:")
	assertContains(t, compose, "FROM langchain/langgraph-api:3.11")
	assertContains(t, compose, "LANGSERVE_GRAPHS=")
	assertContains(t, compose, "context: .")
	// Should use escaped variable names for compose
	assertContains(t, compose, "$$dep")
}

func TestConfigToComposeEnvVars(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
		"env":          map[string]any{"OPENAI_API_KEY": "key"},
	})

	compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
		BaseImage: "langchain/langgraph-api-custom",
	})
	if err != nil {
		t.Fatalf("ConfigToCompose failed: %v", err)
	}

	assertContains(t, compose, `OPENAI_API_KEY: "key"`)
	assertContains(t, compose, "FROM langchain/langgraph-api-custom:3.11")
}

func TestConfigToComposeEnvFile(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
		"env":          ".env",
	})

	compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToCompose failed: %v", err)
	}

	assertContains(t, compose, "env_file: .env")
}

func TestConfigToComposeWithAPIVersion(t *testing.T) {
	t.Run("python", func(t *testing.T) {
		dir := t.TempDir()
		configPath := setupSimplePythonProject(t, dir)

		cfg := mustValidate(t, map[string]any{
			"dependencies": []any{"."},
			"graphs":       map[string]any{"agent": "./agent.py:graph"},
		})

		compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
			BaseImage:  "langchain/langgraph-api",
			APIVersion: "0.2.74",
		})
		if err != nil {
			t.Fatalf("ConfigToCompose failed: %v", err)
		}

		assertContains(t, compose, "FROM langchain/langgraph-api:0.2.74-py3.11")
	})

	t.Run("nodejs", func(t *testing.T) {
		dir := t.TempDir()
		configPath := setupNodeProject(t, dir)

		cfg := mustValidate(t, map[string]any{
			"node_version": "20",
			"graphs":       map[string]any{"agent": "./graphs/agent.js:graph"},
		})

		compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
			BaseImage:  "langchain/langgraphjs-api",
			APIVersion: "0.2.74",
		})
		if err != nil {
			t.Fatalf("ConfigToCompose failed: %v", err)
		}

		assertContains(t, compose, "FROM langchain/langgraphjs-api:0.2.74-node20")
	})
}

func TestConfigToComposeDistributedMode(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	})

	compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
		BaseImage:         "langchain/langgraph-api",
		EngineRuntimeMode: "distributed",
	})
	if err != nil {
		t.Fatalf("ConfigToCompose failed: %v", err)
	}

	assertContains(t, compose, "FROM langchain/langgraph-api:3.11")
	assertContains(t, compose, "langgraph-orchestrator:")
	assertContains(t, compose, "EXECUTOR_TARGET: langgraph-executor:8188")
	assertContains(t, compose, "langgraph-executor:")
	assertContains(t, compose, "FROM langchain/langgraph-executor:3.11")
	assertContains(t, compose, `entrypoint: ["sh", "/storage/executor_entrypoint.sh"]`)
	assertContains(t, compose, "EXECUTOR_GRPC_PORT:")
	assertContains(t, compose, "ENGINE_GRPC_ADDRESS:")
	assertContains(t, compose, "LSD_GRPC_SERVER_ADDRESS:")
	assertContains(t, compose, `LANGGRAPH_HTTP: ""`)
	assertContains(t, compose, "REDIS_URI: redis://langgraph-redis:6379")
}

func TestConfigToComposeDistributedModeWithEnvFile(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
		"env":          ".env",
	})

	compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
		BaseImage:         "langchain/langgraph-api",
		EngineRuntimeMode: "distributed",
	})
	if err != nil {
		t.Fatalf("ConfigToCompose failed: %v", err)
	}

	count := strings.Count(compose, "env_file: .env")
	if count != 3 {
		t.Fatalf("expected env_file to appear 3 times (api, orchestrator, executor), got %d", count)
	}
}

func TestConfigToComposeDistributedTwoDockerfiles(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	})

	compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
		BaseImage:         "langchain/langgraph-api",
		EngineRuntimeMode: "distributed",
	})
	if err != nil {
		t.Fatalf("ConfigToCompose failed: %v", err)
	}

	var fromLines []string
	for _, line := range strings.Split(compose, "\n") {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "FROM ") {
			fromLines = append(fromLines, trimmed)
		}
	}
	if len(fromLines) != 2 {
		t.Fatalf("expected 2 FROM lines, got %d: %v", len(fromLines), fromLines)
	}
	assertContains(t, fromLines[0], "FROM langchain/langgraph-api:3.11")
	assertContains(t, fromLines[1], "FROM langchain/langgraph-executor:3.11")
}

func TestConfigToComposeCombinedModeNoOrchestrator(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	})

	compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
		BaseImage:         "langchain/langgraph-api",
		EngineRuntimeMode: "combined_queue_worker",
	})
	if err != nil {
		t.Fatalf("ConfigToCompose failed: %v", err)
	}

	assertNotContains(t, compose, "langgraph-orchestrator:")
	assertNotContains(t, compose, "langgraph-executor:")
}

func TestConfigToComposeDefaultModeNoOrchestrator(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	})

	compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToCompose failed: %v", err)
	}

	assertNotContains(t, compose, "langgraph-orchestrator:")
	assertNotContains(t, compose, "langgraph-executor:")
}

func TestConfigToComposeDistributedCorrectPaths(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	})

	compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
		BaseImage:         "langchain/langgraph-api",
		EngineRuntimeMode: "distributed",
	})
	if err != nil {
		t.Fatalf("ConfigToCompose failed: %v", err)
	}

	// Both API and executor should contain LANGSERVE_GRAPHS
	count := 0
	for _, line := range strings.Split(compose, "\n") {
		if strings.Contains(strings.TrimSpace(line), "LANGSERVE_GRAPHS=") {
			count++
		}
	}
	if count != 2 {
		t.Fatalf("expected 2 LANGSERVE_GRAPHS lines, got %d", count)
	}
}

// ---------------------------------------------------------------------------
// ConfigToCompose -- watch mode
// ---------------------------------------------------------------------------

func TestConfigToComposeWatch(t *testing.T) {
	dir := t.TempDir()
	configPath := setupSimplePythonProject(t, dir)

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"."},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	})

	compose, err := ConfigToCompose(configPath, cfg, ComposeOpts{
		BaseImage: "langchain/langgraph-api",
		Watch:     true,
	})
	if err != nil {
		t.Fatalf("ConfigToCompose failed: %v", err)
	}

	assertContains(t, compose, "develop:")
	assertContains(t, compose, "watch:")
	assertContains(t, compose, "action: rebuild")
}

// ---------------------------------------------------------------------------
// AssembleLocalDeps tests
// ---------------------------------------------------------------------------

func TestAssembleLocalDeps(t *testing.T) {
	t.Run("faux_package_flat_layout", func(t *testing.T) {
		dir := t.TempDir()
		// Create a flat-layout faux package (directory with __init__.py)
		writeFile(t, filepath.Join(dir, "mypkg", "__init__.py"), "")
		writeFile(t, filepath.Join(dir, "mypkg", "graph.py"), "graph = None\n")
		configPath := filepath.Join(dir, "langgraph.json")
		writeFile(t, configPath, "{}\n")

		cfg := map[string]any{
			"dependencies": []any{"./mypkg"},
			"graphs":       map[string]any{"agent": "./mypkg/graph.py:graph"},
		}

		deps, err := AssembleLocalDeps(configPath, cfg)
		if err != nil {
			t.Fatalf("AssembleLocalDeps failed: %v", err)
		}

		if len(deps.FauxPkgs) != 1 {
			t.Fatalf("expected 1 faux package, got %d", len(deps.FauxPkgs))
		}
		for _, faux := range deps.FauxPkgs {
			assertContains(t, faux.ContainerPath, "/deps/outer-mypkg/mypkg")
		}
	})

	t.Run("real_package_pyproject", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, filepath.Join(dir, "mypkg", "pyproject.toml"), `[project]
name = "test"
version = "0.1"`)
		writeFile(t, filepath.Join(dir, "mypkg", "graph.py"), "graph = None\n")
		configPath := filepath.Join(dir, "langgraph.json")
		writeFile(t, configPath, "{}\n")

		cfg := map[string]any{
			"dependencies": []any{"./mypkg"},
			"graphs":       map[string]any{"agent": "./mypkg/graph.py:graph"},
		}

		deps, err := AssembleLocalDeps(configPath, cfg)
		if err != nil {
			t.Fatalf("AssembleLocalDeps failed: %v", err)
		}

		if len(deps.RealPkgs) != 1 {
			t.Fatalf("expected 1 real package, got %d", len(deps.RealPkgs))
		}
	})

	t.Run("real_package_setup_py", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, filepath.Join(dir, "mypkg", "setup.py"), "from setuptools import setup; setup(name='test')")
		writeFile(t, filepath.Join(dir, "mypkg", "graph.py"), "graph = None\n")
		configPath := filepath.Join(dir, "langgraph.json")
		writeFile(t, configPath, "{}\n")

		cfg := map[string]any{
			"dependencies": []any{"./mypkg"},
			"graphs":       map[string]any{"agent": "./mypkg/graph.py:graph"},
		}

		deps, err := AssembleLocalDeps(configPath, cfg)
		if err != nil {
			t.Fatalf("AssembleLocalDeps failed: %v", err)
		}

		if len(deps.RealPkgs) != 1 {
			t.Fatalf("expected 1 real package, got %d", len(deps.RealPkgs))
		}
	})

	t.Run("dot_dependency_sets_workdir", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, filepath.Join(dir, "pyproject.toml"), `[project]
name = "myapp"
version = "0.1"`)
		writeFile(t, filepath.Join(dir, "graph.py"), "graph = None\n")
		configPath := filepath.Join(dir, "langgraph.json")
		writeFile(t, configPath, "{}\n")

		cfg := map[string]any{
			"dependencies": []any{"."},
			"graphs":       map[string]any{"agent": "./graph.py:graph"},
		}

		deps, err := AssembleLocalDeps(configPath, cfg)
		if err != nil {
			t.Fatalf("AssembleLocalDeps failed: %v", err)
		}

		baseName := filepath.Base(dir)
		expected := "/deps/" + baseName
		if deps.WorkingDir != expected {
			t.Fatalf("expected WorkingDir %q, got %q", expected, deps.WorkingDir)
		}
	})

	t.Run("missing_dependency_dir", func(t *testing.T) {
		dir := t.TempDir()
		configPath := filepath.Join(dir, "langgraph.json")
		writeFile(t, configPath, "{}\n")

		cfg := map[string]any{
			"dependencies": []any{"./missing"},
			"graphs":       map[string]any{"agent": "./agent.py:graph"},
		}

		_, err := AssembleLocalDeps(configPath, cfg)
		if err == nil {
			t.Fatal("expected error for missing dependency, got nil")
		}
		assertContains(t, err.Error(), "Could not find local dependency")
	})

	t.Run("requirements_txt_detected", func(t *testing.T) {
		dir := t.TempDir()
		writeFile(t, filepath.Join(dir, "mypkg", "requirements.txt"), "langchain>=0.1\n")
		writeFile(t, filepath.Join(dir, "mypkg", "graph.py"), "graph = None\n")
		configPath := filepath.Join(dir, "langgraph.json")
		writeFile(t, configPath, "{}\n")

		cfg := map[string]any{
			"dependencies": []any{"./mypkg"},
			"graphs":       map[string]any{"agent": "./mypkg/graph.py:graph"},
		}

		deps, err := AssembleLocalDeps(configPath, cfg)
		if err != nil {
			t.Fatalf("AssembleLocalDeps failed: %v", err)
		}

		if len(deps.PipReqs) != 1 {
			t.Fatalf("expected 1 pip req, got %d", len(deps.PipReqs))
		}
	})
}

// ---------------------------------------------------------------------------
// ConfigToDocker -- invalid inputs
// ---------------------------------------------------------------------------

func TestConfigToDockerInvalidInputs(t *testing.T) {
	t.Run("missing_local_dependency", func(t *testing.T) {
		dir := t.TempDir()
		configPath := filepath.Join(dir, "langgraph.json")
		writeFile(t, configPath, "{}\n")

		cfg := mustValidate(t, map[string]any{
			"dependencies": []any{"./missing"},
			"graphs":       map[string]any{"agent": "./agent.py:graph"},
		})

		_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage: "langchain/langgraph-api",
		})
		if err == nil {
			t.Fatal("expected error for missing dependency")
		}
	})

	t.Run("missing_local_module", func(t *testing.T) {
		dir := t.TempDir()
		configPath := setupSimplePythonProject(t, dir)

		cfg := mustValidate(t, map[string]any{
			"dependencies": []any{"."},
			"graphs":       map[string]any{"agent": "./missing_agent.py:graph"},
		})

		_, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
			BaseImage: "langchain/langgraph-api",
		})
		if err == nil {
			t.Fatal("expected error for missing module")
		}
	})
}

// ---------------------------------------------------------------------------
// ConfigToDocker -- requirements.txt with faux package
// ---------------------------------------------------------------------------

func TestConfigToDockerWithRequirements(t *testing.T) {
	dir := t.TempDir()
	writeFile(t, filepath.Join(dir, "mypkg", "requirements.txt"), "langchain>=0.1\n")
	writeFile(t, filepath.Join(dir, "mypkg", "subpkg", "agent.py"), "graph = None\n")
	configPath := filepath.Join(dir, "langgraph.json")
	writeFile(t, configPath, "{}\n")

	cfg := mustValidate(t, map[string]any{
		"dependencies": []any{"./mypkg"},
		"graphs":       map[string]any{"agent": "./mypkg/subpkg/agent.py:graph"},
	})

	dockerfile, _, err := ConfigToDocker(configPath, cfg, DockerOpts{
		BaseImage: "langchain/langgraph-api",
	})
	if err != nil {
		t.Fatalf("ConfigToDocker failed: %v", err)
	}

	assertContains(t, dockerfile, "Installing local requirements")
	assertContains(t, dockerfile, "requirements.txt")
}
