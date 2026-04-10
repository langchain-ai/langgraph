package config

import (
	"strings"
	"testing"
)

// baseConfig returns a minimal valid config map. Tests should copy and modify it.
func baseConfig() map[string]any {
	return map[string]any{
		"dependencies": []any{"langchain"},
		"graphs":       map[string]any{"agent": "./agent.py:graph"},
	}
}

// copyMap returns a shallow copy of m.
func copyMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

// mustSucceed is a test helper that fails if err is non-nil.
func mustSucceed(t *testing.T, err error) {
	t.Helper()
	if err != nil {
		t.Fatalf("expected success but got error: %v", err)
	}
}

// mustFail is a test helper that fails if err is nil.
func mustFail(t *testing.T, err error) {
	t.Helper()
	if err == nil {
		t.Fatal("expected error but got nil")
	}
}

// mustContain checks that err is non-nil and its message contains substr.
func mustContain(t *testing.T, err error, substr string) {
	t.Helper()
	if err == nil {
		t.Fatalf("expected error containing %q but got nil", substr)
	}
	if !strings.Contains(err.Error(), substr) {
		t.Fatalf("expected error to contain %q, got: %s", substr, err.Error())
	}
}

func TestValidateConfigValid(t *testing.T) {
	t.Run("minimal config", func(t *testing.T) {
		raw := baseConfig()
		result, err := ValidateConfig(raw)
		mustSucceed(t, err)

		if pv, _ := result["python_version"].(string); pv != "3.11" {
			t.Fatalf("expected python_version '3.11', got %q", pv)
		}
		if id, _ := result["image_distro"].(string); id != "debian" {
			t.Fatalf("expected image_distro 'debian', got %q", id)
		}
	})

	t.Run("full config with all optional fields", func(t *testing.T) {
		raw := map[string]any{
			"python_version":   "3.12",
			"image_distro":     "wolfi",
			"pip_installer":    "uv",
			"dependencies":     []any{"langchain", "langgraph"},
			"graphs":           map[string]any{"agent": "./agent.py:graph"},
			"env":              map[string]any{"FOO": "bar"},
			"dockerfile_lines": []any{"RUN apt-get update"},
			"auth":             map[string]any{"path": "./auth.py:handler"},
			"encryption":       map[string]any{"path": "./enc.py:enc"},
			"http":             map[string]any{"app": "./app.py:app"},
			"keep_pkg_tools":   true,
			"api_version":      "0.8",
		}
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})
}

func TestValidateConfigPythonVersion(t *testing.T) {
	validVersions := []string{"3.11", "3.12", "3.13"}
	for _, v := range validVersions {
		t.Run("valid "+v, func(t *testing.T) {
			raw := copyMap(baseConfig())
			raw["python_version"] = v
			_, err := ValidateConfig(raw)
			mustSucceed(t, err)
		})
	}

	t.Run("valid 3.12-slim suffix stripped", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["python_version"] = "3.12-slim"
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	tooOld := []string{"3.10", "3.9"}
	for _, v := range tooOld {
		t.Run("too old "+v, func(t *testing.T) {
			raw := copyMap(baseConfig())
			raw["python_version"] = v
			_, err := ValidateConfig(raw)
			mustContain(t, err, "Minimum required version")
		})
	}

	badFormat := []struct {
		version string
	}{
		{"3.11.0"},
		{"3"},
		{"abc.def"},
	}
	for _, tc := range badFormat {
		t.Run("bad format "+tc.version, func(t *testing.T) {
			raw := copyMap(baseConfig())
			raw["python_version"] = tc.version
			_, err := ValidateConfig(raw)
			mustContain(t, err, "Invalid Python version format")
		})
	}
}

func TestValidateConfigNodeVersion(t *testing.T) {
	// Need a node graph to trigger node_version validation
	nodeBase := func() map[string]any {
		return map[string]any{
			"dependencies": []any{"langchain"},
			"graphs":       map[string]any{"agent": "./agent.py:graph"},
		}
	}

	t.Run("valid 20", func(t *testing.T) {
		raw := nodeBase()
		raw["node_version"] = "20"
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	t.Run("valid 22", func(t *testing.T) {
		raw := nodeBase()
		raw["node_version"] = "22"
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	t.Run("too old 18", func(t *testing.T) {
		raw := nodeBase()
		raw["node_version"] = "18"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Minimum required version is 20")
	})

	t.Run("minor version 20.1", func(t *testing.T) {
		raw := nodeBase()
		raw["node_version"] = "20.1"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "major version only")
	})
}

func TestValidateConfigGraphs(t *testing.T) {
	t.Run("empty graphs", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["graphs"] = map[string]any{}
		_, err := ValidateConfig(raw)
		mustContain(t, err, "No graphs found")
	})

	t.Run("missing graphs key", func(t *testing.T) {
		raw := map[string]any{
			"dependencies": []any{"langchain"},
		}
		_, err := ValidateConfig(raw)
		mustContain(t, err, "No graphs found")
	})
}

func TestValidateConfigImageDistro(t *testing.T) {
	validDistroTests := []string{"debian", "wolfi", "bookworm"}
	for _, d := range validDistroTests {
		t.Run("valid "+d, func(t *testing.T) {
			raw := copyMap(baseConfig())
			raw["image_distro"] = d
			_, err := ValidateConfig(raw)
			mustSucceed(t, err)
		})
	}

	t.Run("bullseye deprecated", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["image_distro"] = "bullseye"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "deprecated")
	})

	t.Run("invalid ubuntu", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["image_distro"] = "ubuntu"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Invalid image_distro")
	})

	t.Run("invalid alpine", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["image_distro"] = "alpine"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Invalid image_distro")
	})

	t.Run("default is debian", func(t *testing.T) {
		raw := baseConfig()
		// no image_distro key
		result, err := ValidateConfig(raw)
		mustSucceed(t, err)
		if id, _ := result["image_distro"].(string); id != "debian" {
			t.Fatalf("expected default image_distro 'debian', got %q", id)
		}
	})
}

func TestValidateConfigPipInstaller(t *testing.T) {
	valid := []string{"auto", "pip", "uv"}
	for _, pi := range valid {
		t.Run("valid "+pi, func(t *testing.T) {
			raw := copyMap(baseConfig())
			raw["pip_installer"] = pi
			_, err := ValidateConfig(raw)
			mustSucceed(t, err)
		})
	}

	invalid := []string{"conda", "uv_lock"}
	for _, pi := range invalid {
		t.Run("invalid "+pi, func(t *testing.T) {
			raw := copyMap(baseConfig())
			raw["pip_installer"] = pi
			_, err := ValidateConfig(raw)
			mustContain(t, err, "Invalid pip_installer")
		})
	}
}

func TestValidateConfigSource(t *testing.T) {
	t.Run("valid uv source with root", func(t *testing.T) {
		raw := map[string]any{
			"python_version": "3.12",
			"graphs":         map[string]any{"agent": "./agent.py:graph"},
			"source":         map[string]any{"kind": "uv", "root": "../.."},
		}
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	t.Run("invalid source kind poetry", func(t *testing.T) {
		raw := map[string]any{
			"python_version": "3.12",
			"graphs":         map[string]any{"agent": "./agent.py:graph"},
			"source":         map[string]any{"kind": "poetry"},
		}
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Invalid source.kind")
	})

	t.Run("source as string not object", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["source"] = "not-an-object"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "`source` must be an object")
	})

	t.Run("uv source with dependencies", func(t *testing.T) {
		raw := map[string]any{
			"python_version": "3.12",
			"graphs":         map[string]any{"agent": "./agent.py:graph"},
			"source":         map[string]any{"kind": "uv", "root": ".."},
			"dependencies":   []any{"langchain"},
		}
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Remove `dependencies`")
	})

	t.Run("uv source with root as number", func(t *testing.T) {
		raw := map[string]any{
			"python_version": "3.12",
			"graphs":         map[string]any{"agent": "./agent.py:graph"},
			"source":         map[string]any{"kind": "uv", "root": 123},
		}
		_, err := ValidateConfig(raw)
		mustContain(t, err, "source.root` must be a string")
	})

	t.Run("uv source with package as number", func(t *testing.T) {
		raw := map[string]any{
			"python_version": "3.12",
			"graphs":         map[string]any{"agent": "./agent.py:graph"},
			"source":         map[string]any{"kind": "uv", "root": "..", "package": 123},
		}
		_, err := ValidateConfig(raw)
		mustContain(t, err, "source.package` must be a non-empty string")
	})
}

func TestValidateConfigAPIVersion(t *testing.T) {
	t.Run("valid 0.8", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["api_version"] = "0.8"
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	t.Run("valid 0.8.1", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["api_version"] = "0.8.1"
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	t.Run("invalid abc", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["api_version"] = "abc"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Invalid version format")
	})

	t.Run("invalid 1.2.3.4 too many parts", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["api_version"] = "1.2.3.4"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "major or major.minor")
	})
}

func TestValidateConfigMutualExclusion(t *testing.T) {
	t.Run("both _INTERNAL_docker_tag and api_version", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["_INTERNAL_docker_tag"] = "some-tag"
		raw["api_version"] = "0.8"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Cannot specify both")
	})
}

func TestValidateConfigAuthPath(t *testing.T) {
	t.Run("valid auth path with colon", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["auth"] = map[string]any{"path": "./auth.py:handler"}
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	t.Run("invalid auth path without colon", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["auth"] = map[string]any{"path": "../../examples/my_app.py"}
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Invalid auth.path format")
	})
}

func TestValidateConfigEncryptionPath(t *testing.T) {
	t.Run("valid encryption path with colon", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["encryption"] = map[string]any{"path": "./enc.py:enc"}
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	t.Run("invalid encryption path without colon", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["encryption"] = map[string]any{"path": "./enc.py"}
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Invalid encryption.path format")
	})
}

func TestValidateConfigHTTPApp(t *testing.T) {
	t.Run("valid http app with colon", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["http"] = map[string]any{"app": "./app.py:app"}
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	t.Run("invalid http app without colon", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["http"] = map[string]any{"app": "./app.py"}
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Invalid http.app format")
	})
}

func TestValidateConfigKeepPkgTools(t *testing.T) {
	t.Run("bool true", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["keep_pkg_tools"] = true
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	t.Run("valid list", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["keep_pkg_tools"] = []any{"pip", "wheel"}
		_, err := ValidateConfig(raw)
		mustSucceed(t, err)
	})

	t.Run("invalid list item", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["keep_pkg_tools"] = []any{"invalid"}
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Invalid keep_pkg_tools")
	})

	t.Run("invalid string type", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["keep_pkg_tools"] = "string"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "Invalid keep_pkg_tools")
	})
}

func TestValidateConfigLegacyKeys(t *testing.T) {
	t.Run("project_root legacy", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["project_root"] = ".."
		_, err := ValidateConfig(raw)
		mustContain(t, err, "no longer supported")
	})

	t.Run("package legacy", func(t *testing.T) {
		raw := copyMap(baseConfig())
		raw["package"] = "foo"
		_, err := ValidateConfig(raw)
		mustContain(t, err, "no longer supported")
	})
}

func TestValidateConfigNodeGraphDetection(t *testing.T) {
	t.Run("ts extension auto-sets node_version", func(t *testing.T) {
		raw := map[string]any{
			"dependencies": []any{"langchain"},
			"graphs":       map[string]any{"agent": "./agent.py:graph", "bot": "./bot.ts:bot"},
		}
		result, err := ValidateConfig(raw)
		mustSucceed(t, err)
		if nv, _ := result["node_version"].(string); nv != "20" {
			t.Fatalf("expected node_version '20', got %q", nv)
		}
	})

	t.Run("js extension auto-sets node_version", func(t *testing.T) {
		raw := map[string]any{
			"dependencies": []any{"langchain"},
			"graphs":       map[string]any{"agent": "./agent.py:graph", "bot": "./bot.js:bot"},
		}
		result, err := ValidateConfig(raw)
		mustSucceed(t, err)
		if nv, _ := result["node_version"].(string); nv != "20" {
			t.Fatalf("expected node_version '20', got %q", nv)
		}
	})

	t.Run("py extension does not set node_version", func(t *testing.T) {
		raw := baseConfig()
		result, err := ValidateConfig(raw)
		mustSucceed(t, err)
		if nv, _ := result["node_version"].(string); nv != "" {
			t.Fatalf("expected node_version '', got %q", nv)
		}
	})
}

func TestGetUnknownKeys(t *testing.T) {
	t.Run("typo suggests correction", func(t *testing.T) {
		raw := map[string]any{
			"grpahs":       map[string]any{"agent": "./agent.py:graph"},
			"dependencies": []any{"langchain"},
		}
		warnings := GetUnknownKeys(raw)
		found := false
		for _, w := range warnings {
			if strings.Contains(w, "did you mean 'graphs'") {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected warning suggesting 'graphs', got: %v", warnings)
		}
	})

	t.Run("totally unknown key", func(t *testing.T) {
		raw := map[string]any{
			"totally_unknown": "value",
			"graphs":          map[string]any{"agent": "./agent.py:graph"},
			"dependencies":    []any{"langchain"},
		}
		warnings := GetUnknownKeys(raw)
		found := false
		for _, w := range warnings {
			if strings.Contains(w, "not a recognized config field") {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected 'not a recognized config field' warning, got: %v", warnings)
		}
	})

	t.Run("only known keys gives no warnings", func(t *testing.T) {
		raw := baseConfig()
		warnings := GetUnknownKeys(raw)
		if len(warnings) != 0 {
			t.Fatalf("expected no warnings, got: %v", warnings)
		}
	})
}

func TestValidateConfigMultiplatform(t *testing.T) {
	t.Run("only JS graphs", func(t *testing.T) {
		raw := map[string]any{
			"graphs": map[string]any{"bot": "./bot.ts:bot"},
		}
		result, err := ValidateConfig(raw)
		mustSucceed(t, err)
		if nv, _ := result["node_version"].(string); nv != "20" {
			t.Fatalf("expected node_version '20', got %q", nv)
		}
		if pv, _ := result["python_version"].(string); pv != "" {
			t.Fatalf("expected python_version '', got %q", pv)
		}
	})

	t.Run("only Python graphs", func(t *testing.T) {
		raw := baseConfig()
		result, err := ValidateConfig(raw)
		mustSucceed(t, err)
		if pv, _ := result["python_version"].(string); pv != "3.11" {
			t.Fatalf("expected python_version '3.11', got %q", pv)
		}
		if nv, _ := result["node_version"].(string); nv != "" {
			t.Fatalf("expected node_version '', got %q", nv)
		}
	})

	t.Run("mixed graphs", func(t *testing.T) {
		raw := map[string]any{
			"dependencies": []any{"langchain"},
			"graphs":       map[string]any{"agent": "./agent.py:graph", "bot": "./bot.ts:bot"},
		}
		result, err := ValidateConfig(raw)
		mustSucceed(t, err)
		if pv, _ := result["python_version"].(string); pv != "3.11" {
			t.Fatalf("expected python_version '3.11', got %q", pv)
		}
		if nv, _ := result["node_version"].(string); nv != "20" {
			t.Fatalf("expected node_version '20', got %q", nv)
		}
	})
}
