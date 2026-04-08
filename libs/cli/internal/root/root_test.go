package root

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunHelp(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run(nil, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("expected exit code 0, got %d", exitCode)
	}
	if stderr.Len() != 0 {
		t.Fatalf("expected no stderr output, got %q", stderr.String())
	}
	if !strings.Contains(stdout.String(), "validate") {
		t.Fatalf("expected help text to contain 'validate', got %q", stdout.String())
	}
}

func TestRunVersion(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run([]string{"version"}, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("expected exit code 0, got %d", exitCode)
	}
	if stderr.Len() != 0 {
		t.Fatalf("expected no stderr output, got %q", stderr.String())
	}
	if !strings.Contains(stdout.String(), "langgraph") {
		t.Fatalf("unexpected stdout: %q", stdout.String())
	}
}

func TestRunUnknownCommand(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run([]string{"nonexistent-cmd"}, &stdout, &stderr)

	if exitCode != 1 {
		t.Fatalf("expected exit code 1, got %d", exitCode)
	}
	if !strings.Contains(stderr.String(), "is not a langgraph command") {
		t.Fatalf("unexpected stderr: %q", stderr.String())
	}
}

func writeTempConfig(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "langgraph.json")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatalf("failed to write temp config: %v", err)
	}
	return path
}

func TestRunValidateWithValidConfig(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	path := writeTempConfig(t, `{"dependencies": ["langchain"], "graphs": {"agent": "./agent.py:graph"}}`)

	exitCode := Run([]string{"validate", "-c", path}, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %q", exitCode, stderr.String())
	}
	if !strings.Contains(stdout.String(), "is valid") {
		t.Fatalf("expected stdout to contain 'is valid', got %q", stdout.String())
	}
	if !strings.Contains(stdout.String(), "1 graph") {
		t.Fatalf("expected stdout to contain '1 graph', got %q", stdout.String())
	}
}

func TestRunValidateWithInvalidConfig(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	path := writeTempConfig(t, `{"graphs": {}}`)

	exitCode := Run([]string{"validate", "-c", path}, &stdout, &stderr)

	if exitCode != 1 {
		t.Fatalf("expected exit code 1, got %d", exitCode)
	}
	if !strings.Contains(stderr.String(), "No graphs found") {
		t.Fatalf("expected stderr to contain 'No graphs found', got %q", stderr.String())
	}
}

func TestRunValidateWithInvalidJSON(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	path := writeTempConfig(t, `{invalid json`)

	exitCode := Run([]string{"validate", "-c", path}, &stdout, &stderr)

	if exitCode != 1 {
		t.Fatalf("expected exit code 1, got %d", exitCode)
	}
	if !strings.Contains(stderr.String(), "Invalid JSON") {
		t.Fatalf("expected stderr to contain 'Invalid JSON', got %q", stderr.String())
	}
}

func TestRunValidateDefaultConfigMissing(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	// Use a path that definitely does not exist.
	nonexistent := filepath.Join(t.TempDir(), "langgraph.json")

	exitCode := Run([]string{"validate", "-c", nonexistent}, &stdout, &stderr)

	if exitCode != 1 {
		t.Fatalf("expected exit code 1, got %d", exitCode)
	}
	if !strings.Contains(stderr.String(), "does not exist") {
		t.Fatalf("expected stderr to contain 'does not exist', got %q", stderr.String())
	}
}

func TestRunValidateWithUnknownKeys(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	path := writeTempConfig(t, `{"dependencies": ["langchain"], "graphs": {"agent": "./agent.py:graph"}, "grpahs": {}}`)

	exitCode := Run([]string{"validate", "-c", path}, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %q", exitCode, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(strings.ToLower(out), "warning") {
		t.Fatalf("expected stdout to contain 'warning', got %q", out)
	}
	if !strings.Contains(strings.ToLower(out), "did you mean") {
		t.Fatalf("expected stdout to contain 'did you mean', got %q", out)
	}
}

func TestRunValidateHelp(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	exitCode := Run([]string{"validate", "--help"}, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("expected exit code 0, got %d", exitCode)
	}
	if !strings.Contains(stdout.String(), "Validate the LangGraph configuration file") {
		t.Fatalf("expected stdout to contain validate help text, got %q", stdout.String())
	}
}

func TestRunValidateMultipleGraphs(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	path := writeTempConfig(t, `{"dependencies": ["langchain"], "graphs": {"agent": "./a.py:g", "bot": "./b.py:g"}}`)

	exitCode := Run([]string{"validate", "-c", path}, &stdout, &stderr)

	if exitCode != 0 {
		t.Fatalf("expected exit code 0, got %d; stderr: %q", exitCode, stderr.String())
	}
	if !strings.Contains(stdout.String(), "2 graphs found") {
		t.Fatalf("expected stdout to contain '2 graphs found', got %q", stdout.String())
	}
}

func TestRunValidateWithWarningsAndErrors(t *testing.T) {
	var stdout bytes.Buffer
	var stderr bytes.Buffer

	path := writeTempConfig(t, `{"graphs": {}, "grpahs": {}}`)

	exitCode := Run([]string{"validate", "-c", path}, &stdout, &stderr)

	if exitCode != 1 {
		t.Fatalf("expected exit code 1, got %d", exitCode)
	}
	errOut := stderr.String()
	if !strings.Contains(errOut, "No graphs found") {
		t.Fatalf("expected stderr to contain 'No graphs found', got %q", errOut)
	}
	if !strings.Contains(strings.ToLower(errOut), "warning") {
		t.Fatalf("expected stderr to contain 'warning', got %q", errOut)
	}
}
