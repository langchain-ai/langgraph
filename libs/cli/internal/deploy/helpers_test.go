package deploy

import (
	"os"
	"sort"
	"testing"
)

func TestNormalizeImageName(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"MyApp", "myapp"},
		{"", "app"},
		{"!!!", "app"},
		{"my-app", "my-app"},
		{"My Cool App", "my-cool-app"},
		{"..leading-dots", "leading-dots"},
		{"trailing-dots..", "trailing-dots"},
		{"hello_world.v2", "hello_world.v2"},
	}
	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got := NormalizeImageName(tc.input)
			if got != tc.want {
				t.Errorf("NormalizeImageName(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestNormalizeImageTag(t *testing.T) {
	tests := []struct {
		input   string
		want    string
		wantErr bool
	}{
		{"v1.2.3", "v1.2.3", false},
		{"", "latest", false},
		{"has space", "", true},
		{"valid_tag-1.0", "valid_tag-1.0", false},
		{"tag/slash", "", true},
	}
	for _, tc := range tests {
		t.Run(tc.input, func(t *testing.T) {
			got, err := NormalizeImageTag(tc.input)
			if tc.wantErr {
				if err == nil {
					t.Errorf("NormalizeImageTag(%q) expected error, got nil", tc.input)
				}
				return
			}
			if err != nil {
				t.Errorf("NormalizeImageTag(%q) unexpected error: %v", tc.input, err)
				return
			}
			if got != tc.want {
				t.Errorf("NormalizeImageTag(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestValidateDeploymentSelector(t *testing.T) {
	tests := []struct {
		name         string
		deploymentID string
		depName      string
		wantErr      bool
	}{
		{"both empty", "", "", true},
		{"id set", "abc-123", "", false},
		{"name set", "", "my-deploy", false},
		{"both set", "abc-123", "my-deploy", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := ValidateDeploymentSelector(tc.deploymentID, tc.depName)
			if tc.wantErr && err == nil {
				t.Error("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestSecretsFromEnv(t *testing.T) {
	envVars := map[string]string{
		"MY_SECRET":         "value1",
		"ANOTHER_SECRET":    "value2",
		"LANGCHAIN_API_KEY": "should-be-filtered",
		"POSTGRES_URI":      "should-be-filtered",
		"EMPTY_VAR":         "",
	}

	secrets := SecretsFromEnv(envVars)

	// Sort for deterministic comparison
	sort.Slice(secrets, func(i, j int) bool {
		return secrets[i].Name < secrets[j].Name
	})

	if len(secrets) != 2 {
		t.Fatalf("expected 2 secrets, got %d: %+v", len(secrets), secrets)
	}

	if secrets[0].Name != "ANOTHER_SECRET" || secrets[0].Value != "value2" {
		t.Errorf("unexpected secret[0]: %+v", secrets[0])
	}
	if secrets[1].Name != "MY_SECRET" || secrets[1].Value != "value1" {
		t.Errorf("unexpected secret[1]: %+v", secrets[1])
	}
}

func TestResolveAPIKey(t *testing.T) {
	// Flag value takes precedence
	got := ResolveAPIKey("flag-key", map[string]string{"LANGSMITH_API_KEY": "env-key"})
	if got != "flag-key" {
		t.Errorf("expected flag-key, got %q", got)
	}

	// envVars map is checked next
	got = ResolveAPIKey("", map[string]string{"LANGSMITH_API_KEY": "env-key"})
	if got != "env-key" {
		t.Errorf("expected env-key, got %q", got)
	}

	// os.Getenv fallback
	os.Setenv("LANGSMITH_API_KEY", "os-env-key")
	defer os.Unsetenv("LANGSMITH_API_KEY")

	got = ResolveAPIKey("", nil)
	if got != "os-env-key" {
		t.Errorf("expected os-env-key, got %q", got)
	}

	// Flag still takes precedence over os env
	got = ResolveAPIKey("flag-value", nil)
	if got != "flag-value" {
		t.Errorf("expected flag-value, got %q", got)
	}

	// Empty everything returns empty
	os.Unsetenv("LANGSMITH_API_KEY")
	os.Unsetenv("LANGGRAPH_HOST_API_KEY")
	os.Unsetenv("LANGCHAIN_API_KEY")
	got = ResolveAPIKey("", nil)
	if got != "" {
		t.Errorf("expected empty string, got %q", got)
	}
}
