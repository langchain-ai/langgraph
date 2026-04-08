package docker

import (
	"fmt"
	"strings"
	"testing"
)

func cleanEmptyLines(s string) string {
	lines := strings.Split(s, "\n")
	var result []string
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			result = append(result, line)
		}
	}
	return strings.Join(result, "\n")
}

var defaultCaps = &DockerCapabilities{
	VersionDocker:            Version{Major: 26, Minor: 1, Patch: 1},
	VersionCompose:           Version{Major: 2, Minor: 27, Patch: 0},
	HealthcheckStartInterval: false,
}

func TestComposeCustomDBNoDebugger(t *testing.T) {
	port := 8123
	actual := Compose(defaultCaps, ComposeOpts{
		Port:        port,
		PostgresURI: "custom_postgres_uri",
	})
	expected := fmt.Sprintf(`services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-api:
        ports:
            - "%d:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: custom_postgres_uri`, port)

	if cleanEmptyLines(actual) != expected {
		t.Errorf("mismatch.\nExpected:\n%s\n\nGot:\n%s", expected, cleanEmptyLines(actual))
	}
}

func TestComposeCustomDBWithHealthcheck(t *testing.T) {
	port := 8123
	capsHC := &DockerCapabilities{
		VersionDocker:            Version{Major: 26, Minor: 1, Patch: 1},
		VersionCompose:           Version{Major: 2, Minor: 27, Patch: 0},
		HealthcheckStartInterval: true,
	}
	actual := Compose(capsHC, ComposeOpts{
		Port:        port,
		PostgresURI: "custom_postgres_uri",
	})
	expected := fmt.Sprintf(`services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-api:
        ports:
            - "%d:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: custom_postgres_uri
        healthcheck:
            test: python /api/healthcheck.py
            interval: 60s
            start_interval: 1s
            start_period: 10s`, port)

	if cleanEmptyLines(actual) != expected {
		t.Errorf("mismatch.\nExpected:\n%s\n\nGot:\n%s", expected, cleanEmptyLines(actual))
	}
}

func TestComposeDefaultDB(t *testing.T) {
	port := 8123
	actual := Compose(defaultCaps, ComposeOpts{Port: port})
	expected := fmt.Sprintf(`volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-postgres:
        image: pgvector/pgvector:pg16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        command:
            - postgres
            - -c
            - shared_preload_libraries=vector
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-api:
        ports:
            - "%d:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: %s`, port, DefaultPostgresURI)

	if cleanEmptyLines(actual) != expected {
		t.Errorf("mismatch.\nExpected:\n%s\n\nGot:\n%s", expected, cleanEmptyLines(actual))
	}
}

func TestComposeDistributedMode(t *testing.T) {
	port := 8123
	actual := Compose(defaultCaps, ComposeOpts{
		Port:              port,
		PostgresURI:       "custom_postgres_uri",
		EngineRuntimeMode: "distributed",
	})
	expected := fmt.Sprintf(`services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-api:
        ports:
            - "%d:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: custom_postgres_uri
            N_JOBS_PER_WORKER: "0"`, port)

	if cleanEmptyLines(actual) != expected {
		t.Errorf("mismatch.\nExpected:\n%s\n\nGot:\n%s", expected, cleanEmptyLines(actual))
	}
}

func TestComposeCombinedModeNoNJobs(t *testing.T) {
	actual := Compose(defaultCaps, ComposeOpts{
		Port:              8123,
		EngineRuntimeMode: "combined_queue_worker",
	})
	if strings.Contains(actual, "N_JOBS_PER_WORKER") {
		t.Error("combined mode should not contain N_JOBS_PER_WORKER")
	}
}

func TestComposeDebuggerDefaultDB(t *testing.T) {
	port := 8123
	debuggerPort := 8001
	actual := Compose(defaultCaps, ComposeOpts{
		Port:         port,
		DebuggerPort: debuggerPort,
	})
	expected := fmt.Sprintf(`volumes:
    langgraph-data:
        driver: local
services:
    langgraph-redis:
        image: redis:6
        healthcheck:
            test: redis-cli ping
            interval: 5s
            timeout: 1s
            retries: 5
    langgraph-postgres:
        image: pgvector/pgvector:pg16
        ports:
            - "5433:5432"
        environment:
            POSTGRES_DB: postgres
            POSTGRES_USER: postgres
            POSTGRES_PASSWORD: postgres
        command:
            - postgres
            - -c
            - shared_preload_libraries=vector
        volumes:
            - langgraph-data:/var/lib/postgresql/data
        healthcheck:
            test: pg_isready -U postgres
            start_period: 10s
            timeout: 1s
            retries: 5
            interval: 5s
    langgraph-debugger:
        image: langchain/langgraph-debugger
        restart: on-failure
        depends_on:
            langgraph-postgres:
                condition: service_healthy
        ports:
            - "%d:3968"
    langgraph-api:
        ports:
            - "%d:8000"
        depends_on:
            langgraph-redis:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        environment:
            REDIS_URI: redis://langgraph-redis:6379
            POSTGRES_URI: %s`, debuggerPort, port, DefaultPostgresURI)

	if cleanEmptyLines(actual) != expected {
		t.Errorf("mismatch.\nExpected:\n%s\n\nGot:\n%s", expected, cleanEmptyLines(actual))
	}
}

func TestParseVersion(t *testing.T) {
	tests := []struct {
		input    string
		expected Version
	}{
		{"1.2.3", Version{1, 2, 3}},
		{"v1.2.3", Version{1, 2, 3}},
		{"1.2.3-alpha", Version{1, 2, 3}},
		{"1.2.3+1", Version{1, 2, 3}},
		{"1.2.3-alpha+build", Version{1, 2, 3}},
		{"1.2", Version{1, 2, 0}},
		{"1", Version{1, 0, 0}},
		{"v28.1.1+1", Version{28, 1, 1}},
		{"2.0.0-beta.1+exp.sha.5114f85", Version{2, 0, 0}},
		{"v3.4.5-rc1+build.123", Version{3, 4, 5}},
	}

	for _, tc := range tests {
		result := ParseVersion(tc.input)
		if result != tc.expected {
			t.Errorf("ParseVersion(%q) = %v, want %v", tc.input, result, tc.expected)
		}
	}
}

func TestVersionGreaterOrEqual(t *testing.T) {
	tests := []struct {
		v, other Version
		want     bool
	}{
		{Version{25, 0, 0}, Version{25, 0, 0}, true},
		{Version{26, 1, 1}, Version{25, 0, 0}, true},
		{Version{24, 9, 9}, Version{25, 0, 0}, false},
	}

	for _, tc := range tests {
		got := tc.v.GreaterOrEqual(tc.other)
		if got != tc.want {
			t.Errorf("%v.GreaterOrEqual(%v) = %v, want %v", tc.v, tc.other, got, tc.want)
		}
	}
}
