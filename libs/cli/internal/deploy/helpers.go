package deploy

import (
	"bufio"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

// APIKeyEnvNames lists the environment variable names checked (in order) when
// resolving a LangSmith / LangGraph API key.
var APIKeyEnvNames = []string{
	"LANGGRAPH_HOST_API_KEY",
	"LANGSMITH_API_KEY",
	"LANGCHAIN_API_KEY",
}

// DefaultHostURL is the default host backend URL.
const DefaultHostURL = "https://api.host.langchain.com"

// reservedEnvVars contains environment variable names that must not be sent as
// deployment secrets. The set mirrors the Python CLI's RESERVED_ENV_VARS.
var reservedEnvVars = map[string]bool{
	// LANGCHAIN_RESERVED_ENV_VARS from host-backend
	"LANGCHAIN_TRACING_V2":            true,
	"LANGSMITH_TRACING_V2":            true,
	"LANGCHAIN_ENDPOINT":              true,
	"LANGCHAIN_PROJECT":               true,
	"LANGSMITH_PROJECT":               true,
	"LANGSMITH_LANGGRAPH_GIT_REPO":    true,
	"LANGGRAPH_GIT_REPO_PATH":         true,
	"LANGCHAIN_API_KEY":               true,
	"LANGSMITH_CONTROL_PLANE_API_KEY": true,
	"POSTGRES_URI":                    true,
	"POSTGRES_PASSWORD":               true,
	"DATABASE_URI":                    true,
	"LANGSMITH_LANGGRAPH_GIT_REF":     true,
	"LANGSMITH_LANGGRAPH_GIT_REF_SHA": true,
	"LANGGRAPH_AUTH_TYPE":             true,
	"LANGSMITH_AUTH_ENDPOINT":         true,
	"LANGSMITH_TENANT_ID":             true,
	"LANGSMITH_AUTH_VERIFY_TENANT_ID": true,
	"LANGSMITH_HOST_PROJECT_ID":       true,
	"LANGSMITH_HOST_PROJECT_NAME":     true,
	"LANGSMITH_HOST_REVISION_ID":      true,
	"LOG_JSON":                        true,
	"LOG_DICT_TRACEBACKS":             true,
	"REDIS_URI":                       true,
	"LANGCHAIN_CALLBACKS_BACKGROUND":  true,
	"DD_TRACE_PSYCOPG_ENABLED":        true,
	"DD_TRACE_REDIS_ENABLED":          true,
	"LANGSMITH_DEPLOYMENT_NAME":       true,
	"LANGGRAPH_CLOUD_LICENSE_KEY":     true,
	// ALLOWED_SELF_HOSTED_ENV_VARS (rejected for non-self-hosted)
	"LANGSMITH_API_KEY":   true,
	"LANGSMITH_ENDPOINT":  true,
	"POSTGRES_URI_CUSTOM": true,
	"REDIS_URI_CUSTOM":    true,
	"PATH":                true,
	"PORT":                true,
	"MOUNT_PREFIX":        true,
	"LSD_ENV":             true,
	"LSD_DD_API_KEY":      true,
	"LSD_DD_ENDPOINT":     true,
	"LSD_DEPLOYMENT_TYPE": true,
}

var (
	invalidImageNameChars = regexp.MustCompile(`[^a-z0-9._-]+`)
	validImageTag         = regexp.MustCompile(`^[A-Za-z0-9_.-]+$`)
)

// NormalizeImageName sanitizes a deployment/directory name into a valid Docker
// repository name. Invalid characters are replaced with hyphens and the result
// is lowercased. Returns "app" if the result would be empty.
func NormalizeImageName(name string) string {
	if name == "" {
		return "app"
	}
	slug := invalidImageNameChars.ReplaceAllString(strings.ToLower(name), "-")
	slug = strings.TrimLeft(slug, "-.")
	slug = strings.TrimRight(slug, "-.")
	if slug == "" {
		return "app"
	}
	return slug
}

// NormalizeImageTag validates and returns a Docker image tag. Tags may only
// contain [A-Za-z0-9_.-]. Defaults to "latest" when empty.
func NormalizeImageTag(tag string) (string, error) {
	if tag == "" {
		return "latest", nil
	}
	if !validImageTag.MatchString(tag) {
		return "", fmt.Errorf("image tag may only contain characters A-Z, a-z, 0-9, '_', '-', '.'")
	}
	return tag, nil
}

// ResolveAPIKey resolves an API key by checking (in order): the explicit flag
// value, the provided envVars map, and the process environment. Returns an
// empty string if no key is found (the caller should prompt the user).
func ResolveAPIKey(flagValue string, envVars map[string]string) string {
	if flagValue != "" {
		return flagValue
	}
	for _, keyName := range APIKeyEnvNames {
		if envVars != nil {
			if v, ok := envVars[keyName]; ok && v != "" {
				return v
			}
		}
		if v := os.Getenv(keyName); v != "" {
			return v
		}
	}
	return ""
}

// ParseEnvFromConfig resolves environment variables from the langgraph.json
// config. If the "env" field is a dict (map), those values are used directly.
// If it is a string, it is treated as a path to a .env file (resolved relative
// to the config file's directory). Otherwise, a .env file in the config
// directory is attempted as a fallback.
func ParseEnvFromConfig(configJSON map[string]any, configPath string) map[string]string {
	envField, ok := configJSON["env"]
	if !ok {
		// Fallback: try .env in config dir.
		return parseDotEnvFile(filepath.Join(filepath.Dir(configPath), ".env"))
	}

	// If env is a dict (map[string]any), convert to map[string]string.
	if envMap, ok := envField.(map[string]any); ok && len(envMap) > 0 {
		result := make(map[string]string, len(envMap))
		for k, v := range envMap {
			result[k] = fmt.Sprintf("%v", v)
		}
		return result
	}

	// If env is a string path, parse that .env file.
	if envStr, ok := envField.(string); ok && envStr != "" {
		envPath := filepath.Join(filepath.Dir(configPath), envStr)
		absPath, err := filepath.Abs(envPath)
		if err != nil {
			return map[string]string{}
		}
		if _, err := os.Stat(absPath); os.IsNotExist(err) {
			return map[string]string{}
		}
		return parseDotEnvFile(absPath)
	}

	// Fallback: try .env in config dir.
	return parseDotEnvFile(filepath.Join(filepath.Dir(configPath), ".env"))
}

// parseDotEnvFile reads a .env file and returns its key-value pairs. Lines
// starting with # are treated as comments. Empty values are skipped.
func parseDotEnvFile(path string) map[string]string {
	f, err := os.Open(path)
	if err != nil {
		return map[string]string{}
	}
	defer f.Close()

	result := map[string]string{}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		idx := strings.IndexByte(line, '=')
		if idx < 0 {
			continue
		}
		key := strings.TrimSpace(line[:idx])
		value := strings.TrimSpace(line[idx+1:])
		// Strip surrounding quotes if present.
		if len(value) >= 2 {
			if (value[0] == '"' && value[len(value)-1] == '"') ||
				(value[0] == '\'' && value[len(value)-1] == '\'') {
				value = value[1 : len(value)-1]
			}
		}
		if key != "" && value != "" {
			result[key] = value
		}
	}
	return result
}

// FindDeploymentIDByName lists deployments matching the given name and returns
// the ID of the first exact match. Returns an empty string (and no error) if
// no exact match is found.
func FindDeploymentIDByName(client *HostBackendClient, name string) (string, error) {
	if name == "" {
		return "", nil
	}
	existing, err := client.ListDeployments(name)
	if err != nil {
		return "", err
	}
	resources, ok := existing["resources"]
	if !ok {
		return "", nil
	}
	resourceList, ok := resources.([]any)
	if !ok {
		return "", nil
	}
	for _, item := range resourceList {
		dep, ok := item.(map[string]any)
		if !ok {
			continue
		}
		depName, _ := dep["name"].(string)
		if depName == name {
			if id, ok := dep["id"]; ok {
				return fmt.Sprintf("%v", id), nil
			}
		}
	}
	return "", nil
}

// ValidateDeploymentSelector ensures at least one of deploymentID or name is
// provided.
func ValidateDeploymentSelector(deploymentID, name string) error {
	if deploymentID != "" {
		return nil
	}
	if name == "" {
		return fmt.Errorf("either --deployment-id or --name is required")
	}
	return nil
}

// ResolvedReservedEnvVars returns the set of reserved environment variable
// names that must not be sent as deployment secrets.
func ResolvedReservedEnvVars() map[string]bool {
	// Return a copy to prevent callers from mutating the package-level map.
	result := make(map[string]bool, len(reservedEnvVars))
	for k, v := range reservedEnvVars {
		result[k] = v
	}
	return result
}

// SecretsFromEnv converts an env var map into a Secret slice, filtering out
// reserved variable names and empty values.
func SecretsFromEnv(envVars map[string]string) []Secret {
	var secrets []Secret
	for name, value := range envVars {
		if reservedEnvVars[name] {
			continue
		}
		if value == "" {
			continue
		}
		secrets = append(secrets, Secret{Name: name, Value: value})
	}
	return secrets
}
