// Package config provides validation for langgraph.json configuration files.
package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
)

const (
	MinNodeVersion       = "20"
	DefaultNodeVersion   = "20"
	MinPythonVersion     = "3.11"
	DefaultPythonVersion = "3.11"
	DefaultImageDistro   = "debian"
)

var validDistros = []string{"debian", "wolfi", "bookworm"}

var knownConfigKeys = map[string]bool{
	"python_version":       true,
	"node_version":         true,
	"api_version":          true,
	"base_image":           true,
	"image_distro":         true,
	"pip_config_file":      true,
	"pip_installer":        true,
	"source":               true,
	"dependencies":         true,
	"dockerfile_lines":     true,
	"graphs":               true,
	"env":                  true,
	"store":                true,
	"auth":                 true,
	"encryption":           true,
	"http":                 true,
	"webhooks":             true,
	"checkpointer":         true,
	"ui":                   true,
	"ui_config":            true,
	"keep_pkg_tools":       true,
	"_INTERNAL_docker_tag": true,
	"project_root":         true,
	"package":              true,
}

var nodeExtensions = map[string]bool{
	".ts": true, ".mts": true, ".cts": true,
	".js": true, ".mjs": true, ".cjs": true,
}

// isNodeGraph checks whether a graph spec refers to a Node.js file.
func isNodeGraph(spec any) bool {
	var filePath string
	switch v := spec.(type) {
	case string:
		filePath = strings.SplitN(v, ":", 2)[0]
	case map[string]any:
		if p, _ := v["path"].(string); p != "" {
			filePath = strings.SplitN(p, ":", 2)[0]
		}
	}
	return nodeExtensions[filepath.Ext(filePath)]
}

// getSourceKind extracts source.kind from a raw config.
func getSourceKind(raw map[string]any) string {
	source, ok := raw["source"]
	if !ok {
		return ""
	}
	m, ok := source.(map[string]any)
	if !ok {
		return ""
	}
	kind, _ := m["kind"].(string)
	return kind
}

// getString returns the string value for key, or "" if missing/wrong type.
func getString(raw map[string]any, key string) string {
	v, _ := raw[key].(string)
	return v
}

// parseVersion parses "3.11" or "0.8.1" into integer parts.
func parseVersion(s string) ([]int, error) {
	s = strings.SplitN(s, "-", 2)[0]
	parts := strings.Split(s, ".")
	result := make([]int, len(parts))
	for i, p := range parts {
		n, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid version part: %s", p)
		}
		result[i] = n
	}
	return result, nil
}

// versionLessThan returns true if a < b (component-wise).
func versionLessThan(a, b []int) bool {
	for i := 0; i < len(a) && i < len(b); i++ {
		if a[i] < b[i] {
			return true
		}
		if a[i] > b[i] {
			return false
		}
	}
	return len(a) < len(b)
}

// ValidateConfig validates a raw config map and returns a normalised copy.
// Errors match the Python CLI's click.UsageError messages exactly.
func ValidateConfig(raw map[string]any) (map[string]any, error) {
	// --- detect graph types ---
	graphs, _ := raw["graphs"].(map[string]any)
	hasNode, hasPython := false, false
	for _, spec := range graphs {
		if isNodeGraph(spec) {
			hasNode = true
		} else {
			hasPython = true
		}
	}

	// --- version defaults ---
	nodeVersion := getString(raw, "node_version")
	pythonVersion := getString(raw, "python_version")

	if hasNode && nodeVersion == "" {
		nodeVersion = DefaultNodeVersion
	}
	if hasPython && pythonVersion == "" {
		pythonVersion = DefaultPythonVersion
	}

	imageDistro := getString(raw, "image_distro")
	if imageDistro == "" {
		imageDistro = DefaultImageDistro
	}

	// --- mutual exclusion: _INTERNAL_docker_tag vs api_version ---
	_, hasInternalTag := raw["_INTERNAL_docker_tag"]
	_, hasAPIVersion := raw["api_version"]
	if hasInternalTag && hasAPIVersion {
		return nil, fmt.Errorf("Cannot specify both _INTERNAL_docker_tag and api_version.")
	}

	// --- api_version format ---
	if apiVersion := getString(raw, "api_version"); apiVersion != "" {
		base := strings.SplitN(apiVersion, "-", 2)[0]
		parts := strings.Split(base, ".")
		if len(parts) > 3 {
			return nil, fmt.Errorf("Version must be major or major.minor or major.minor.patch.")
		}
		for _, p := range parts {
			if _, err := strconv.Atoi(p); err != nil {
				return nil, fmt.Errorf(
					"Invalid version format: %s.\n\n"+
						"Pin to a minor version, e.g.:\n"+
						"  \"api_version\": \"0.8\"", apiVersion)
			}
		}
	}

	// --- build result config with defaults ---
	config := map[string]any{
		"node_version":     nodeVersion,
		"python_version":   pythonVersion,
		"pip_config_file":  raw["pip_config_file"],
		"pip_installer":    "auto",
		"source":           raw["source"],
		"base_image":       raw["base_image"],
		"image_distro":     imageDistro,
		"dependencies":     raw["dependencies"],
		"dockerfile_lines": raw["dockerfile_lines"],
		"graphs":           raw["graphs"],
		"env":              raw["env"],
		"store":            raw["store"],
		"auth":             raw["auth"],
		"encryption":       raw["encryption"],
		"http":             raw["http"],
		"webhooks":         raw["webhooks"],
		"checkpointer":     raw["checkpointer"],
		"ui":               raw["ui"],
		"ui_config":        raw["ui_config"],
		"keep_pkg_tools":   raw["keep_pkg_tools"],
	}
	if raw["pip_installer"] != nil {
		config["pip_installer"] = raw["pip_installer"]
	}
	if hasInternalTag {
		config["_INTERNAL_docker_tag"] = raw["_INTERNAL_docker_tag"]
	}
	if hasAPIVersion {
		config["api_version"] = raw["api_version"]
	}

	// Apply list defaults.
	if config["dependencies"] == nil {
		config["dependencies"] = []any{}
	}
	if config["dockerfile_lines"] == nil {
		config["dockerfile_lines"] = []any{}
	}
	if config["graphs"] == nil {
		config["graphs"] = map[string]any{}
	}
	if config["env"] == nil {
		config["env"] = map[string]any{}
	}

	// --- node_version validation ---
	if nodeVersion != "" {
		if strings.Contains(nodeVersion, ".") {
			return nil, fmt.Errorf("Node.js version must be major version only")
		}
		major, err := strconv.Atoi(nodeVersion)
		if err != nil {
			return nil, fmt.Errorf(
				"Invalid Node.js version format: %s. Use major version only (e.g., '20').",
				nodeVersion)
		}
		minMajor, _ := strconv.Atoi(MinNodeVersion)
		if major < minMajor {
			return nil, fmt.Errorf(
				"Node.js version %s is not supported. "+
					"Minimum required version is %s.\n\n"+
					"Set node_version to %s or higher:\n"+
					"  \"node_version\": \"%s\"",
				nodeVersion, MinNodeVersion, MinNodeVersion, MinNodeVersion)
		}
	}

	// --- pip_installer validation ---
	if pi, ok := raw["pip_installer"].(string); ok {
		switch pi {
		case "auto", "pip", "uv":
			// valid
		default:
			return nil, fmt.Errorf(
				"Invalid pip_installer: '%s'. "+
					"Consider using uv-based source management instead:\n\n"+
					"  \"source\": {\"kind\": \"uv\", \"root\": \"..\"}",
				pi)
		}
	}

	// --- source validation ---
	sourceKind := getSourceKind(raw)
	if source := raw["source"]; source != nil {
		if _, ok := source.(map[string]any); !ok {
			return nil, fmt.Errorf(
				"`source` must be an object, e.g.:\n" +
					"  \"source\": {\"kind\": \"uv\", \"root\": \"..\"}")
		}
		if sourceKind != "uv" {
			return nil, fmt.Errorf(
				"Invalid source.kind. The only supported value is 'uv':\n" +
					"  \"source\": {\"kind\": \"uv\", \"root\": \"..\"}")
		}
	}

	// --- python_version validation ---
	if pythonVersion != "" {
		base := strings.SplitN(pythonVersion, "-", 2)[0]
		dotParts := strings.Split(base, ".")
		allDigits := true
		for _, p := range dotParts {
			if _, err := strconv.Atoi(p); err != nil {
				allDigits = false
				break
			}
		}
		if len(dotParts) != 2 || !allDigits {
			fix := MinPythonVersion
			if len(dotParts) >= 2 {
				fix = dotParts[0] + "." + dotParts[1]
			}
			return nil, fmt.Errorf(
				"Invalid Python version format: %s. "+
					"Use 'major.minor' format — patch version cannot be specified.\n\n"+
					"  \"python_version\": \"%s\"",
				pythonVersion, fix)
		}
		pyParsed, _ := parseVersion(pythonVersion)
		minParsed, _ := parseVersion(MinPythonVersion)
		if versionLessThan(pyParsed, minParsed) {
			return nil, fmt.Errorf(
				"Python version %s is not supported. "+
					"Minimum required version is %s.\n\n"+
					"  \"python_version\": \"%s\"",
				pythonVersion, MinPythonVersion, MinPythonVersion)
		}
		if strings.Contains(pythonVersion, "bullseye") {
			return nil, fmt.Errorf(
				"Bullseye images were deprecated in version 0.4.13. " +
					"Please use 'bookworm' or 'debian' instead.")
		}

		// dependencies required when not uv
		deps, _ := config["dependencies"].([]any)
		if sourceKind != "uv" && len(deps) == 0 {
			return nil, fmt.Errorf(
				"No dependencies found in config. " +
					"Consider using uv-based source management:\n\n" +
					"  \"source\": {\"kind\": \"uv\", \"root\": \"..\"}")
		}
	}

	// --- graphs required ---
	graphMap, _ := config["graphs"].(map[string]any)
	if len(graphMap) == 0 {
		return nil, fmt.Errorf(
			"No graphs found in config. Add at least one graph, e.g.:\n" +
				"  \"graphs\": {\n" +
				"    \"agent\": \"./my_agent/graph.py:graph\"\n" +
				"  }")
	}

	// --- image_distro validation ---
	if imageDistro == "bullseye" {
		return nil, fmt.Errorf(
			"Bullseye images were deprecated in version 0.4.13. " +
				"Please use 'bookworm' or 'debian' instead.")
	}
	validDistro := false
	for _, d := range validDistros {
		if imageDistro == d {
			validDistro = true
			break
		}
	}
	if !validDistro {
		quoted := make([]string, len(validDistros))
		for i, d := range validDistros {
			quoted[i] = fmt.Sprintf("'%s'", d)
		}
		return nil, fmt.Errorf(
			"Invalid image_distro: '%s'. "+
				"Must be one of: %s.\n\n"+
				"  \"image_distro\": \"wolfi\"  (recommended)",
			imageDistro, strings.Join(quoted, ", "))
	}

	// --- uv source mode validation ---
	if sourceKind == "uv" {
		var errs []string
		if pythonVersion == "" {
			errs = append(errs, "source.kind 'uv' requires `python_version` — it is a Python-only deployment mode. Node.js-only graphs are not supported.")
		}

		deps, _ := raw["dependencies"].([]any)
		if deps != nil && len(deps) > 0 {
			errs = append(errs, "Remove `dependencies` from your config. With `source.kind = \"uv\"`, all dependencies are read from your pyproject.toml and uv.lock instead.")
		}
		// Also check if dependencies key exists even if empty array.
		if deps == nil {
			if rawDeps, exists := raw["dependencies"]; exists && rawDeps != nil {
				// dependencies key present but not an array — still flag it
				if depsArr, ok := rawDeps.([]any); ok && len(depsArr) > 0 {
					errs = append(errs, "Remove `dependencies` from your config. With `source.kind = \"uv\"`, all dependencies are read from your pyproject.toml and uv.lock instead.")
				}
			}
		}

		sourceMap, _ := raw["source"].(map[string]any)
		if root, exists := sourceMap["root"]; exists {
			rootStr, ok := root.(string)
			if !ok {
				errs = append(errs, fmt.Sprintf("`source.root` must be a string, got %T.", root))
			} else if rootStr == "" {
				errs = append(errs, "`source.root` must be a non-empty string. Use `\".\"`.")
			}
		}

		if pkg, exists := sourceMap["package"]; exists {
			if pkg != nil {
				pkgStr, ok := pkg.(string)
				if !ok {
					errs = append(errs, "`source.package` must be a non-empty string.")
				} else if pkgStr == "" {
					errs = append(errs, "`source.package` must be a non-empty string.")
				}
			}
		}

		if len(errs) > 0 {
			formatted := ""
			for i, e := range errs {
				formatted += fmt.Sprintf("\n  %d. %s", i+1, e)
			}
			return nil, fmt.Errorf(
				"source.kind 'uv' requires a different config shape than dependency-based installs:%s",
				formatted)
		}
	}

	// --- legacy project_root / package ---
	_, hasProjectRoot := raw["project_root"]
	_, hasPackage := raw["package"]
	if hasProjectRoot || hasPackage {
		return nil, fmt.Errorf(
			"Top-level `project_root` and `package` are no longer supported. " +
				"Use `source.root` and `source.package` instead.")
	}

	// --- auth path validation ---
	if auth, ok := raw["auth"].(map[string]any); ok {
		if authPath, _ := auth["path"].(string); authPath != "" {
			if !strings.Contains(authPath, ":") {
				return nil, fmt.Errorf(
					"Invalid auth.path format: '%s'. "+
						"Must be in format './path/to/file.py:attribute_name'",
					authPath)
			}
		}
	}

	// --- encryption path validation ---
	if enc, ok := raw["encryption"].(map[string]any); ok {
		if encPath, _ := enc["path"].(string); encPath != "" {
			if !strings.Contains(encPath, ":") {
				return nil, fmt.Errorf(
					"Invalid encryption.path format: '%s'. "+
						"Must be in format './path/to/file.py:attribute_name'",
					encPath)
			}
		}
	}

	// --- http.app path validation ---
	if httpConf, ok := raw["http"].(map[string]any); ok {
		if app, _ := httpConf["app"].(string); app != "" {
			if !strings.Contains(app, ":") {
				return nil, fmt.Errorf(
					"Invalid http.app format: '%s'. "+
						"Must be in format './path/to/file.py:attribute_name'",
					app)
			}
		}
	}

	// --- keep_pkg_tools validation ---
	if kpt := raw["keep_pkg_tools"]; kpt != nil {
		validBuildTools := map[string]bool{"pip": true, "setuptools": true, "wheel": true}
		switch v := kpt.(type) {
		case bool:
			// ok
		case []any:
			for _, item := range v {
				tool, ok := item.(string)
				if !ok || !validBuildTools[tool] {
					return nil, fmt.Errorf(
						"Invalid keep_pkg_tools: '%v'. "+
							"Must be one of 'pip', 'setuptools', 'wheel'.",
						item)
				}
			}
		default:
			return nil, fmt.Errorf(
				"Invalid keep_pkg_tools: '%v'. "+
					"Must be bool or list[str] (with values 'pip', 'setuptools', and/or 'wheel').",
				kpt)
		}
	}

	return config, nil
}

// ValidateConfigFile loads a config file, validates it, and returns the result.
func ValidateConfigFile(configPath string) (map[string]any, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("could not read config file: %w", err)
	}

	var raw map[string]any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, fmt.Errorf("Invalid JSON in %s: %s", configPath, err.Error())
	}

	validated, err := ValidateConfig(raw)
	if err != nil {
		return nil, err
	}

	// Check package.json node version if node_version is set.
	if nv, _ := validated["node_version"].(string); nv != "" {
		dir := filepath.Dir(configPath)
		pkgJSONPath := filepath.Join(dir, "package.json")
		if info, statErr := os.Stat(pkgJSONPath); statErr == nil && !info.IsDir() {
			if pkgErr := validatePackageJSON(pkgJSONPath); pkgErr != nil {
				return nil, pkgErr
			}
		}
	}

	return validated, nil
}

func validatePackageJSON(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}

	var pkg map[string]any
	if err := json.Unmarshal(data, &pkg); err != nil {
		return fmt.Errorf("Invalid package.json: %s", err.Error())
	}

	enginesRaw, ok := pkg["engines"]
	if !ok {
		return nil
	}
	engines, ok := enginesRaw.(map[string]any)
	if !ok {
		return nil
	}

	for k := range engines {
		if k != "node" {
			keys := make([]string, 0, len(engines))
			for ek := range engines {
				keys = append(keys, ek)
			}
			return fmt.Errorf(
				"Only 'node' engine is supported in package.json engines. Got engines: %v",
				keys)
		}
	}

	if nodeVer, ok := engines["node"].(string); ok && nodeVer != "" {
		if strings.Contains(nodeVer, ".") {
			return fmt.Errorf(
				"Node.js version in package.json engines must be >= %s "+
					"(major version only), got '%s'. "+
					"Minor/patch versions (like '20.x.y') are not supported to "+
					"prevent deployment issues when new Node.js versions are released.",
				MinNodeVersion, nodeVer)
		}
		major, err := strconv.Atoi(nodeVer)
		if err == nil {
			minMajor, _ := strconv.Atoi(MinNodeVersion)
			if major < minMajor {
				return fmt.Errorf(
					"Node.js version in package.json engines must be >= %s "+
						"(major version only), got '%s'. "+
						"Minor/patch versions (like '20.x.y') are not supported to "+
						"prevent deployment issues when new Node.js versions are released.",
					MinNodeVersion, nodeVer)
			}
		}
	}

	return nil
}

// GetUnknownKeys returns warnings for unrecognised top-level keys.
func GetUnknownKeys(raw map[string]any) []string {
	var unknown []string
	for k := range raw {
		if !knownConfigKeys[k] {
			unknown = append(unknown, k)
		}
	}
	sortStrings(unknown)

	var warnings []string
	knownList := make([]string, 0, len(knownConfigKeys))
	for k := range knownConfigKeys {
		knownList = append(knownList, k)
	}

	for _, key := range unknown {
		if close := closestMatch(key, knownList); close != "" {
			warnings = append(warnings, fmt.Sprintf("Unknown key '%s' — did you mean '%s'?", key, close))
		} else {
			warnings = append(warnings, fmt.Sprintf("Unknown key '%s' is not a recognized config field.", key))
		}
	}
	return warnings
}

// closestMatch finds the best match for word among candidates using edit distance.
// Returns "" if no match is close enough (ratio >= 0.6).
func closestMatch(word string, candidates []string) string {
	best := ""
	bestRatio := 0.6 // minimum threshold
	for _, c := range candidates {
		ratio := similarity(word, c)
		if ratio > bestRatio {
			bestRatio = ratio
			best = c
		}
	}
	return best
}

// similarity returns a ratio in [0,1] based on Levenshtein distance.
func similarity(a, b string) float64 {
	maxLen := len(a)
	if len(b) > maxLen {
		maxLen = len(b)
	}
	if maxLen == 0 {
		return 1.0
	}
	dist := editDistance(a, b)
	return 1.0 - float64(dist)/float64(maxLen)
}

// editDistance computes Levenshtein distance between two strings.
func editDistance(a, b string) int {
	la, lb := len(a), len(b)
	if la == 0 {
		return lb
	}
	if lb == 0 {
		return la
	}

	prev := make([]int, lb+1)
	curr := make([]int, lb+1)

	for j := 0; j <= lb; j++ {
		prev[j] = j
	}
	for i := 1; i <= la; i++ {
		curr[0] = i
		for j := 1; j <= lb; j++ {
			cost := 1
			if a[i-1] == b[j-1] {
				cost = 0
			}
			ins := curr[j-1] + 1
			del := prev[j] + 1
			sub := prev[j-1] + cost
			curr[j] = min3(ins, del, sub)
		}
		prev, curr = curr, prev
	}
	return prev[lb]
}

func min3(a, b, c int) int {
	if a < b {
		if a < c {
			return a
		}
		return c
	}
	if b < c {
		return b
	}
	return c
}

// sortStrings sorts a slice of strings in place (simple insertion sort, fine for small n).
func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}
