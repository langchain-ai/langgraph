package config

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// UvLockSourceEntry represents a single [tool.uv.sources] entry with its
// provenance (which pyproject.toml declared it, and from which directory).
type UvLockSourceEntry struct {
	Name          string
	Value         any
	DeclaredRoot  string // absolute path
	PyprojectPath string
}

// UvLockPackage represents one workspace package discovered from pyproject.toml.
type UvLockPackage struct {
	Name                  string
	NormalizedName        string
	Root                  string // absolute path
	PyprojectPath         string
	RawDependencySpecs    any
	RawUvTool             any
	PackageEnabled        bool
	DependencyNames       []string
	WorkspaceDependencies []string
}

// UvLockWorkspace holds the result of discovering all workspace packages.
type UvLockWorkspace struct {
	RawRootSourceEntries any
	PackagesByName       map[string]*UvLockPackage
	PackagesByRoot       map[string]*UvLockPackage
}

// UvLockPlan is the fully resolved build plan for a uv-lock deployment.
type UvLockPlan struct {
	ProjectRoot       string
	PyprojectPath     string
	UvLockPath        string
	Target            *UvLockPackage
	TargetRoot        string
	InstallOrder      []*UvLockPackage
	ContainerRoots    map[string]string // hostRoot -> containerPath
	WorkingDir        string
	AllWorkspaceRoots map[string]bool
}

// ---------------------------------------------------------------------------
// Minimal TOML parser
// ---------------------------------------------------------------------------

// parseTOML parses a subset of TOML needed for pyproject.toml files.
// Supports: string values, boolean values, arrays of strings, inline tables,
// and dotted table headers.
func parseTOML(data string) map[string]any {
	result := make(map[string]any)
	currentSection := result
	currentPath := []string{}

	lines := strings.Split(data, "\n")
	for i := 0; i < len(lines); i++ {
		line := strings.TrimSpace(lines[i])

		// Skip comments and empty lines
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}

		// Table header: [section] or [section.subsection]
		if strings.HasPrefix(line, "[") && !strings.HasPrefix(line, "[[") {
			end := strings.Index(line, "]")
			if end < 0 {
				continue
			}
			sectionKey := strings.TrimSpace(line[1:end])
			parts := strings.Split(sectionKey, ".")
			currentPath = parts
			currentSection = ensureNestedMap(result, parts)
			continue
		}

		// Array of tables: [[section]]
		if strings.HasPrefix(line, "[[") {
			end := strings.Index(line, "]]")
			if end < 0 {
				continue
			}
			// We don't need array-of-tables for our use case, skip
			continue
		}

		// Key = value
		eqIdx := strings.Index(line, "=")
		if eqIdx < 0 {
			continue
		}
		key := strings.TrimSpace(line[:eqIdx])
		valStr := strings.TrimSpace(line[eqIdx+1:])

		// Handle dotted keys like tool.uv.package
		keyParts := strings.Split(key, ".")
		var target map[string]any
		var finalKey string
		if len(keyParts) > 1 {
			target = ensureNestedMap(currentSection, keyParts[:len(keyParts)-1])
			finalKey = keyParts[len(keyParts)-1]
		} else {
			target = currentSection
			finalKey = key
		}

		// Multi-line arrays
		if strings.HasPrefix(valStr, "[") && !strings.Contains(valStr, "]") {
			// Collect continuation lines
			for i+1 < len(lines) {
				i++
				cont := strings.TrimSpace(lines[i])
				valStr += " " + cont
				if strings.Contains(cont, "]") {
					break
				}
			}
		}

		// Multi-line inline tables
		if strings.HasPrefix(valStr, "{") && !strings.Contains(valStr, "}") {
			for i+1 < len(lines) {
				i++
				cont := strings.TrimSpace(lines[i])
				valStr += " " + cont
				if strings.Contains(cont, "}") {
					break
				}
			}
		}

		value := parseTOMLValue(valStr)
		// Use the full path for context when resolving dotted keys in sections
		_ = currentPath
		target[finalKey] = value
	}
	return result
}

// ensureNestedMap navigates/creates nested maps for a dotted key path.
func ensureNestedMap(root map[string]any, parts []string) map[string]any {
	current := root
	for _, part := range parts {
		if existing, ok := current[part]; ok {
			if m, ok := existing.(map[string]any); ok {
				current = m
			} else {
				// Overwrite non-map with map (shouldn't happen in valid TOML)
				m := make(map[string]any)
				current[part] = m
				current = m
			}
		} else {
			m := make(map[string]any)
			current[part] = m
			current = m
		}
	}
	return current
}

// parseTOMLValue parses a single TOML value from a string.
func parseTOMLValue(s string) any {
	s = strings.TrimSpace(s)

	// Boolean
	if s == "true" {
		return true
	}
	if s == "false" {
		return false
	}

	// String (double-quoted)
	if strings.HasPrefix(s, "\"") {
		return parseTOMLString(s)
	}
	// String (single-quoted / literal)
	if strings.HasPrefix(s, "'") {
		end := strings.LastIndex(s, "'")
		if end > 0 {
			return s[1:end]
		}
		return s[1:]
	}

	// Array
	if strings.HasPrefix(s, "[") {
		return parseTOMLArray(s)
	}

	// Inline table
	if strings.HasPrefix(s, "{") {
		return parseTOMLInlineTable(s)
	}

	// Number or other - return as string
	// Strip trailing comments
	if idx := strings.Index(s, " #"); idx >= 0 {
		s = strings.TrimSpace(s[:idx])
	}
	return s
}

// parseTOMLString extracts a double-quoted TOML string.
func parseTOMLString(s string) string {
	if len(s) < 2 || s[0] != '"' {
		return s
	}
	// Find closing quote, handling escapes
	result := strings.Builder{}
	i := 1
	for i < len(s) {
		if s[i] == '\\' && i+1 < len(s) {
			switch s[i+1] {
			case '"':
				result.WriteByte('"')
			case '\\':
				result.WriteByte('\\')
			case 'n':
				result.WriteByte('\n')
			case 't':
				result.WriteByte('\t')
			default:
				result.WriteByte('\\')
				result.WriteByte(s[i+1])
			}
			i += 2
			continue
		}
		if s[i] == '"' {
			break
		}
		result.WriteByte(s[i])
		i++
	}
	return result.String()
}

// parseTOMLArray parses a TOML array value like ["a", "b", "c"].
func parseTOMLArray(s string) []any {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "[") {
		return nil
	}

	// Find matching close bracket
	end := findMatchingBracket(s, 0, '[', ']')
	if end < 0 {
		end = len(s) - 1
	}
	inner := strings.TrimSpace(s[1:end])
	if inner == "" {
		return []any{}
	}

	var result []any
	for _, item := range splitTOMLItems(inner) {
		item = strings.TrimSpace(item)
		if item == "" {
			continue
		}
		result = append(result, parseTOMLValue(item))
	}
	return result
}

// parseTOMLInlineTable parses an inline table like { workspace = true }.
func parseTOMLInlineTable(s string) map[string]any {
	s = strings.TrimSpace(s)
	if !strings.HasPrefix(s, "{") {
		return nil
	}
	end := strings.LastIndex(s, "}")
	if end < 0 {
		end = len(s)
	}
	inner := strings.TrimSpace(s[1:end])
	if inner == "" {
		return map[string]any{}
	}

	result := make(map[string]any)
	for _, pair := range splitTOMLItems(inner) {
		pair = strings.TrimSpace(pair)
		if pair == "" {
			continue
		}
		eqIdx := strings.Index(pair, "=")
		if eqIdx < 0 {
			continue
		}
		key := strings.TrimSpace(pair[:eqIdx])
		val := strings.TrimSpace(pair[eqIdx+1:])
		result[key] = parseTOMLValue(val)
	}
	return result
}

// splitTOMLItems splits comma-separated TOML items, respecting nesting.
func splitTOMLItems(s string) []string {
	var items []string
	depth := 0
	inStr := false
	strChar := byte(0)
	start := 0

	for i := 0; i < len(s); i++ {
		ch := s[i]
		if inStr {
			if ch == '\\' {
				i++ // skip escape
				continue
			}
			if ch == strChar {
				inStr = false
			}
			continue
		}
		if ch == '"' || ch == '\'' {
			inStr = true
			strChar = ch
			continue
		}
		if ch == '[' || ch == '{' {
			depth++
			continue
		}
		if ch == ']' || ch == '}' {
			depth--
			continue
		}
		if ch == ',' && depth == 0 {
			items = append(items, s[start:i])
			start = i + 1
		}
	}
	if start < len(s) {
		items = append(items, s[start:])
	}
	return items
}

// findMatchingBracket finds the index of the matching close bracket.
func findMatchingBracket(s string, start int, open, close byte) int {
	depth := 0
	inStr := false
	strChar := byte(0)
	for i := start; i < len(s); i++ {
		ch := s[i]
		if inStr {
			if ch == '\\' {
				i++
				continue
			}
			if ch == strChar {
				inStr = false
			}
			continue
		}
		if ch == '"' || ch == '\'' {
			inStr = true
			strChar = ch
			continue
		}
		if ch == open {
			depth++
		} else if ch == close {
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

// ---------------------------------------------------------------------------
// Core helpers
// ---------------------------------------------------------------------------

var normalizeNamePattern = regexp.MustCompile(`[-_.]+`)

// normalizePackageName replaces sequences of [-_.] with - and lowercases.
func normalizePackageName(name string) string {
	return strings.ToLower(normalizeNamePattern.ReplaceAllString(name, "-"))
}

var depNamePattern = regexp.MustCompile(`^\s*([A-Za-z0-9][A-Za-z0-9._-]*)`)

// parseDependencyName extracts the package name from a PEP 508 string.
func parseDependencyName(dep, packageName, pyprojectPath string) (string, error) {
	match := depNamePattern.FindStringSubmatch(dep)
	if match == nil {
		return "", fmt.Errorf(
			"source.kind 'uv' only supports PEP 508 dependency strings "+
				"with an explicit package name. Could not parse dependency "+
				"%q in %s for package '%s'.",
			dep, pyprojectPath, packageName)
	}
	return normalizePackageName(match[1]), nil
}

// loadPyproject reads and parses a pyproject.toml file.
func loadPyproject(path string) (map[string]any, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("could not read %s: %w", path, err)
	}
	return parseTOML(string(data)), nil
}

// getNestedMap navigates a nested map via dotted key path.
func getNestedMap(m map[string]any, keys ...string) map[string]any {
	current := m
	for _, k := range keys {
		next, ok := current[k].(map[string]any)
		if !ok {
			return map[string]any{}
		}
		current = next
	}
	return current
}

// getNestedValue navigates a nested map and returns the final value.
func getNestedValue(m map[string]any, keys ...string) any {
	if len(keys) == 0 {
		return nil
	}
	current := m
	for _, k := range keys[:len(keys)-1] {
		next, ok := current[k].(map[string]any)
		if !ok {
			return nil
		}
		current = next
	}
	return current[keys[len(keys)-1]]
}

// ---------------------------------------------------------------------------
// Dependency resolution
// ---------------------------------------------------------------------------

// getDependencyNames parses and normalizes [project].dependencies.
func getDependencyNames(depSpecs any, packageName, pyprojectPath string) ([]string, error) {
	if depSpecs == nil {
		return nil, nil
	}
	arr, ok := depSpecs.([]any)
	if !ok {
		return nil, fmt.Errorf(
			"source.kind 'uv' requires [project].dependencies to be a "+
				"list of strings in %s.", pyprojectPath)
	}
	var names []string
	for _, item := range arr {
		s, ok := item.(string)
		if !ok {
			return nil, fmt.Errorf(
				"source.kind 'uv' requires [project].dependencies to be a "+
					"list of strings in %s.", pyprojectPath)
		}
		name, err := parseDependencyName(s, packageName, pyprojectPath)
		if err != nil {
			return nil, err
		}
		names = append(names, name)
	}
	return names, nil
}

// getUvLockPackageEnabled returns whether tool.uv.package is true (default: true).
func getUvLockPackageEnabled(pkg *UvLockPackage) (bool, error) {
	uvTool := pkg.RawUvTool
	if uvTool == nil {
		return true, nil
	}
	uvMap, ok := uvTool.(map[string]any)
	if !ok {
		return false, fmt.Errorf(
			"source.kind 'uv' requires [tool.uv] to be a table in %s.",
			pkg.PyprojectPath)
	}
	pkgVal, exists := uvMap["package"]
	if !exists {
		return true, nil
	}
	b, ok := pkgVal.(bool)
	if !ok {
		return false, fmt.Errorf(
			"source.kind 'uv' requires [tool.uv].package to be a boolean in %s.",
			pkg.PyprojectPath)
	}
	return b, nil
}

// getUvLockSourceEntries merges root-level and package-level [tool.uv.sources].
func getUvLockSourceEntries(
	pkg *UvLockPackage,
	projectRoot, rootPyprojectPath string,
	rawRootSourceEntries any,
) ([]UvLockSourceEntry, error) {
	rootSources, ok := rawRootSourceEntries.(map[string]any)
	if !ok {
		return nil, fmt.Errorf(
			"source.kind 'uv' requires [tool.uv.sources] to be a table in %s.",
			rootPyprojectPath)
	}

	var pkgSources map[string]any
	if uvMap, ok := pkg.RawUvTool.(map[string]any); ok {
		if s, ok := uvMap["sources"].(map[string]any); ok {
			pkgSources = s
		} else if uvMap["sources"] != nil {
			return nil, fmt.Errorf(
				"source.kind 'uv' requires [tool.uv.sources] to be a table in %s.",
				pkg.PyprojectPath)
		}
	}

	// Build merged entries: root first, then package-level overrides
	entryMap := make(map[string]UvLockSourceEntry)
	for name, val := range rootSources {
		entryMap[name] = UvLockSourceEntry{
			Name:          name,
			Value:         val,
			DeclaredRoot:  projectRoot,
			PyprojectPath: rootPyprojectPath,
		}
	}
	for name, val := range pkgSources {
		entryMap[name] = UvLockSourceEntry{
			Name:          name,
			Value:         val,
			DeclaredRoot:  pkg.Root,
			PyprojectPath: pkg.PyprojectPath,
		}
	}

	entries := make([]UvLockSourceEntry, 0, len(entryMap))
	for _, e := range entryMap {
		entries = append(entries, e)
	}
	return entries, nil
}

// validateUvLockSourceEntry validates a single source entry and returns
// the set of workspace dependency names it references.
func validateUvLockSourceEntry(
	sourceName string,
	sourceValue any,
	declaredRoot, pyprojectPath, projectRoot string,
	pkgsByName, pkgsByRoot map[string]*UvLockPackage,
) (map[string]bool, error) {
	wsDeps := make(map[string]bool)

	// Recurse into lists
	if arr, ok := sourceValue.([]any); ok {
		for _, item := range arr {
			sub, err := validateUvLockSourceEntry(
				sourceName, item, declaredRoot, pyprojectPath, projectRoot,
				pkgsByName, pkgsByRoot)
			if err != nil {
				return nil, err
			}
			for k := range sub {
				wsDeps[k] = true
			}
		}
		return wsDeps, nil
	}

	m, ok := sourceValue.(map[string]any)
	if !ok {
		return wsDeps, nil
	}

	normalizedSourceName := normalizePackageName(sourceName)

	// workspace = true
	if ws, ok := m["workspace"]; ok && ws == true {
		pkg := pkgsByName[normalizedSourceName]
		if pkg == nil {
			return nil, fmt.Errorf(
				"'%s' in %s is marked as `{ workspace = true }` but no matching "+
					"workspace package was found under project_root (%s). Check that "+
					"'%s' appears in [tool.uv.workspace].members.",
				sourceName, pyprojectPath, projectRoot, sourceName)
		}
		enabled, err := getUvLockPackageEnabled(pkg)
		if err != nil {
			return nil, err
		}
		if !enabled {
			return nil, fmt.Errorf(
				"'%s' in %s is a workspace dependency but sets "+
					"`tool.uv.package = false` in %s. "+
					"Workspace dependencies must be buildable packages.",
				sourceName, pyprojectPath, pkg.PyprojectPath)
		}
		wsDeps[pkg.NormalizedName] = true
	}

	// path = "..."
	if pathRef, ok := m["path"]; ok {
		pathStr, ok := pathRef.(string)
		if ok {
			if filepath.IsAbs(pathStr) || strings.HasPrefix(pathStr, "/") {
				return nil, fmt.Errorf(
					"'%s' in %s uses an absolute path (%s), which is not supported. "+
						"Use a relative path or `{ workspace = true }` instead.",
					sourceName, pyprojectPath, pathStr)
			}

			resolved, _ := filepath.Abs(filepath.Join(declaredRoot, pathStr))
			resolved = filepath.Clean(resolved)
			if !isEqualOrChild(resolved, projectRoot) {
				return nil, fmt.Errorf(
					"'%s' in %s uses a path source that resolves to %s, "+
						"which is outside project_root (%s).",
					sourceName, pyprojectPath, resolved, projectRoot)
			}

			pkg := pkgsByRoot[resolved]
			if pkg == nil {
				return nil, fmt.Errorf(
					"'%s' in %s uses a path source that resolves to %s, "+
						"which is not a workspace package under project_root (%s).",
					sourceName, pyprojectPath, resolved, projectRoot)
			}
			if pkg.NormalizedName != normalizedSourceName {
				return nil, fmt.Errorf(
					"'%s' in %s points to %s, which defines package '%s'. "+
						"The dependency name and the workspace package name must match.",
					sourceName, pyprojectPath, resolved, pkg.Name)
			}
			enabled, err := getUvLockPackageEnabled(pkg)
			if err != nil {
				return nil, err
			}
			if !enabled {
				return nil, fmt.Errorf(
					"'%s' in %s resolves to workspace package '%s', which sets "+
						"`tool.uv.package = false` in %s. "+
						"Workspace dependencies must be buildable packages.",
					sourceName, pyprojectPath, pkg.Name, pkg.PyprojectPath)
			}
			wsDeps[pkg.NormalizedName] = true
		}
	}

	// Recurse into other dict values
	for key, val := range m {
		if key == "workspace" || key == "path" {
			continue
		}
		sub, err := validateUvLockSourceEntry(
			sourceName, val, declaredRoot, pyprojectPath, projectRoot,
			pkgsByName, pkgsByRoot)
		if err != nil {
			return nil, err
		}
		for k := range sub {
			wsDeps[k] = true
		}
	}

	return wsDeps, nil
}

// validateUvLockPackage validates a package's dependencies and source entries.
func validateUvLockPackage(
	pkg *UvLockPackage,
	projectRoot, rootPyprojectPath string,
	rawRootSourceEntries any,
	pkgsByName, pkgsByRoot map[string]*UvLockPackage,
) error {
	depNames, err := getDependencyNames(
		pkg.RawDependencySpecs, pkg.Name, pkg.PyprojectPath)
	if err != nil {
		return err
	}
	depNameSet := make(map[string]bool, len(depNames))
	for _, n := range depNames {
		depNameSet[n] = true
	}

	wsDeps := make(map[string]bool)
	sourceEntries, err := getUvLockSourceEntries(
		pkg, projectRoot, rootPyprojectPath, rawRootSourceEntries)
	if err != nil {
		return err
	}

	for _, entry := range sourceEntries {
		if !depNameSet[normalizePackageName(entry.Name)] {
			continue
		}
		sub, err := validateUvLockSourceEntry(
			entry.Name, entry.Value,
			entry.DeclaredRoot, entry.PyprojectPath, projectRoot,
			pkgsByName, pkgsByRoot)
		if err != nil {
			return err
		}
		for k := range sub {
			wsDeps[k] = true
		}
	}

	enabled, err := getUvLockPackageEnabled(pkg)
	if err != nil {
		return err
	}
	pkg.PackageEnabled = enabled
	pkg.DependencyNames = depNames

	// Filter to workspace deps preserving dependency order
	var wsDepOrdered []string
	for _, n := range depNames {
		if wsDeps[n] {
			wsDepOrdered = append(wsDepOrdered, n)
		}
	}
	pkg.WorkspaceDependencies = wsDepOrdered
	return nil
}

// ---------------------------------------------------------------------------
// Workspace discovery
// ---------------------------------------------------------------------------

// discoverWorkspacePackages parses the root pyproject.toml, globs workspace
// members, and returns the full UvLockWorkspace.
func discoverWorkspacePackages(projectRoot, pyprojectPath string) (*UvLockWorkspace, error) {
	rootData, err := loadPyproject(pyprojectPath)
	if err != nil {
		return nil, err
	}

	rootSourceEntries := getNestedValue(rootData, "tool", "uv", "sources")
	if rootSourceEntries == nil {
		rootSourceEntries = map[string]any{}
	}

	var candidateRoots []string

	// Root project itself (if it has [project].name)
	rootProject := getNestedMap(rootData, "project")
	if name, _ := rootProject["name"].(string); name != "" {
		candidateRoots = append(candidateRoots, projectRoot)
	}

	// Workspace members
	workspaceMembers := getNestedValue(rootData, "tool", "uv", "workspace", "members")
	if workspaceMembers != nil {
		membersList, ok := workspaceMembers.([]any)
		if !ok {
			return nil, fmt.Errorf(
				"source.kind 'uv' requires [tool.uv.workspace].members to be a list.")
		}
		for _, patternAny := range membersList {
			pattern, ok := patternAny.(string)
			if !ok {
				return nil, fmt.Errorf(
					"source.kind 'uv' requires every [tool.uv.workspace].members " +
						"entry to be a string.")
			}
			globPattern := filepath.Join(projectRoot, pattern)
			matches, err := filepath.Glob(globPattern)
			if err != nil {
				// Invalid glob pattern; skip
				continue
			}
			sort.Strings(matches)
			for _, match := range matches {
				info, err := os.Stat(match)
				if err != nil {
					continue
				}
				pkgRoot := match
				if !info.IsDir() {
					pkgRoot = filepath.Dir(match)
				}
				pkgRoot, _ = filepath.Abs(pkgRoot)
				pkgRoot = filepath.Clean(pkgRoot)
				pyprojectFile := filepath.Join(pkgRoot, "pyproject.toml")
				if fi, err := os.Stat(pyprojectFile); err == nil && !fi.IsDir() {
					candidateRoots = append(candidateRoots, pkgRoot)
				}
			}
		}
	}

	// Deduplicate while preserving order
	seen := make(map[string]bool)
	var uniqueRoots []string
	for _, r := range candidateRoots {
		if !seen[r] {
			seen[r] = true
			uniqueRoots = append(uniqueRoots, r)
		}
	}

	// Parse each member
	var packages []*UvLockPackage
	for _, pkgRoot := range uniqueRoots {
		memberPyprojectPath := filepath.Join(pkgRoot, "pyproject.toml")
		pyData, err := loadPyproject(memberPyprojectPath)
		if err != nil {
			return nil, err
		}

		projectData := getNestedMap(pyData, "project")
		pkgName, _ := projectData["name"].(string)
		if pkgName == "" {
			return nil, fmt.Errorf(
				"source.kind 'uv' requires every workspace package to define "+
					"[project].name in %s.", memberPyprojectPath)
		}

		packages = append(packages, &UvLockPackage{
			Name:               pkgName,
			NormalizedName:     normalizePackageName(pkgName),
			Root:               pkgRoot,
			PyprojectPath:      memberPyprojectPath,
			RawDependencySpecs: projectData["dependencies"],
			RawUvTool:          getNestedValue(pyData, "tool", "uv"),
		})
	}

	pkgsByName := make(map[string]*UvLockPackage, len(packages))
	pkgsByRoot := make(map[string]*UvLockPackage, len(packages))
	for _, pkg := range packages {
		if existing, ok := pkgsByName[pkg.NormalizedName]; ok {
			return nil, fmt.Errorf(
				"source.kind 'uv' requires unique workspace package names, "+
					"but both %s and %s define '%s'.",
				existing.PyprojectPath, pkg.PyprojectPath, pkg.Name)
		}
		pkgsByName[pkg.NormalizedName] = pkg
		pkgsByRoot[pkg.Root] = pkg
	}

	return &UvLockWorkspace{
		RawRootSourceEntries: rootSourceEntries,
		PackagesByName:       pkgsByName,
		PackagesByRoot:       pkgsByRoot,
	}, nil
}

// ---------------------------------------------------------------------------
// Target inference
// ---------------------------------------------------------------------------

// inferTargetPackage determines which workspace package is the deployment target.
func inferTargetPackage(
	configRoot, projectRoot string,
	source map[string]any,
	pkgsByName, pkgsByRoot map[string]*UvLockPackage,
) (*UvLockPackage, error) {
	if pkgNameAny, exists := source["package"]; exists {
		pkgName, ok := pkgNameAny.(string)
		if !ok || strings.TrimSpace(pkgName) == "" {
			return nil, fmt.Errorf("`source.package` must be a non-empty string.")
		}
		target := pkgsByName[normalizePackageName(pkgName)]
		if target == nil {
			available := sortedPackageNames(pkgsByName)
			return nil, fmt.Errorf(
				"Could not find source.package '%s' in the uv project at %s. "+
					"It must match a [project].name from one of the discovered "+
					"packages. Available packages: %s.",
				pkgName, projectRoot, available)
		}
		return target, nil
	}

	// Find containing packages (sorted by depth, deepest first)
	type pkgWithDepth struct {
		pkg   *UvLockPackage
		depth int
	}
	var containing []pkgWithDepth
	for root, pkg := range pkgsByRoot {
		if configRoot == root || isEqualOrChild(configRoot, root) {
			parts := strings.Split(root, string(filepath.Separator))
			containing = append(containing, pkgWithDepth{pkg: pkg, depth: len(parts)})
		}
	}
	sort.Slice(containing, func(i, j int) bool {
		return containing[i].depth > containing[j].depth
	})

	if len(containing) > 0 {
		target := containing[0].pkg
		if target.Root != projectRoot ||
			len(pkgsByName) == 1 ||
			configRoot == projectRoot {
			return target, nil
		}
	}

	if len(pkgsByName) == 1 {
		for _, pkg := range pkgsByName {
			return pkg, nil
		}
	}

	available := sortedPackageNames(pkgsByName)
	return nil, fmt.Errorf(
		"source.package is required because source.root resolves to a uv "+
			"workspace with multiple packages and no unique target package could be "+
			"inferred from langgraph.json at %s. Available packages: %s. "+
			"Move langgraph.json into the target package or set source.package.",
		configRoot, available)
}

// ---------------------------------------------------------------------------
// Container path mapping
// ---------------------------------------------------------------------------

const containerWorkspaceRoot = "/deps/workspace"

// containerRootForPackage maps a package root to its container path.
func containerRootForPackage(projectRoot, packageRoot string) string {
	rel, err := filepath.Rel(projectRoot, packageRoot)
	if err != nil || rel == "." {
		return containerWorkspaceRoot
	}
	// Use forward slashes for container paths
	return containerWorkspaceRoot + "/" + filepath.ToSlash(rel)
}

// resolveUvLockContainerPath maps a host path to a container path using the plan.
func resolveUvLockContainerPath(hostPath string, plan *UvLockPlan) string {
	// Sort by depth (deepest first) so more specific roots match first
	type rootEntry struct {
		root string
		path string
	}
	var entries []rootEntry
	for root, cpath := range plan.ContainerRoots {
		entries = append(entries, rootEntry{root: root, path: cpath})
	}
	sort.Slice(entries, func(i, j int) bool {
		return len(strings.Split(entries[i].root, string(filepath.Separator))) >
			len(strings.Split(entries[j].root, string(filepath.Separator)))
	})

	for _, entry := range entries {
		if hostPath == entry.root || isEqualOrChild(hostPath, entry.root) {
			// Guard against workspace root matching unrelated members
			if len(plan.AllWorkspaceRoots) > 0 &&
				pathInUnrelatedMember(hostPath, entry.root, plan) {
				continue
			}
			rel, err := filepath.Rel(entry.root, hostPath)
			if err != nil {
				continue
			}
			if rel == "." {
				return entry.path
			}
			return entry.path + "/" + filepath.ToSlash(rel)
		}
	}
	return ""
}

// pathInUnrelatedMember returns true if hostPath is inside a workspace member
// that is NOT in the closure (container_roots).
func pathInUnrelatedMember(hostPath, matchedRoot string, plan *UvLockPlan) bool {
	for wsRoot := range plan.AllWorkspaceRoots {
		if wsRoot == matchedRoot {
			continue
		}
		if isEqualOrChild(matchedRoot, wsRoot) {
			// matchedRoot is inside wsRoot -> wsRoot is less specific -> skip
			continue
		}
		if hostPath == wsRoot || isEqualOrChild(hostPath, wsRoot) {
			if _, inClosure := plan.ContainerRoots[wsRoot]; !inClosure {
				return true
			}
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Copy items for workspace packages
// ---------------------------------------------------------------------------

// uvLockPackageCopyItems returns (source, destination) pairs for COPY/ADD.
// source is relative to projectRoot; destination is the container path.
func uvLockPackageCopyItems(pkg *UvLockPackage, plan *UvLockPlan) ([][2]string, error) {
	if pkg.Root != plan.ProjectRoot {
		rel, err := filepath.Rel(plan.ProjectRoot, pkg.Root)
		if err != nil {
			return nil, err
		}
		return [][2]string{{filepath.ToSlash(rel), plan.ContainerRoots[pkg.Root]}}, nil
	}

	// Root package: enumerate entries, skipping workspace member roots
	rootContainer := plan.ContainerRoots[pkg.Root]
	wsMemberRoots := make(map[string]bool)
	for wsRoot := range plan.AllWorkspaceRoots {
		if wsRoot != plan.ProjectRoot {
			wsMemberRoots[wsRoot] = true
		}
	}

	var iterEntries func(currentDir string) ([][2]string, error)
	iterEntries = func(currentDir string) ([][2]string, error) {
		dirEntries, err := os.ReadDir(currentDir)
		if err != nil {
			return nil, err
		}
		// Sort for deterministic output
		sort.Slice(dirEntries, func(i, j int) bool {
			return dirEntries[i].Name() < dirEntries[j].Name()
		})

		var result [][2]string
		for _, entry := range dirEntries {
			childPath := filepath.Join(currentDir, entry.Name())
			childAbs, _ := filepath.Abs(childPath)

			if wsMemberRoots[childAbs] {
				continue
			}

			// Check if any workspace member is a descendant
			hasDescendantMember := false
			if entry.IsDir() {
				for wsRoot := range wsMemberRoots {
					if isEqualOrChild(wsRoot, childAbs) && wsRoot != childAbs {
						hasDescendantMember = true
						break
					}
				}
			}

			if entry.IsDir() && hasDescendantMember {
				sub, err := iterEntries(childAbs)
				if err != nil {
					return nil, err
				}
				result = append(result, sub...)
				continue
			}

			relChild, err := filepath.Rel(plan.ProjectRoot, childAbs)
			if err != nil {
				continue
			}
			relPosix := filepath.ToSlash(relChild)
			result = append(result, [2]string{
				relPosix,
				rootContainer + "/" + relPosix,
			})
		}
		return result, nil
	}

	return iterEntries(plan.ProjectRoot)
}

// ---------------------------------------------------------------------------
// Plan construction
// ---------------------------------------------------------------------------

// planUvLockWorkspace is the main planning function: resolve paths, discover
// workspace, infer target, validate, build install order, compute container paths.
func planUvLockWorkspace(configPath string, config map[string]any) (*UvLockPlan, error) {
	configPathAbs, err := filepath.Abs(configPath)
	if err != nil {
		return nil, err
	}
	configRoot := filepath.Dir(configPathAbs)

	source, _ := config["source"].(map[string]any)
	root := "."
	if r, ok := source["root"].(string); ok && strings.TrimSpace(r) != "" {
		root = r
	}

	projectRoot, _ := filepath.Abs(filepath.Join(configRoot, root))
	projectRoot = filepath.Clean(projectRoot)
	pyprojectPath := filepath.Join(projectRoot, "pyproject.toml")
	uvLockPath := filepath.Join(projectRoot, "uv.lock")

	if _, err := os.Stat(uvLockPath); os.IsNotExist(err) {
		return nil, fmt.Errorf(
			"No uv.lock found at %s. Your langgraph.json sets "+
				"source.root=%q, which resolves to "+
				"%s. Make sure this is the directory where you run "+
				"`uv lock` (it should contain both pyproject.toml and uv.lock).",
			uvLockPath, root, projectRoot)
	}
	if _, err := os.Stat(pyprojectPath); os.IsNotExist(err) {
		return nil, fmt.Errorf(
			"No pyproject.toml found at %s. Your langgraph.json "+
				"sets source.root=%q, which resolves to "+
				"%s. This should be your uv workspace root.",
			pyprojectPath, root, projectRoot)
	}

	workspace, err := discoverWorkspacePackages(projectRoot, pyprojectPath)
	if err != nil {
		return nil, err
	}

	target, err := inferTargetPackage(
		configRoot, projectRoot, source,
		workspace.PackagesByName, workspace.PackagesByRoot)
	if err != nil {
		return nil, err
	}

	if err := validateUvLockPackage(
		target, projectRoot, pyprojectPath,
		workspace.RawRootSourceEntries,
		workspace.PackagesByName, workspace.PackagesByRoot,
	); err != nil {
		return nil, err
	}

	if !target.PackageEnabled {
		return nil, fmt.Errorf(
			"'%s' has `tool.uv.package = false` in %s, so it cannot be "+
				"deployed. Either remove that setting or point `source.package` "+
				"at a different workspace member.",
			target.Name, target.PyprojectPath)
	}

	// Build install order via DFS
	var installOrder []*UvLockPackage
	visited := make(map[string]bool)
	validated := map[string]bool{target.NormalizedName: true}

	var visit func(pkg *UvLockPackage) error
	visit = func(pkg *UvLockPackage) error {
		if visited[pkg.NormalizedName] {
			return nil
		}
		visited[pkg.NormalizedName] = true

		if !validated[pkg.NormalizedName] {
			if err := validateUvLockPackage(
				pkg, projectRoot, pyprojectPath,
				workspace.RawRootSourceEntries,
				workspace.PackagesByName, workspace.PackagesByRoot,
			); err != nil {
				return err
			}
			validated[pkg.NormalizedName] = true
		}

		for _, depName := range pkg.WorkspaceDependencies {
			dep := workspace.PackagesByName[depName]
			if dep != nil {
				if err := visit(dep); err != nil {
					return err
				}
			}
		}
		installOrder = append(installOrder, pkg)
		return nil
	}

	if err := visit(target); err != nil {
		return nil, err
	}

	// Container roots for the install closure
	containerRoots := make(map[string]string, len(installOrder))
	for _, pkg := range installOrder {
		containerRoots[pkg.Root] = containerRootForPackage(projectRoot, pkg.Root)
	}

	// All workspace roots (for exclusion logic)
	allWsRoots := make(map[string]bool)
	for _, pkg := range workspace.PackagesByName {
		allWsRoots[pkg.Root] = true
	}

	// Determine working dir by resolving configRoot
	tempPlan := &UvLockPlan{
		ProjectRoot:       projectRoot,
		PyprojectPath:     pyprojectPath,
		UvLockPath:        uvLockPath,
		Target:            target,
		TargetRoot:        target.Root,
		InstallOrder:      installOrder,
		ContainerRoots:    containerRoots,
		WorkingDir:        containerRootForPackage(projectRoot, target.Root),
		AllWorkspaceRoots: allWsRoots,
	}
	workingDir := resolveUvLockContainerPath(configRoot, tempPlan)
	if workingDir == "" {
		workingDir = containerRoots[target.Root]
	}

	return &UvLockPlan{
		ProjectRoot:       projectRoot,
		PyprojectPath:     pyprojectPath,
		UvLockPath:        uvLockPath,
		Target:            target,
		TargetRoot:        target.Root,
		InstallOrder:      installOrder,
		ContainerRoots:    containerRoots,
		WorkingDir:        workingDir,
		AllWorkspaceRoots: allWsRoots,
	}, nil
}

// ---------------------------------------------------------------------------
// Import path rewriting
// ---------------------------------------------------------------------------

// rewriteUvLockImportPath rewrites a "module:attr" import string so that the
// module path points to the correct container location.
func rewriteUvLockImportPath(
	configPath, importStr string, plan *UvLockPlan, label string,
) (string, error) {
	parts := strings.SplitN(importStr, ":", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return "", fmt.Errorf(
			"Import string %q must be in format \"<module>:<attribute>\".",
			importStr)
	}
	moduleStr := parts[0]
	attrStr := parts[1]

	if !strings.Contains(moduleStr, "/") && !strings.Contains(moduleStr, "\\") {
		return importStr, nil
	}

	configDir := filepath.Dir(configPath)
	resolved, _ := filepath.Abs(filepath.Join(configDir, moduleStr))
	resolved = filepath.Clean(resolved)

	info, err := os.Stat(resolved)
	if err != nil {
		return "", fmt.Errorf("Could not find %s: %s", label, resolved)
	}
	if info.IsDir() {
		return "", fmt.Errorf("%s must be a file: %s",
			strings.ToUpper(label[:1])+label[1:], resolved)
	}

	containerPath := resolveUvLockContainerPath(resolved, plan)
	if containerPath == "" {
		var copiedDirs []string
		for _, pkg := range plan.InstallOrder {
			rel, err := filepath.Rel(plan.ProjectRoot, pkg.Root)
			if err != nil || rel == "." {
				copiedDirs = append(copiedDirs, ".")
			} else {
				copiedDirs = append(copiedDirs, filepath.ToSlash(rel))
			}
		}
		return "", fmt.Errorf(
			"%s '%s' resolves to %s, which is not inside the target "+
				"package '%s' or any of its workspace dependencies. Only these "+
				"directories are copied into the container: %s. If this file "+
				"lives in another workspace package, add it as a dependency of "+
				"'%s' with `{ workspace = true }` in [tool.uv.sources].",
			strings.ToUpper(label[:1])+label[1:],
			importStr, resolved, plan.Target.Name,
			strings.Join(copiedDirs, ", "), plan.Target.Name)
	}

	return containerPath + ":" + attrStr, nil
}

// updateUvLockGraphPaths rewrites graph import paths for uv-lock mode.
func updateUvLockGraphPaths(configPath string, config map[string]any, plan *UvLockPlan) error {
	graphs, _ := config["graphs"].(map[string]any)
	for graphID, data := range graphs {
		switch v := data.(type) {
		case map[string]any:
			pathStr, ok := v["path"].(string)
			if !ok || pathStr == "" {
				return fmt.Errorf(
					"Graph '%s' must contain a 'path' key if it is a dictionary.",
					graphID)
			}
			rewritten, err := rewriteUvLockImportPath(
				configPath, pathStr, plan, fmt.Sprintf("graph '%s'", graphID))
			if err != nil {
				return err
			}
			v["path"] = rewritten
		case string:
			rewritten, err := rewriteUvLockImportPath(
				configPath, v, plan, fmt.Sprintf("graph '%s'", graphID))
			if err != nil {
				return err
			}
			graphs[graphID] = rewritten
		default:
			return fmt.Errorf(
				"Graph '%s' must be a string or a dictionary with a 'path' key.",
				graphID)
		}
	}
	return nil
}

// updateUvLockComponentPath rewrites a single section.key import path.
func updateUvLockComponentPath(
	configPath string, config map[string]any, plan *UvLockPlan,
	section, key, label string,
) error {
	sectionMap, ok := config[section].(map[string]any)
	if !ok {
		return nil
	}
	pathStr, ok := sectionMap[key].(string)
	if !ok || pathStr == "" {
		return nil
	}
	rewritten, err := rewriteUvLockImportPath(configPath, pathStr, plan, label)
	if err != nil {
		return err
	}
	sectionMap[key] = rewritten
	return nil
}

// updateUvLockUIPaths rewrites UI file paths for uv-lock mode.
func updateUvLockUIPaths(configPath string, config map[string]any, plan *UvLockPlan) error {
	ui, ok := config["ui"].(map[string]any)
	if !ok {
		return nil
	}

	configDir := filepath.Dir(configPath)
	for uiName, pathAny := range ui {
		pathStr, ok := pathAny.(string)
		if !ok {
			continue
		}

		resolved, _ := filepath.Abs(filepath.Join(configDir, pathStr))
		resolved = filepath.Clean(resolved)

		info, err := os.Stat(resolved)
		if err != nil {
			return fmt.Errorf("Could not find ui '%s': %s", uiName, resolved)
		}
		if info.IsDir() {
			return fmt.Errorf("Ui '%s' must be a file: %s", uiName, resolved)
		}

		containerPath := resolveUvLockContainerPath(resolved, plan)
		if containerPath == "" {
			var copiedDirs []string
			for _, pkg := range plan.InstallOrder {
				rel, err := filepath.Rel(plan.ProjectRoot, pkg.Root)
				if err != nil || rel == "." {
					copiedDirs = append(copiedDirs, ".")
				} else {
					copiedDirs = append(copiedDirs, filepath.ToSlash(rel))
				}
			}
			return fmt.Errorf(
				"Ui '%s' resolves to %s, which is not inside the target "+
					"package '%s' or any of its workspace dependencies. Only these "+
					"directories are copied into the container: %s. If this file "+
					"lives in another workspace package, add it as a dependency of "+
					"'%s' with `{ workspace = true }` in [tool.uv.sources].",
				uiName, resolved, plan.Target.Name,
				strings.Join(copiedDirs, ", "), plan.Target.Name)
		}

		ui[uiName] = containerPath
	}
	return nil
}

// ---------------------------------------------------------------------------
// Dockerfile generation
// ---------------------------------------------------------------------------

// PythonConfigToDockerUVLock generates a Dockerfile and additional build contexts
// for a Python-based LangGraph configuration using uv lock mode.
func PythonConfigToDockerUVLock(
	configPath string,
	config map[string]any,
	baseImage string,
	apiVersion string,
	buildToolsToUninstall []string,
) (string, map[string]string, error) {

	if !ImageSupportsUV(baseImage) {
		return "", nil, fmt.Errorf(
			"source.kind 'uv' requires a base image with uv support " +
				"(langchain/langgraph-api >= 0.2.47)")
	}

	configPathAbs, _ := filepath.Abs(configPath)
	configRoot := filepath.Dir(configPathAbs)

	installCmd := "uv pip install --system"
	_, globalReqsPipInstall, pipConfigFileStr := buildPythonInstallCommands(config, installCmd)

	plan, err := planUvLockWorkspace(configPath, config)
	if err != nil {
		return "", nil, err
	}

	// Rewrite graph paths
	if err := updateUvLockGraphPaths(configPathAbs, config, plan); err != nil {
		return "", nil, err
	}

	// Rewrite component paths
	for _, sc := range [][3]string{
		{"auth", "path", "auth.path"},
		{"encryption", "path", "encryption.path"},
		{"checkpointer", "path", "checkpointer.path"},
		{"http", "app", "http.app"},
	} {
		if err := updateUvLockComponentPath(
			configPathAbs, config, plan, sc[0], sc[1], sc[2]); err != nil {
			return "", nil, err
		}
	}

	// Rewrite UI paths
	if err := updateUvLockUIPaths(configPathAbs, config, plan); err != nil {
		return "", nil, err
	}

	// Additional contexts
	additionalContexts := make(map[string]string)
	var workspaceContextName string
	if plan.ProjectRoot != configRoot && !isEqualOrChild(plan.ProjectRoot, configRoot) {
		workspaceContextName = "uv-workspace-root"
		additionalContexts[workspaceContextName] = plan.ProjectRoot
	}

	copyFromProjectRoot := func(relativePath, destination string) string {
		if workspaceContextName != "" {
			source := relativePath
			if source == "" || source == "." {
				source = "."
			}
			return fmt.Sprintf("COPY --from=%s %s %s", workspaceContextName, source, destination)
		}
		sourcePath := filepath.Join(plan.ProjectRoot, relativePath)
		relSource, _ := filepath.Rel(configRoot, sourcePath)
		return fmt.Sprintf("ADD %s %s", filepath.ToSlash(relSource), destination)
	}

	uvExportProjectDir := "/tmp/uv_export/project"
	envVars := BuildRuntimeEnvVars(config)
	imageStr := DockerTag(config, baseImage, apiVersion)

	var lines []string

	if len(additionalContexts) > 0 {
		lines = append(lines, "# syntax=docker/dockerfile:1.4", "")
	}

	lines = append(lines, fmt.Sprintf("FROM %s", imageStr), "")

	// dockerfile_lines
	dfLines := configSlice(config, "dockerfile_lines")
	if len(dfLines) > 0 {
		for _, l := range dfLines {
			if s, ok := l.(string); ok && s != "" {
				lines = append(lines, s)
			}
		}
		lines = append(lines, "")
	}

	// install node
	nv, _ := config["node_version"].(string)
	if (config["ui"] != nil || nv != "") && plan.WorkingDir != "" {
		lines = append(lines, "RUN /storage/install-node.sh", "")
	}

	// pip config
	if pipConfigFileStr != "" {
		lines = append(lines, pipConfigFileStr, "")
	}

	// -- Installing dependencies from uv.lock --
	lines = append(lines, "# -- Installing dependencies from uv.lock --")
	lines = append(lines,
		copyFromProjectRoot("pyproject.toml", uvExportProjectDir+"/pyproject.toml"))
	lines = append(lines,
		copyFromProjectRoot("uv.lock", uvExportProjectDir+"/uv.lock"))

	// Copy workspace member pyproject.toml files into the export dir
	// so that `uv export --package` can resolve workspace dependencies.
	for _, pkg := range plan.InstallOrder {
		if pkg.Root == plan.ProjectRoot {
			continue
		}
		rel, err := filepath.Rel(plan.ProjectRoot, pkg.PyprojectPath)
		if err != nil {
			continue
		}
		relPosix := filepath.ToSlash(rel)
		lines = append(lines,
			copyFromProjectRoot(relPosix, uvExportProjectDir+"/"+relPosix))
	}

	lines = append(lines, fmt.Sprintf("WORKDIR %s", uvExportProjectDir))

	quotedName := shellQuote(plan.Target.Name)
	lines = append(lines, fmt.Sprintf("RUN uv export --package %s --frozen --no-hashes --no-emit-project --no-emit-workspace -o uv_requirements.txt", quotedName))
	lines = append(lines, fmt.Sprintf("RUN %s -r uv_requirements.txt", globalReqsPipInstall))
	lines = append(lines, "RUN rm -rf /tmp/uv_export")
	lines = append(lines, "# -- End of uv.lock dependencies install --", "")

	// Add workspace packages in install order
	for _, pkg := range plan.InstallOrder {
		rel, err := filepath.Rel(plan.ProjectRoot, pkg.Root)
		if err != nil {
			rel = "."
		}
		packageLabel := filepath.ToSlash(rel)
		if packageLabel == "." {
			packageLabel = "."
		}

		lines = append(lines, fmt.Sprintf("# -- Adding workspace package %s --", packageLabel))

		copyItems, err := uvLockPackageCopyItems(pkg, plan)
		if err != nil {
			return "", nil, err
		}
		for _, item := range copyItems {
			lines = append(lines, copyFromProjectRoot(item[0], item[1]))
		}

		lines = append(lines, fmt.Sprintf("WORKDIR %s", plan.ContainerRoots[pkg.Root]))
		lines = append(lines, fmt.Sprintf("RUN %s --no-deps -e .", globalReqsPipInstall))
		lines = append(lines, fmt.Sprintf("# -- End of workspace package %s --", packageLabel), "")
	}

	// env vars
	if len(envVars) > 0 {
		lines = append(lines, envVars...)
		lines = append(lines, "")
	}

	// JS install
	if (config["ui"] != nil || nv != "") && plan.WorkingDir != "" {
		nodeVer := nv
		if nodeVer == "" {
			nodeVer = DefaultNodeVersion
		}
		lines = append(lines,
			"# -- Installing JS dependencies --",
			fmt.Sprintf("ENV NODE_VERSION=%s", nodeVer),
			fmt.Sprintf("WORKDIR %s", plan.WorkingDir),
			fmt.Sprintf("RUN %s && tsx /api/langgraph_api/js/build.mts",
				GetNodePMInstallCmd(plan.TargetRoot)),
			"# -- End of JS dependencies install --",
			"",
		)
	}

	// pip cleanup
	lines = append(lines,
		GetPipCleanupLines(installCmd, buildToolsToUninstall, "uv"),
		"",
	)

	// working dir
	if plan.WorkingDir != "" {
		lines = append(lines, fmt.Sprintf("WORKDIR %s", plan.WorkingDir))
	}

	return strings.Join(lines, "\n"), additionalContexts, nil
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

// isEqualOrChild returns true if child equals parent or is a subdirectory of parent.
func isEqualOrChild(child, parent string) bool {
	if child == parent {
		return true
	}
	rel, err := filepath.Rel(parent, child)
	if err != nil {
		return false
	}
	return !strings.HasPrefix(rel, "..")
}

// sortedPackageNames returns a sorted comma-separated list of package names,
// or "(none)" if the map is empty.
func sortedPackageNames(pkgsByName map[string]*UvLockPackage) string {
	if len(pkgsByName) == 0 {
		return "(none)"
	}
	names := make([]string, 0, len(pkgsByName))
	for _, pkg := range pkgsByName {
		names = append(names, pkg.Name)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}
