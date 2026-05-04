package config

import (
	"encoding/json"
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

// PipReq represents a (hostPath, containerPath) pair for a requirements.txt.
type PipReq struct {
	HostPath      string
	ContainerPath string
}

// RealPkg represents a real Python package (has pyproject.toml or setup.py).
type RealPkg struct {
	RelPath       string
	ContainerName string
}

// FauxPkg represents a directory without packaging metadata (faux package).
type FauxPkg struct {
	RelPath       string
	ContainerPath string
}

// LocalDeps holds all resolved local dependency information for Dockerfile generation.
type LocalDeps struct {
	PipReqs            []PipReq
	RealPkgs           map[string]RealPkg // hostPath -> RealPkg
	FauxPkgs           map[string]FauxPkg // hostPath -> FauxPkg
	WorkingDir         string             // "" if not set
	AdditionalContexts []string           // resolved paths needing extra build contexts
}

// DockerOpts groups options for ConfigToDocker.
type DockerOpts struct {
	BaseImage       string
	APIVersion      string
	InstallCommand  string
	BuildCommand    string
	BuildContext    string
	EscapeVariables bool
}

// ComposeOpts groups options for ConfigToCompose.
type ComposeOpts struct {
	BaseImage         string
	APIVersion        string
	Image             string
	Watch             bool
	EngineRuntimeMode string
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

var buildTools = []string{"pip", "setuptools", "wheel"}

var reservedPackageNames = map[string]bool{
	"src":            true,
	"langgraph-api":  true,
	"langgraph_api":  true,
	"langgraph":      true,
	"langchain-core": true,
	"langchain_core": true,
	"pydantic":       true,
	"orjson":         true,
	"fastapi":        true,
	"uvicorn":        true,
	"psycopg":        true,
	"httpx":          true,
	"langsmith":      true,
}

var semverPattern = regexp.MustCompile(`:(\d+(?:\.\d+)?(?:\.\d+)?)(?:-|$)`)

// ---------------------------------------------------------------------------
// 1. DefaultBaseImage
// ---------------------------------------------------------------------------

// DefaultBaseImage returns the base Docker image for a config.
func DefaultBaseImage(config map[string]any, engineRuntimeMode string) string {
	if bi, _ := config["base_image"].(string); bi != "" {
		return bi
	}
	nv, _ := config["node_version"].(string)
	pv, _ := config["python_version"].(string)
	if nv != "" && pv == "" {
		return "langchain/langgraphjs-api"
	}
	if engineRuntimeMode == "distributed" {
		return "langchain/langgraph-executor"
	}
	return "langchain/langgraph-api"
}

// ---------------------------------------------------------------------------
// 2. DockerTag
// ---------------------------------------------------------------------------

// DockerTag computes the full image:tag string for a config.
func DockerTag(config map[string]any, baseImage, apiVersion string) string {
	if apiVersion == "" {
		apiVersion, _ = config["api_version"].(string)
	}
	if baseImage == "" {
		baseImage = DefaultBaseImage(config, "combined_queue_worker")
	}

	imageDistro, _ := config["image_distro"].(string)
	distroTag := ""
	if imageDistro != "" && imageDistro != DefaultImageDistro {
		distroTag = "-" + imageDistro
	}

	if tag, _ := config["_INTERNAL_docker_tag"].(string); tag != "" {
		return baseImage + ":" + tag
	}

	nv, _ := config["node_version"].(string)
	pv, _ := config["python_version"].(string)

	var language, version string
	if nv != "" && pv == "" {
		language = "node"
		version = nv
	} else {
		language = "py"
		version = pv
	}

	versionDistroTag := version + distroTag

	if apiVersion != "" {
		fullTag := apiVersion + "-" + language + versionDistroTag
		return baseImage + ":" + fullTag
	}
	if strings.Contains(baseImage, "/langgraph-server") && !strings.Contains(baseImage, versionDistroTag) {
		return baseImage + "-" + language + versionDistroTag
	}
	return baseImage + ":" + versionDistroTag
}

// ---------------------------------------------------------------------------
// 4. AssembleLocalDeps
// ---------------------------------------------------------------------------

// AssembleLocalDeps inspects the config's dependencies list and classifies
// each local (dot-prefixed) dependency as a real package, faux package, etc.
func AssembleLocalDeps(configPath string, config map[string]any) (*LocalDeps, error) {
	configPath, err := filepath.Abs(configPath)
	if err != nil {
		return nil, err
	}
	configDir := filepath.Dir(configPath)

	reserved := make(map[string]bool)
	for k, v := range reservedPackageNames {
		reserved[k] = v
	}

	checkReserved := func(name, ref string) error {
		if reserved[name] {
			return fmt.Errorf(
				"Package name '%s' used in local dep '%s' is reserved. "+
					"Rename the directory.", name, ref)
		}
		reserved[name] = true
		return nil
	}

	counter := map[string]int{}
	var pipReqs []PipReq
	realPkgs := map[string]RealPkg{}
	fauxPkgs := map[string]FauxPkg{}
	workingDir := ""
	var additionalContexts []string

	deps := configSlice(config, "dependencies")
	for _, depAny := range deps {
		localDep, ok := depAny.(string)
		if !ok || !strings.HasPrefix(localDep, ".") {
			continue
		}

		resolved, err := filepath.Abs(filepath.Join(configDir, localDep))
		if err != nil {
			return nil, err
		}

		info, err := os.Stat(resolved)
		if err != nil {
			return nil, fmt.Errorf("Could not find local dependency: %s", resolved)
		}
		if !info.IsDir() {
			return nil, fmt.Errorf("Local dependency must be a directory: %s", resolved)
		}

		// Check if resolved is same as configDir or a child
		if resolved != configDir {
			// Check if configDir is a parent of resolved
			rel, relErr := filepath.Rel(configDir, resolved)
			if relErr != nil || strings.HasPrefix(rel, "..") {
				additionalContexts = append(additionalContexts, resolved)
			}
		}

		entries, err := os.ReadDir(resolved)
		if err != nil {
			return nil, err
		}
		fileNames := make(map[string]bool)
		for _, e := range entries {
			fileNames[e.Name()] = true
		}

		if fileNames["pyproject.toml"] || fileNames["setup.py"] {
			// Real package
			containerName := filepath.Base(resolved)
			if counter[containerName] > 0 {
				containerName = fmt.Sprintf("%s_%d", containerName, counter[containerName])
			}
			counter[containerName]++

			realPkgs[resolved] = RealPkg{RelPath: localDep, ContainerName: containerName}
			if localDep == "." {
				workingDir = "/deps/" + containerName
			}
		} else {
			baseName := filepath.Base(resolved)
			var containerPath string

			if fileNames["__init__.py"] {
				// Flat layout
				if strings.Contains(baseName, "-") {
					return nil, fmt.Errorf(
						"Package name '%s' contains a hyphen. "+
							"Rename the directory to use it as flat-layout package.",
						baseName)
				}
				if err := checkReserved(baseName, localDep); err != nil {
					return nil, err
				}
				containerPath = fmt.Sprintf("/deps/outer-%s/%s", baseName, baseName)
			} else {
				// Src layout
				containerPath = fmt.Sprintf("/deps/outer-%s/src", baseName)
				for _, entry := range entries {
					if !entry.IsDir() || entry.Name() == "__pycache__" || strings.HasPrefix(entry.Name(), ".") {
						continue
					}
					subDir := filepath.Join(resolved, entry.Name())
					subEntries, subErr := os.ReadDir(subDir)
					if subErr != nil {
						continue // permission error etc.
					}
					for _, sf := range subEntries {
						if strings.HasSuffix(sf.Name(), ".py") {
							if err := checkReserved(entry.Name(), localDep); err != nil {
								return nil, err
							}
							break
						}
					}
				}
			}

			fauxPkgs[resolved] = FauxPkg{RelPath: localDep, ContainerPath: containerPath}
			if localDep == "." {
				workingDir = containerPath
			}

			if fileNames["requirements.txt"] {
				rfile := filepath.Join(resolved, "requirements.txt")
				pipReqs = append(pipReqs, PipReq{
					HostPath:      rfile,
					ContainerPath: containerPath + "/requirements.txt",
				})
			}
		}
	}

	return &LocalDeps{
		PipReqs:            pipReqs,
		RealPkgs:           realPkgs,
		FauxPkgs:           fauxPkgs,
		WorkingDir:         workingDir,
		AdditionalContexts: additionalContexts,
	}, nil
}

// ---------------------------------------------------------------------------
// 5. UpdateGraphPaths
// ---------------------------------------------------------------------------

// UpdateGraphPaths remaps each graph's import path to the correct in-container path.
func UpdateGraphPaths(configPath string, config map[string]any, deps *LocalDeps) error {
	configPath, _ = filepath.Abs(configPath)
	configDir := filepath.Dir(configPath)

	graphs, _ := config["graphs"].(map[string]any)
	for graphID, data := range graphs {
		var importStr string
		switch v := data.(type) {
		case string:
			importStr = v
		case map[string]any:
			p, ok := v["path"].(string)
			if !ok || p == "" {
				return fmt.Errorf(
					"Graph '%s' must contain a 'path' key if  it is a dictionary.",
					graphID)
			}
			importStr = p
		default:
			return fmt.Errorf(
				"Graph '%s' must be a string or a dictionary with a 'path' key.",
				graphID)
		}

		parts := strings.SplitN(importStr, ":", 2)
		if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
			return fmt.Errorf(
				"Import string \"%s\" must be in format \"<module>:<attribute>\".",
				importStr)
		}
		moduleStr := parts[0]
		attrStr := parts[1]

		if strings.Contains(moduleStr, "/") || strings.Contains(moduleStr, "\\") {
			resolved, err := filepath.Abs(filepath.Join(configDir, moduleStr))
			if err != nil {
				return err
			}
			info, statErr := os.Stat(resolved)
			if statErr != nil {
				return fmt.Errorf("Could not find local module: %s", resolved)
			}
			if info.IsDir() {
				return fmt.Errorf("Local module must be a file: %s", resolved)
			}

			found := false
			// Check real packages
			for pkgPath, pkg := range deps.RealPkgs {
				rel, relErr := filepath.Rel(pkgPath, resolved)
				if relErr == nil && !strings.HasPrefix(rel, "..") {
					containerPath := "/deps/" + pkg.ContainerName + "/" + filepath.ToSlash(rel)
					moduleStr = containerPath
					found = true
					break
				}
			}

			if !found {
				// Check faux packages
				for fauxPath, faux := range deps.FauxPkgs {
					rel, relErr := filepath.Rel(fauxPath, resolved)
					if relErr == nil && !strings.HasPrefix(rel, "..") {
						moduleStr = faux.ContainerPath + "/" + filepath.ToSlash(rel)
						found = true
						break
					}
				}
			}

			if !found {
				return fmt.Errorf(
					"Module '%s' not found in 'dependencies' list. "+
						"Add its containing package to 'dependencies' list.",
					importStr)
			}

			// Update config
			switch data.(type) {
			case map[string]any:
				data.(map[string]any)["path"] = moduleStr + ":" + attrStr
			default:
				graphs[graphID] = moduleStr + ":" + attrStr
			}
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// 6. UpdateConfigPaths
// ---------------------------------------------------------------------------

// UpdateConfigPaths rewrites auth, encryption, checkpointer, and http paths
// to point to the correct location inside the Docker container.
func UpdateConfigPaths(configPath string, config map[string]any, deps *LocalDeps) error {
	configPath, _ = filepath.Abs(configPath)
	configDir := filepath.Dir(configPath)

	if err := updateModulePath(configDir, config, deps, "auth", "path", "Auth file"); err != nil {
		return err
	}
	if err := updateModulePath(configDir, config, deps, "encryption", "path", "Encryption file"); err != nil {
		return err
	}
	if err := updateModulePath(configDir, config, deps, "checkpointer", "path", "Checkpointer file"); err != nil {
		return err
	}
	if err := updateHTTPAppPath(configDir, config, deps); err != nil {
		return err
	}
	return nil
}

// updateModulePath is a helper for auth.path, encryption.path, checkpointer.path.
func updateModulePath(configDir string, config map[string]any, deps *LocalDeps, section, key, label string) error {
	sectionMap, ok := config[section].(map[string]any)
	if !ok || sectionMap == nil {
		return nil
	}
	pathStr, _ := sectionMap[key].(string)
	if pathStr == "" {
		return nil
	}

	parts := strings.SplitN(pathStr, ":", 2)
	if len(parts) != 2 {
		return nil // already validated elsewhere
	}
	moduleStr := parts[0]
	attrStr := parts[1]

	if !strings.HasPrefix(moduleStr, ".") {
		return nil // absolute path or module import
	}

	resolved, err := filepath.Abs(filepath.Join(configDir, moduleStr))
	if err != nil {
		return err
	}
	info, statErr := os.Stat(resolved)
	if statErr != nil {
		return fmt.Errorf("%s not found: %s (from %s)", label, resolved, pathStr)
	}
	if info.IsDir() {
		return fmt.Errorf("%s path must be a file: %s", label, resolved)
	}

	// Check faux packages first (higher priority)
	for fauxPath, faux := range deps.FauxPkgs {
		rel, relErr := filepath.Rel(fauxPath, resolved)
		if relErr == nil && !strings.HasPrefix(rel, "..") {
			sectionMap[key] = faux.ContainerPath + "/" + filepath.ToSlash(rel) + ":" + attrStr
			return nil
		}
	}

	// Check real packages
	for realPath, pkg := range deps.RealPkgs {
		rel, relErr := filepath.Rel(realPath, resolved)
		if relErr == nil && !strings.HasPrefix(rel, "..") {
			sectionMap[key] = "/deps/" + filepath.Base(realPath) + "/" + filepath.ToSlash(rel) + ":" + attrStr
			_ = pkg // use the real_path base, matching Python
			return nil
		}
	}

	depsJSON, _ := json.Marshal(config["dependencies"])
	return fmt.Errorf(
		"%s '%s' not covered by dependencies.\n"+
			"Add its parent directory to the 'dependencies' array in your config.\n"+
			"Current dependencies: %s",
		label, resolved, string(depsJSON))
}

// updateHTTPAppPath handles http.app path remapping.
func updateHTTPAppPath(configDir string, config map[string]any, deps *LocalDeps) error {
	httpConf, ok := config["http"].(map[string]any)
	if !ok || httpConf == nil {
		return nil
	}
	appStr, _ := httpConf["app"].(string)
	if appStr == "" {
		return nil
	}

	parts := strings.SplitN(appStr, ":", 2)
	if len(parts) != 2 || parts[0] == "" || parts[1] == "" {
		return fmt.Errorf(
			"Import string \"%s\" must be in format \"<module>:<attribute>\".",
			appStr)
	}
	moduleStr := parts[0]
	attrStr := parts[1]

	if !strings.Contains(moduleStr, "/") && !strings.Contains(moduleStr, "\\") {
		return nil // not a file path
	}

	resolved, err := filepath.Abs(filepath.Join(configDir, moduleStr))
	if err != nil {
		return err
	}
	info, statErr := os.Stat(resolved)
	if statErr != nil {
		return fmt.Errorf("Could not find HTTP app module: %s", resolved)
	}
	if info.IsDir() {
		return fmt.Errorf("HTTP app module must be a file: %s", resolved)
	}

	// Check real packages
	for pkgPath, pkg := range deps.RealPkgs {
		rel, relErr := filepath.Rel(pkgPath, resolved)
		if relErr == nil && !strings.HasPrefix(rel, "..") {
			containerPath := "/deps/" + pkg.ContainerName + "/" + filepath.ToSlash(rel)
			httpConf["app"] = containerPath + ":" + attrStr
			return nil
		}
	}

	// Check faux packages
	for fauxPath, faux := range deps.FauxPkgs {
		rel, relErr := filepath.Rel(fauxPath, resolved)
		if relErr == nil && !strings.HasPrefix(rel, "..") {
			httpConf["app"] = faux.ContainerPath + "/" + filepath.ToSlash(rel) + ":" + attrStr
			return nil
		}
	}

	return fmt.Errorf(
		"HTTP app module '%s' not found in 'dependencies' list. "+
			"Add its containing package to 'dependencies' list.",
		appStr)
}

// ---------------------------------------------------------------------------
// 7. BuildRuntimeEnvVars
// ---------------------------------------------------------------------------

// BuildRuntimeEnvVars generates ENV lines for the Dockerfile from config sections.
func BuildRuntimeEnvVars(config map[string]any) []string {
	var envVars []string

	envSections := []struct {
		key    string
		envVar string
	}{
		{"store", "LANGGRAPH_STORE"},
		{"auth", "LANGGRAPH_AUTH"},
		{"encryption", "LANGGRAPH_ENCRYPTION"},
		{"http", "LANGGRAPH_HTTP"},
		{"webhooks", "LANGGRAPH_WEBHOOKS"},
		{"checkpointer", "LANGGRAPH_CHECKPOINTER"},
		{"ui", "LANGGRAPH_UI"},
		{"ui_config", "LANGGRAPH_UI_CONFIG"},
	}

	for _, s := range envSections {
		if val := config[s.key]; val != nil {
			j, _ := json.Marshal(val)
			envVars = append(envVars, fmt.Sprintf("ENV %s='%s'", s.envVar, string(j)))
		}
	}

	graphsJSON, _ := json.Marshal(config["graphs"])
	envVars = append(envVars, fmt.Sprintf("ENV LANGSERVE_GRAPHS='%s'", string(graphsJSON)))
	return envVars
}

// ---------------------------------------------------------------------------
// 8. ImageSupportsUV
// ---------------------------------------------------------------------------

// ImageSupportsUV returns true if the base image supports the uv pip installer.
func ImageSupportsUV(baseImage string) bool {
	if baseImage == "langchain/langgraph-trial" {
		return false
	}
	match := semverPattern.FindStringSubmatch(baseImage)
	if match == nil {
		return true
	}
	versionStr := match[1]
	parts := strings.Split(versionStr, ".")
	version := make([]int, len(parts))
	for i, p := range parts {
		fmt.Sscanf(p, "%d", &version[i])
	}

	minUV := []int{0, 2, 47}
	return !versionSliceLessThan(version, minUV)
}

// versionSliceLessThan compares two version slices.
func versionSliceLessThan(a, b []int) bool {
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

// ---------------------------------------------------------------------------
// 9. GetBuildToolsToUninstall
// ---------------------------------------------------------------------------

// GetBuildToolsToUninstall returns the list of build tools that should be
// removed from the final image.
func GetBuildToolsToUninstall(config map[string]any) ([]string, error) {
	kpt := config["keep_pkg_tools"]
	if kpt == nil {
		return []string{"pip", "setuptools", "wheel"}, nil
	}
	// Check for boolean false (falsy)
	if b, ok := kpt.(bool); ok {
		if b {
			return nil, nil
		}
		return []string{"pip", "setuptools", "wheel"}, nil
	}

	// Check for list
	if arr, ok := kpt.([]any); ok {
		keepSet := map[string]bool{}
		for _, item := range arr {
			tool, ok := item.(string)
			if !ok {
				return nil, fmt.Errorf(
					"Invalid build tool to uninstall: %v. Expected one of %v",
					item, buildTools)
			}
			valid := false
			for _, bt := range buildTools {
				if tool == bt {
					valid = true
					break
				}
			}
			if !valid {
				return nil, fmt.Errorf(
					"Invalid build tool to uninstall: %s. Expected one of %v",
					tool, buildTools)
			}
			keepSet[tool] = true
		}
		var result []string
		for _, bt := range buildTools {
			if !keepSet[bt] {
				result = append(result, bt)
			}
		}
		sort.Strings(result)
		return result, nil
	}

	return nil, fmt.Errorf(
		"Invalid value for keep_pkg_tools: %v."+
			" Expected True or a list containing any of %v.",
		kpt, buildTools)
}

// ---------------------------------------------------------------------------
// 10. GetPipCleanupLines
// ---------------------------------------------------------------------------

// GetPipCleanupLines generates the RUN commands for pip cleanup in the Dockerfile.
func GetPipCleanupLines(installCmd string, toUninstall []string, pipInstaller string) string {
	var commands []string
	commands = append(commands, fmt.Sprintf(`# -- Ensure user deps didn't inadvertently overwrite langgraph-api
RUN mkdir -p /api/langgraph_api /api/langgraph_runtime /api/langgraph_license && \
touch /api/langgraph_api/__init__.py /api/langgraph_runtime/__init__.py /api/langgraph_license/__init__.py
RUN PYTHONDONTWRITEBYTECODE=1 %s --no-cache-dir --no-deps -e /api
# -- End of ensuring user deps didn't inadvertently overwrite langgraph-api --
# -- Removing build deps from the final image ~<:===~~~ --`, installCmd))

	if len(toUninstall) > 0 {
		// Validate
		for _, pack := range toUninstall {
			valid := false
			for _, bt := range buildTools {
				if pack == bt {
					valid = true
					break
				}
			}
			if !valid {
				// This matches the Python ValueError
				panic(fmt.Sprintf("Invalid build tool: %s; must be one of %s",
					pack, strings.Join(buildTools, ", ")))
			}
		}

		sorted := make([]string, len(toUninstall))
		copy(sorted, toUninstall)
		sort.Strings(sorted)
		packsStr := strings.Join(sorted, " ")

		commands = append(commands, fmt.Sprintf("RUN pip uninstall -y %s", packsStr))

		// Remove from /usr/local/lib
		var localRm []string
		for _, pack := range toUninstall {
			localRm = append(localRm, fmt.Sprintf("/usr/local/lib/python*/site-packages/%s*", pack))
		}
		localRmStr := strings.Join(localRm, " ")
		hasPip := false
		for _, p := range toUninstall {
			if p == "pip" {
				hasPip = true
				break
			}
		}
		if hasPip {
			localRmStr += ` && find /usr/local/bin -name "pip*" -delete || true`
		}
		commands = append(commands, fmt.Sprintf("RUN rm -rf %s", localRmStr))

		// Remove from /usr/lib (wolfi)
		var wolfiRm []string
		for _, pack := range toUninstall {
			wolfiRm = append(wolfiRm, fmt.Sprintf("/usr/lib/python*/site-packages/%s*", pack))
		}
		wolfiRmStr := strings.Join(wolfiRm, " ")
		if hasPip {
			wolfiRmStr += ` && find /usr/bin -name "pip*" -delete || true`
		}
		commands = append(commands, fmt.Sprintf("RUN rm -rf %s", wolfiRmStr))

		if pipInstaller == "uv" {
			commands = append(commands, fmt.Sprintf(
				"RUN uv pip uninstall --system %s && rm /usr/bin/uv /usr/bin/uvx", packsStr))
		}
	} else {
		if pipInstaller == "uv" {
			commands = append(commands,
				"RUN rm /usr/bin/uv /usr/bin/uvx\n# -- End of build deps removal --")
		}
	}

	return strings.Join(commands, "\n")
}

// ---------------------------------------------------------------------------
// 11. PythonConfigToDocker
// ---------------------------------------------------------------------------

// PythonConfigToDocker generates a Dockerfile and additional build contexts
// for a Python-based LangGraph configuration.
func PythonConfigToDocker(
	configPath string,
	config map[string]any,
	baseImage string,
	apiVersion string,
	escapeVariables bool,
) (string, map[string]string, error) {
	sourceKind := getSourceKind(config)

	buildToolsToUninstall, err := GetBuildToolsToUninstall(config)
	if err != nil {
		return "", nil, err
	}

	if sourceKind == "uv" {
		return PythonConfigToDockerUVLock(configPath, config, baseImage, apiVersion, buildToolsToUninstall)
	}

	pipInstaller, _ := config["pip_installer"].(string)
	if pipInstaller == "" {
		pipInstaller = "auto"
	}
	if pipInstaller == "auto" {
		if ImageSupportsUV(baseImage) {
			pipInstaller = "uv"
		} else {
			pipInstaller = "pip"
		}
	}

	var installCmd string
	switch pipInstaller {
	case "uv":
		installCmd = "uv pip install --system"
	case "pip":
		installCmd = "pip install"
	default:
		return "", nil, fmt.Errorf("Invalid pip_installer: %s", pipInstaller)
	}

	localReqsPipInstall, globalReqsPipInstall, pipConfigFileStr := buildPythonInstallCommands(config, installCmd)

	// Collect PyPI dependencies (non-local)
	deps := configSlice(config, "dependencies")
	var pypiDeps []string
	for _, d := range deps {
		s, ok := d.(string)
		if ok && !strings.HasPrefix(s, ".") {
			pypiDeps = append(pypiDeps, s)
		}
	}

	configPathAbs, _ := filepath.Abs(configPath)
	configDir := filepath.Dir(configPathAbs)

	localDeps, err := AssembleLocalDeps(configPath, config)
	if err != nil {
		return "", nil, err
	}

	if err := UpdateGraphPaths(configPath, config, localDeps); err != nil {
		return "", nil, err
	}
	if err := UpdateConfigPaths(configPath, config, localDeps); err != nil {
		return "", nil, err
	}

	// PyPI install line
	pipPkgsStr := ""
	if len(pypiDeps) > 0 {
		pipPkgsStr = fmt.Sprintf("RUN %s %s", localReqsPipInstall, strings.Join(pypiDeps, " "))
	}

	// Requirements.txt install
	pipReqsStr := ""
	if len(localDeps.PipReqs) > 0 {
		var addLines []string
		for _, req := range localDeps.PipReqs {
			isAdditional := false
			reqParent := filepath.Dir(req.HostPath)
			for _, ac := range localDeps.AdditionalContexts {
				if reqParent == ac {
					isAdditional = true
					break
				}
			}
			if isAdditional {
				addLines = append(addLines,
					fmt.Sprintf("COPY --from=outer-%s requirements.txt %s",
						filepath.Base(req.HostPath), req.ContainerPath))
			} else {
				relPath, _ := filepath.Rel(configDir, req.HostPath)
				addLines = append(addLines,
					fmt.Sprintf("ADD %s %s", filepath.ToSlash(relPath), req.ContainerPath))
			}
		}
		var reqArgs []string
		for _, req := range localDeps.PipReqs {
			reqArgs = append(reqArgs, "-r "+req.ContainerPath)
		}
		pipReqsStr = fmt.Sprintf("# -- Installing local requirements --\n%s\nRUN %s %s\n# -- End of local requirements install --",
			strings.Join(addLines, "\n"),
			localReqsPipInstall,
			strings.Join(reqArgs, " "))
	}

	// Faux packages
	var fauxParts []string
	for fullpath, faux := range localDeps.FauxPkgs {
		baseName := filepath.Base(fullpath)
		isAdditional := false
		for _, ac := range localDeps.AdditionalContexts {
			if fullpath == ac {
				isAdditional = true
				break
			}
		}

		var addLine string
		if isAdditional {
			addLine = fmt.Sprintf("# -- Adding non-package dependency %s --\nCOPY --from=outer-%s . %s",
				baseName, baseName, faux.ContainerPath)
		} else {
			addLine = fmt.Sprintf("# -- Adding non-package dependency %s --\nADD %s %s",
				baseName, faux.RelPath, faux.ContainerPath)
		}

		pyprojectPath := fmt.Sprintf("/deps/outer-%s/pyproject.toml", baseName)
		// Shell-quote: the Python code uses shlex.quote which wraps in single quotes
		quotedPath := shellQuote(pyprojectPath)

		part := fmt.Sprintf(`%s
RUN set -ex && \
    for line in '[project]' \
                'name = "%s"' \
                'version = "0.1"' \
                '[tool.setuptools.package-data]' \
                '"*" = ["**/*"]' \
                '[build-system]' \
                'requires = ["setuptools>=61"]' \
                'build-backend = "setuptools.build_meta"'; do \
        echo "$line" >> %s; \
    done
# -- End of non-package dependency %s --`, addLine, baseName, quotedPath, baseName)
		fauxParts = append(fauxParts, part)
	}
	fauxPkgsStr := strings.Join(fauxParts, "\n\n")

	// Real packages
	var localParts []string
	for fullpath, pkg := range localDeps.RealPkgs {
		isAdditional := false
		for _, ac := range localDeps.AdditionalContexts {
			if fullpath == ac {
				isAdditional = true
				break
			}
		}

		if isAdditional {
			localParts = append(localParts, fmt.Sprintf(
				"# -- Adding local package %s --\nCOPY --from=%s . /deps/%s\n# -- End of local package %s --",
				pkg.RelPath, pkg.ContainerName, pkg.ContainerName, pkg.RelPath))
		} else {
			localParts = append(localParts, fmt.Sprintf(
				"# -- Adding local package %s --\nADD %s /deps/%s\n# -- End of local package %s --",
				pkg.RelPath, pkg.RelPath, pkg.ContainerName, pkg.RelPath))
		}
	}
	localPkgsStr := strings.Join(localParts, "\n")

	// Additional contexts
	additionalContexts := map[string]string{}
	additionalContextNames := map[string]string{} // path -> name
	usedContextNames := map[string]bool{}

	registerAdditionalContext := func(path, preferredName string) string {
		if name, ok := additionalContextNames[path]; ok {
			return name
		}
		name := preferredName
		suffix := 1
		for usedContextNames[name] {
			name = fmt.Sprintf("%s_%d", preferredName, suffix)
			suffix++
		}
		usedContextNames[name] = true
		additionalContextNames[path] = name
		additionalContexts[name] = path
		return name
	}

	for _, p := range localDeps.AdditionalContexts {
		if pkg, ok := localDeps.RealPkgs[p]; ok {
			registerAdditionalContext(p, pkg.ContainerName)
		} else if _, ok := localDeps.FauxPkgs[p]; ok {
			registerAdditionalContext(p, "outer-"+filepath.Base(p))
		} else {
			return "", nil, fmt.Errorf("Unknown additional context: %s", p)
		}
	}

	// Install node string
	nv, _ := config["node_version"].(string)
	installNodeStr := ""
	if (config["ui"] != nil || nv != "") && localDeps.WorkingDir != "" {
		installNodeStr = "RUN /storage/install-node.sh"
	}

	// Combine install steps
	installSteps := []string{installNodeStr, pipConfigFileStr, pipPkgsStr, pipReqsStr, localPkgsStr, fauxPkgsStr}
	var filteredSteps []string
	for _, s := range installSteps {
		if s != "" {
			filteredSteps = append(filteredSteps, s)
		}
	}
	installs := strings.Join(filteredSteps, "\n\n")

	envVars := BuildRuntimeEnvVars(config)

	// JS install
	jsInstStr := ""
	if (config["ui"] != nil || nv != "") && localDeps.WorkingDir != "" {
		nodeVer := nv
		if nodeVer == "" {
			nodeVer = DefaultNodeVersion
		}
		jsInstStr = strings.Join([]string{
			"# -- Installing JS dependencies --",
			fmt.Sprintf("ENV NODE_VERSION=%s", nodeVer),
			fmt.Sprintf("WORKDIR %s", localDeps.WorkingDir),
			fmt.Sprintf("RUN %s && tsx /api/langgraph_api/js/build.mts", GetNodePMInstallCmd(configDir)),
			"# -- End of JS dependencies install --",
		}, "\n")
	}

	imageStr := DockerTag(config, baseImage, apiVersion)

	// Build Dockerfile
	var dockerFileContents []string

	if len(additionalContexts) > 0 {
		dockerFileContents = append(dockerFileContents, "# syntax=docker/dockerfile:1.4", "")
	}

	depVName := "$dep"
	if escapeVariables {
		depVName = "$$dep"
	}

	dockerfileLines := configSlice(config, "dockerfile_lines")
	var dfLines []string
	for _, l := range dockerfileLines {
		if s, ok := l.(string); ok {
			dfLines = append(dfLines, s)
		}
	}

	localDepsInstallStr := fmt.Sprintf(`RUN for dep in /deps/*; do \
            echo "Installing %s"; \
            if [ -d "%s" ]; then \
                echo "Installing %s"; \
                (cd "%s" && %s -e .); \
            fi; \
        done`, depVName, depVName, depVName, depVName, globalReqsPipInstall)

	dockerFileContents = append(dockerFileContents,
		fmt.Sprintf("FROM %s", imageStr),
		"",
		strings.Join(dfLines, "\n"),
		"",
		installs,
		"",
		"# -- Installing all local dependencies --",
		localDepsInstallStr,
		"# -- End of local dependencies install --",
		strings.Join(envVars, "\n"),
		"",
		jsInstStr,
		"",
		GetPipCleanupLines(installCmd, buildToolsToUninstall, pipInstaller),
		"",
	)

	if localDeps.WorkingDir != "" {
		dockerFileContents = append(dockerFileContents, fmt.Sprintf("WORKDIR %s", localDeps.WorkingDir))
	} else {
		dockerFileContents = append(dockerFileContents, "")
	}

	return strings.Join(dockerFileContents, "\n"), additionalContexts, nil
}

// ---------------------------------------------------------------------------
// 12. NodeConfigToDocker
// ---------------------------------------------------------------------------

// NodeConfigToDocker generates a Dockerfile for a Node.js-based LangGraph configuration.
func NodeConfigToDocker(
	configPath string,
	config map[string]any,
	baseImage string,
	apiVersion string,
	installCommand string,
	buildCommand string,
	buildContext string,
) (string, map[string]string, error) {
	configPathAbs, _ := filepath.Abs(configPath)
	configDir := filepath.Dir(configPathAbs)

	var installRoot string
	if buildContext != "" {
		installRoot, _ = filepath.Abs(buildContext)
	} else {
		installRoot = configDir
	}

	installCmd := installCommand
	if installCmd == "" {
		installCmd = GetNodePMInstallCmd(installRoot)
	}

	var fauxPath string
	var containerRoot string
	if buildContext != "" {
		relWorkdir := calculateRelativeWorkdir(configPathAbs, buildContext)
		containerName := filepath.Base(buildContext)
		containerRoot = "/deps/" + containerName
		if relWorkdir != "" {
			fauxPath = containerRoot + "/" + relWorkdir
		} else {
			fauxPath = containerRoot
		}
	} else {
		fauxPath = "/deps/" + filepath.Base(configDir)
	}

	imageStr := DockerTag(config, baseImage, apiVersion)
	envVars := BuildRuntimeEnvVars(config)

	var installWorkdir string
	var installStep string
	var buildStep string

	if buildContext != "" {
		installWorkdir = containerRoot
		installStep = "RUN " + installCmd
		if buildCommand != "" {
			buildStep = "RUN " + buildCommand
		} else {
			buildStep = `RUN (test ! -f /api/langgraph_api/js/build.mts && echo "Prebuild script not found, skipping") || tsx /api/langgraph_api/js/build.mts`
		}
	} else {
		installWorkdir = fauxPath
		installStep = "RUN " + installCmd
		buildStep = `RUN (test ! -f /api/langgraph_api/js/build.mts && echo "Prebuild script not found, skipping") || tsx /api/langgraph_api/js/build.mts`
	}

	buildWorkdir := fauxPath

	addDest := fauxPath
	if buildContext != "" {
		addDest = containerRoot
	}

	dockerfileLines := configSlice(config, "dockerfile_lines")
	var dfLines []string
	for _, l := range dockerfileLines {
		if s, ok := l.(string); ok {
			dfLines = append(dfLines, s)
		}
	}

	dockerFileContents := []string{
		fmt.Sprintf("FROM %s", imageStr),
		"",
		strings.Join(dfLines, "\n"),
		"",
		fmt.Sprintf("ADD . %s", addDest),
		"",
		fmt.Sprintf("WORKDIR %s", installWorkdir),
		"",
		installStep,
		"",
		strings.Join(envVars, "\n"),
		"",
		fmt.Sprintf("WORKDIR %s", buildWorkdir),
		"",
		buildStep,
	}

	return strings.Join(dockerFileContents, "\n"), map[string]string{}, nil
}

// calculateRelativeWorkdir computes the relative path from build context to config dir.
func calculateRelativeWorkdir(configPath, buildContext string) string {
	configDir, _ := filepath.Abs(filepath.Dir(configPath))
	buildContextPath, _ := filepath.Abs(buildContext)

	rel, err := filepath.Rel(buildContextPath, configDir)
	if err != nil || strings.HasPrefix(rel, "..") {
		// The Python version raises ValueError here
		return ""
	}
	if rel == "." {
		return ""
	}
	return filepath.ToSlash(rel)
}

// ---------------------------------------------------------------------------
// 13. ConfigToDocker
// ---------------------------------------------------------------------------

// ConfigToDocker routes to NodeConfigToDocker or PythonConfigToDocker based on config.
func ConfigToDocker(configPath string, config map[string]any, opts DockerOpts) (string, map[string]string, error) {
	baseImage := opts.BaseImage
	if baseImage == "" {
		baseImage = DefaultBaseImage(config, "combined_queue_worker")
	}

	nv, _ := config["node_version"].(string)
	pv, _ := config["python_version"].(string)

	if nv != "" && pv == "" {
		return NodeConfigToDocker(
			configPath, config, baseImage, opts.APIVersion,
			opts.InstallCommand, opts.BuildCommand, opts.BuildContext,
		)
	}

	return PythonConfigToDocker(
		configPath, config, baseImage, opts.APIVersion, opts.EscapeVariables,
	)
}

// ---------------------------------------------------------------------------
// 14. ConfigToCompose
// ---------------------------------------------------------------------------

// ConfigToCompose generates the compose override section.
func ConfigToCompose(configPath string, config map[string]any, opts ComposeOpts) (string, error) {
	baseImage := opts.BaseImage
	if baseImage == "" {
		baseImage = DefaultBaseImage(config, "combined_queue_worker")
	}

	// Build env vars string
	envVarsStr := ""
	if envMap, ok := config["env"].(map[string]any); ok {
		var lines []string
		// Sort keys for deterministic output
		keys := make([]string, 0, len(envMap))
		for k := range envMap {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			lines = append(lines, fmt.Sprintf("            %s: \"%v\"", k, envMap[k]))
		}
		envVarsStr = strings.Join(lines, "\n")
	}

	envFileStr := ""
	if envStr, ok := config["env"].(string); ok {
		envFileStr = "env_file: " + envStr
	}

	watchStr := ""
	if opts.Watch {
		deps := configSlice(config, "dependencies")
		if len(deps) == 0 {
			deps = []any{"."}
		}
		watchPaths := []string{filepath.Base(configPath)}
		for _, d := range deps {
			if s, ok := d.(string); ok && strings.HasPrefix(s, ".") {
				watchPaths = append(watchPaths, s)
			}
		}
		var watchActions []string
		for _, path := range watchPaths {
			watchActions = append(watchActions,
				fmt.Sprintf("                - path: %s\n                  action: rebuild", path))
		}
		watchStr = fmt.Sprintf("\n        develop:\n            watch:\n%s\n",
			strings.Join(watchActions, "\n"))
	}

	if opts.Image != "" {
		return fmt.Sprintf("\n%s\n        %s\n        %s\n",
			indent(envVarsStr, "            "),
			envFileStr,
			watchStr), nil
	}

	// Deep copy config for potential distributed mode
	var configSnapshot map[string]any
	engineRuntimeMode := opts.EngineRuntimeMode
	if engineRuntimeMode == "" {
		engineRuntimeMode = "combined_queue_worker"
	}
	if engineRuntimeMode == "distributed" {
		configSnapshot = deepCopyConfig(config)
	}

	dockerfile, additionalContexts, err := ConfigToDocker(configPath, config, DockerOpts{
		BaseImage:       baseImage,
		APIVersion:      opts.APIVersion,
		EscapeVariables: true,
	})
	if err != nil {
		return "", err
	}

	additionalContextsStr := ""
	if len(additionalContexts) > 0 {
		var lines []string
		// Sort for deterministic output
		keys := make([]string, 0, len(additionalContexts))
		for k := range additionalContexts {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, name := range keys {
			lines = append(lines, fmt.Sprintf("                - %s: %s", name, additionalContexts[name]))
		}
		additionalContextsStr = fmt.Sprintf("\n            additional_contexts:\n%s",
			strings.Join(lines, "\n"))
	}

	result := fmt.Sprintf("\n%s\n        %s\n        pull_policy: build\n        build:\n            context: .%s\n            dockerfile_inline: |\n%s\n        %s\n",
		indent(envVarsStr, "            "),
		envFileStr,
		additionalContextsStr,
		indent(dockerfile, "                "),
		watchStr)

	if engineRuntimeMode == "distributed" {
		executorBaseImage := DefaultBaseImage(configSnapshot, "distributed")
		executorDockerfile, executorAdditionalContexts, err := ConfigToDocker(
			configPath, configSnapshot, DockerOpts{
				BaseImage:       executorBaseImage,
				APIVersion:      opts.APIVersion,
				EscapeVariables: true,
			})
		if err != nil {
			return "", err
		}

		executorAdditionalContextsStr := ""
		if len(executorAdditionalContexts) > 0 {
			var lines []string
			keys := make([]string, 0, len(executorAdditionalContexts))
			for k := range executorAdditionalContexts {
				keys = append(keys, k)
			}
			sort.Strings(keys)
			for _, name := range keys {
				lines = append(lines, fmt.Sprintf("                    - %s: %s", name, executorAdditionalContexts[name]))
			}
			executorAdditionalContextsStr = fmt.Sprintf("\n                additional_contexts:\n%s",
				strings.Join(lines, "\n"))
		}

		postgresURI := "postgres://postgres:postgres@langgraph-postgres:5432/postgres?sslmode=disable"
		result += fmt.Sprintf(`    langgraph-orchestrator:
        image: langchain/langgraph-orchestrator-licensed:latest
        depends_on:
            langgraph-api:
                condition: service_healthy
            langgraph-postgres:
                condition: service_healthy
        environment:
            DATABASE_URI: %s
            EXECUTOR_TARGET: langgraph-executor:8188
        %s
    langgraph-executor:
        depends_on:
            langgraph-postgres:
                condition: service_healthy
            langgraph-api:
                condition: service_healthy
        entrypoint: ["sh", "/storage/executor_entrypoint.sh"]
        environment:
            DATABASE_URI: %s
            REDIS_URI: redis://langgraph-redis:6379
            EXECUTOR_GRPC_PORT: "8188"
            ENGINE_GRPC_ADDRESS: "langgraph-orchestrator:50054"
            LSD_GRPC_SERVER_ADDRESS: "localhost:50050"
            LANGGRAPH_HTTP: ""
        %s
        pull_policy: build
        build:
            context: .%s
            dockerfile_inline: |
%s
`, postgresURI, envFileStr, postgresURI, envFileStr,
			executorAdditionalContextsStr,
			indent(executorDockerfile, "                "))
	}

	return result, nil
}

// ---------------------------------------------------------------------------
// 15. GetNodePMInstallCmd
// ---------------------------------------------------------------------------

// GetNodePMInstallCmd detects the appropriate Node.js package manager install command.
func GetNodePMInstallCmd(projectDir string) string {
	testFile := func(name string) bool {
		info, err := os.Stat(filepath.Join(projectDir, name))
		return err == nil && !info.IsDir()
	}

	yarn := testFile("yarn.lock")
	pnpm := testFile("pnpm-lock.yaml")
	npm := testFile("package-lock.json")
	bun := testFile("bun.lockb")

	if yarn {
		return "yarn install --frozen-lockfile"
	}
	if pnpm {
		return "pnpm i --frozen-lockfile"
	}
	if npm {
		return "npm ci"
	}
	if bun {
		return "bun i"
	}

	// Fallback: check package.json packageManager field
	pkgManagerName := getPkgManagerName(projectDir)
	switch pkgManagerName {
	case "yarn":
		return "yarn install"
	case "pnpm":
		return "pnpm i"
	case "bun":
		return "bun i"
	default:
		return "npm i"
	}
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// buildPythonInstallCommands builds the install command strings with optional pip config.
func buildPythonInstallCommands(config map[string]any, installCmd string) (localReqs, globalReqs, pipConfigFileStr string) {
	base := fmt.Sprintf("PYTHONDONTWRITEBYTECODE=1 %s --no-cache-dir -c /api/constraints.txt", installCmd)
	localReqs = base
	globalReqs = base

	if pcf, _ := config["pip_config_file"].(string); pcf != "" {
		localReqs = "PIP_CONFIG_FILE=/pipconfig.txt " + localReqs
		globalReqs = "PIP_CONFIG_FILE=/pipconfig.txt " + globalReqs
		pipConfigFileStr = fmt.Sprintf("ADD %s /pipconfig.txt", pcf)
	}
	return
}

// configSlice extracts a []any from config[key], returning nil if not present.
func configSlice(config map[string]any, key string) []any {
	if v, ok := config[key].([]any); ok {
		return v
	}
	return nil
}

// shellQuote wraps a string in single quotes, escaping embedded single quotes.
func shellQuote(s string) string {
	// shlex.quote: wrap in single quotes, replace ' with '"'"'
	return "'" + strings.ReplaceAll(s, "'", `'"'"'`) + "'"
}

// indent prepends prefix to each line of text.
func indent(text, prefix string) string {
	if text == "" {
		return text
	}
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		if line != "" {
			lines[i] = prefix + line
		}
	}
	return strings.Join(lines, "\n")
}

// deepCopyConfig does a JSON round-trip deep copy of a config map.
func deepCopyConfig(config map[string]any) map[string]any {
	data, _ := json.Marshal(config)
	var result map[string]any
	_ = json.Unmarshal(data, &result)
	return result
}

// getPkgManagerName reads the packageManager or devEngines.packageManager.name
// field from package.json.
func getPkgManagerName(projectDir string) string {
	data, err := os.ReadFile(filepath.Join(projectDir, "package.json"))
	if err != nil {
		return ""
	}
	var pkg map[string]any
	if err := json.Unmarshal(data, &pkg); err != nil {
		return ""
	}

	// Check packageManager field
	if pm, ok := pkg["packageManager"].(string); ok && pm != "" {
		pm = strings.TrimLeft(pm, "^")
		parts := strings.SplitN(pm, "@", 2)
		return parts[0]
	}

	// Check devEngines.packageManager.name
	if devEngines, ok := pkg["devEngines"].(map[string]any); ok {
		if pmObj, ok := devEngines["packageManager"].(map[string]any); ok {
			if name, ok := pmObj["name"].(string); ok && name != "" {
				return name
			}
		}
	}

	return ""
}
