// Package templates provides template definitions and project scaffolding
// for the LangGraph CLI `new` command.
package templates

import (
	"archive/zip"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Template describes a project template with language-specific download URLs.
type Template struct {
	Name        string
	Description string
	Languages   map[string]string // lang -> download URL
}

// Templates is the ordered list of available project templates.
var Templates = []Template{
	{
		Name:        "Deep Agent",
		Description: "An opinionated deployment template for a Deep Agent.",
		Languages: map[string]string{
			"python": "https://github.com/langchain-ai/deep-agent-template/archive/refs/heads/main.zip",
			"js":     "https://github.com/langchain-ai/deep-agent-template-js/archive/refs/heads/main.zip",
		},
	},
	{
		Name:        "Agent",
		Description: "A simple agent that can be flexibly extended to many tools.",
		Languages: map[string]string{
			"python": "https://github.com/langchain-ai/simple-agent-template/archive/refs/heads/main.zip",
		},
	},
	{
		Name:        "New LangGraph Project",
		Description: "A simple, minimal chatbot with memory.",
		Languages: map[string]string{
			"python": "https://github.com/langchain-ai/new-langgraph-project/archive/refs/heads/main.zip",
			"js":     "https://github.com/langchain-ai/new-langgraphjs-project/archive/refs/heads/main.zip",
		},
	},
}

// templateIDEntry maps a template ID to its download URL, template name, and language.
type templateIDEntry struct {
	URL      string
	Name     string
	Language string
}

// templateIDMap is built once at init time from the Templates slice.
var templateIDMap map[string]templateIDEntry

func init() {
	templateIDMap = make(map[string]templateIDEntry)
	for _, t := range Templates {
		for lang, url := range t.Languages {
			if lang != "python" && lang != "js" {
				continue
			}
			id := toTemplateID(t.Name, lang)
			templateIDMap[id] = templateIDEntry{
				URL:      url,
				Name:     t.Name,
				Language: lang,
			}
		}
	}
}

// toTemplateID converts a template name and language into a slug like "deep-agent-python".
func toTemplateID(name, lang string) string {
	return strings.ToLower(strings.ReplaceAll(name, " ", "-")) + "-" + lang
}

// ListTemplateIDs returns a sorted list of all available template IDs.
func ListTemplateIDs() []string {
	ids := make([]string, 0, len(templateIDMap))
	for id := range templateIDMap {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}

// TemplateHelp returns a formatted help string listing available templates.
func TemplateHelp() string {
	var b strings.Builder
	b.WriteString("The name of the template to use. Available options:\n")
	for _, id := range ListTemplateIDs() {
		entry := templateIDMap[id]
		// Find the description from the Templates slice.
		var desc string
		for _, t := range Templates {
			if t.Name == entry.Name {
				desc = t.Description
				break
			}
		}
		fmt.Fprintf(&b, "  %s: %s\n", id, desc)
	}
	return b.String()
}

// CreateNew creates a new LangGraph project at path using the given templateID.
//
// If templateID is empty an error listing available templates is returned (the
// Go CLI is non-interactive, so we cannot prompt).  If path is empty an error
// is returned.
func CreateNew(path, templateID string) error {
	if path == "" {
		return fmt.Errorf("path is required: specify the directory for the new project")
	}

	// Resolve to absolute path.
	absPath, err := filepath.Abs(path)
	if err != nil {
		return fmt.Errorf("cannot resolve path: %w", err)
	}
	path = absPath

	// Check if path exists and is not empty.
	entries, err := os.ReadDir(path)
	if err == nil && len(entries) > 0 {
		return fmt.Errorf(
			"the specified directory already exists and is not empty: %s. "+
				"Aborting to prevent overwriting files", path)
	}

	if templateID == "" {
		return fmt.Errorf(
			"template is required. Use one of the following template IDs:\n%s",
			TemplateHelp())
	}

	entry, ok := templateIDMap[templateID]
	if !ok {
		var sb strings.Builder
		sb.WriteString(fmt.Sprintf("template %q not found.\n", templateID))
		sb.WriteString("Please select from the available options:\n")
		for _, id := range ListTemplateIDs() {
			e := templateIDMap[id]
			var desc string
			for _, t := range Templates {
				if t.Name == e.Name {
					desc = t.Description
					break
				}
			}
			fmt.Fprintf(&sb, "  - %s: %s\n", id, desc)
		}
		return fmt.Errorf("%s", sb.String())
	}

	if err := DownloadAndExtract(entry.URL, path); err != nil {
		return fmt.Errorf("failed to download template: %w", err)
	}

	return nil
}

// DownloadAndExtract downloads a ZIP archive from url and extracts it to
// destPath, stripping the top-level wrapper directory that GitHub includes
// in repository archives.
func DownloadAndExtract(url, destPath string) error {
	// Ensure destination directory exists.
	if err := os.MkdirAll(destPath, 0o755); err != nil {
		return fmt.Errorf("cannot create destination directory: %w", err)
	}

	// Download to a temporary file.
	tmpFile, err := os.CreateTemp("", "langgraph-template-*.zip")
	if err != nil {
		return fmt.Errorf("cannot create temp file: %w", err)
	}
	tmpPath := tmpFile.Name()
	defer os.Remove(tmpPath)

	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		tmpFile.Close()
		return fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		tmpFile.Close()
		return fmt.Errorf("HTTP %d: failed to download %s", resp.StatusCode, url)
	}

	if _, err := io.Copy(tmpFile, resp.Body); err != nil {
		tmpFile.Close()
		return fmt.Errorf("failed to write ZIP data: %w", err)
	}
	tmpFile.Close()

	// Open the ZIP archive.
	zr, err := zip.OpenReader(tmpPath)
	if err != nil {
		return fmt.Errorf("failed to open ZIP archive: %w", err)
	}
	defer zr.Close()

	for _, f := range zr.File {
		// Strip the first path component (GitHub's wrapper directory).
		parts := strings.SplitN(f.Name, "/", 2)
		if len(parts) < 2 || parts[1] == "" {
			continue // skip the wrapper directory entry itself
		}
		relPath := parts[1]

		outPath := filepath.Join(destPath, relPath)

		// Ensure the output path is within destPath (zip-slip protection).
		if !strings.HasPrefix(filepath.Clean(outPath), filepath.Clean(destPath)+string(os.PathSeparator)) {
			continue
		}

		if f.FileInfo().IsDir() {
			if err := os.MkdirAll(outPath, f.Mode()); err != nil {
				return fmt.Errorf("cannot create directory %s: %w", outPath, err)
			}
			continue
		}

		// Create parent directories.
		if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
			return fmt.Errorf("cannot create parent directory: %w", err)
		}

		outFile, err := os.OpenFile(outPath, os.O_CREATE|os.O_WRONLY|os.O_TRUNC, f.Mode())
		if err != nil {
			return fmt.Errorf("cannot create file %s: %w", outPath, err)
		}

		rc, err := f.Open()
		if err != nil {
			outFile.Close()
			return fmt.Errorf("cannot read ZIP entry %s: %w", f.Name, err)
		}

		if _, err := io.Copy(outFile, rc); err != nil {
			rc.Close()
			outFile.Close()
			return fmt.Errorf("failed writing %s: %w", outPath, err)
		}
		rc.Close()
		outFile.Close()
	}

	return nil
}
