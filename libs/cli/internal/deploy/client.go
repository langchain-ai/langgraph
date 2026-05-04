// Package deploy provides an HTTP client for the LangGraph host backend
// deployment service.
package deploy

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Secret represents a name/value pair sent as a deployment secret.
type Secret struct {
	Name  string `json:"name"`
	Value string `json:"value"`
}

// HostBackendClient is a minimal JSON HTTP client for the host backend
// deployment service.
type HostBackendClient struct {
	BaseURL  string
	APIKey   string
	TenantID string
	client   *http.Client
}

// retryTransport wraps an http.RoundTripper and retries failed requests.
type retryTransport struct {
	base    http.RoundTripper
	retries int
}

func (t *retryTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	var resp *http.Response
	var err error

	// We need to buffer the body so we can replay it on retries.
	var bodyBytes []byte
	if req.Body != nil {
		bodyBytes, err = io.ReadAll(req.Body)
		if err != nil {
			return nil, err
		}
		req.Body.Close()
	}

	for attempt := 0; attempt <= t.retries; attempt++ {
		if bodyBytes != nil {
			req.Body = io.NopCloser(bytes.NewReader(bodyBytes))
		}
		resp, err = t.base.RoundTrip(req)
		if err == nil {
			return resp, nil
		}
		// Only retry on transport-level errors; do not retry on HTTP error
		// status codes (the caller handles those).
	}
	return resp, err
}

// NewClient creates a new HostBackendClient. The baseURL is stripped of any
// trailing slash. The underlying http.Client uses a 30-second timeout and
// retries transport-level failures up to 3 times.
func NewClient(baseURL, apiKey string) *HostBackendClient {
	return &HostBackendClient{
		BaseURL:  strings.TrimRight(baseURL, "/"),
		APIKey:   apiKey,
		TenantID: "",
		client: &http.Client{
			Timeout: 30 * time.Second,
			Transport: &retryTransport{
				base:    http.DefaultTransport,
				retries: 3,
			},
		},
	}
}

// request executes an HTTP request against the host backend and returns the
// parsed JSON response. It attaches required headers and handles errors.
func (c *HostBackendClient) request(method, path string, payload map[string]any, params map[string]string) (map[string]any, error) {
	fullURL := c.BaseURL + path

	// Append query parameters.
	if len(params) > 0 {
		q := url.Values{}
		for k, v := range params {
			q.Set(k, v)
		}
		fullURL += "?" + q.Encode()
	}

	var body io.Reader
	if payload != nil {
		data, err := json.Marshal(payload)
		if err != nil {
			return nil, fmt.Errorf("marshalling request payload: %w", err)
		}
		body = bytes.NewReader(data)
	}

	req, err := http.NewRequest(method, fullURL, body)
	if err != nil {
		return nil, fmt.Errorf("creating request: %w", err)
	}

	req.Header.Set("X-Api-Key", c.APIKey)
	req.Header.Set("Accept", "application/json")
	if payload != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.TenantID != "" {
		req.Header.Set("X-Tenant-ID", c.TenantID)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("%s %s: %w", method, path, err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("reading response from %s: %w", path, err)
	}

	if resp.StatusCode >= 400 {
		detail := string(respBody)
		if detail == "" {
			detail = fmt.Sprintf("%d", resp.StatusCode)
		}
		return nil, fmt.Errorf("%s %s failed with status %d: %s", method, path, resp.StatusCode, detail)
	}

	if len(respBody) == 0 {
		return nil, nil
	}

	var result map[string]any
	if err := json.Unmarshal(respBody, &result); err != nil {
		return nil, fmt.Errorf("failed to decode response from %s: %w", path, err)
	}
	return result, nil
}

// requestNoBody is a convenience wrapper for requests that return no parsed body.
func (c *HostBackendClient) requestNoBody(method, path string, payload map[string]any, params map[string]string) error {
	_, err := c.request(method, path, payload, params)
	return err
}

// CreateDeployment creates a new deployment.
func (c *HostBackendClient) CreateDeployment(name, deploymentType, source string, configPath string, secrets []Secret) (map[string]any, error) {
	sourceRevisionConfig := map[string]any{}
	if source == "internal_source" && configPath != "" {
		sourceRevisionConfig["langgraph_config_path"] = configPath
	}

	payload := map[string]any{
		"name":                   name,
		"source":                 source,
		"source_config":          map[string]any{"deployment_type": deploymentType},
		"source_revision_config": sourceRevisionConfig,
	}
	if secrets != nil {
		payload["secrets"] = secrets
	}
	return c.request("POST", "/v2/deployments", payload, nil)
}

// ListDeployments lists deployments, optionally filtering by name.
func (c *HostBackendClient) ListDeployments(nameContains string) (map[string]any, error) {
	params := map[string]string{"name_contains": nameContains}
	return c.request("GET", "/v2/deployments", nil, params)
}

// GetDeployment retrieves a single deployment by ID.
func (c *HostBackendClient) GetDeployment(deploymentID string) (map[string]any, error) {
	return c.request("GET", fmt.Sprintf("/v2/deployments/%s", deploymentID), nil, nil)
}

// DeleteDeployment deletes a deployment by ID.
func (c *HostBackendClient) DeleteDeployment(deploymentID string) error {
	return c.requestNoBody("DELETE", fmt.Sprintf("/v2/deployments/%s", deploymentID), nil, nil)
}

// RequestPushToken requests a push token for a deployment.
func (c *HostBackendClient) RequestPushToken(deploymentID string) (map[string]any, error) {
	return c.request("POST", fmt.Sprintf("/v2/deployments/%s/push-token", deploymentID), nil, nil)
}

// RequestUploadURL gets a signed URL for uploading the source tarball.
func (c *HostBackendClient) RequestUploadURL(deploymentID string) (map[string]any, error) {
	return c.request("POST", fmt.Sprintf("/v2/deployments/%s/upload-url", deploymentID), nil, nil)
}

// UpdateDeployment triggers a new revision using a pre-pushed Docker image.
func (c *HostBackendClient) UpdateDeployment(deploymentID, imageURI string, secrets []Secret) (map[string]any, error) {
	payload := map[string]any{
		"revision_source":        "internal_docker",
		"source_revision_config": map[string]any{"image_uri": imageURI},
	}
	if secrets != nil {
		payload["secrets"] = secrets
	}
	return c.request("PATCH", fmt.Sprintf("/v2/deployments/%s", deploymentID), payload, nil)
}

// UpdateDeploymentInternalSource triggers a remote-build revision using an
// uploaded source tarball.
func (c *HostBackendClient) UpdateDeploymentInternalSource(
	deploymentID, sourceTarballPath, configPath string,
	secrets []Secret,
	installCommand, buildCommand string,
) (map[string]any, error) {
	payload := map[string]any{
		"revision_source": "internal_source",
		"source_revision_config": map[string]any{
			"source_tarball_path":   sourceTarballPath,
			"langgraph_config_path": configPath,
		},
	}

	sourceConfig := map[string]any{}
	if installCommand != "" {
		sourceConfig["install_command"] = installCommand
	}
	if buildCommand != "" {
		sourceConfig["build_command"] = buildCommand
	}
	if len(sourceConfig) > 0 {
		payload["source_config"] = sourceConfig
	}

	if secrets != nil {
		payload["secrets"] = secrets
	}
	return c.request("PATCH", fmt.Sprintf("/v2/deployments/%s", deploymentID), payload, nil)
}

// ListRevisions lists revisions for a deployment.
func (c *HostBackendClient) ListRevisions(deploymentID string, limit int) (map[string]any, error) {
	return c.request("GET", fmt.Sprintf("/v2/deployments/%s/revisions", deploymentID), nil, map[string]string{
		"limit": fmt.Sprintf("%d", limit),
	})
}

// GetRevision retrieves a single revision.
func (c *HostBackendClient) GetRevision(deploymentID, revisionID string) (map[string]any, error) {
	return c.request("GET", fmt.Sprintf("/v2/deployments/%s/revisions/%s", deploymentID, revisionID), nil, nil)
}

// GetBuildLogs retrieves build logs for a revision.
func (c *HostBackendClient) GetBuildLogs(projectID, revisionID string, payload map[string]any) (map[string]any, error) {
	return c.request("POST", fmt.Sprintf("/v1/projects/%s/revisions/%s/build_logs", projectID, revisionID), payload, nil)
}

// GetDeployLogs retrieves deploy logs. If revisionID is non-empty, it is
// included in the path to scope the logs.
func (c *HostBackendClient) GetDeployLogs(projectID string, payload map[string]any, revisionID string) (map[string]any, error) {
	var path string
	if revisionID != "" {
		path = fmt.Sprintf("/v1/projects/%s/revisions/%s/deploy_logs", projectID, revisionID)
	} else {
		path = fmt.Sprintf("/v1/projects/%s/deploy_logs", projectID)
	}
	return c.request("POST", path, payload, nil)
}
