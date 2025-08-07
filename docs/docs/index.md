---
hide_comments: true
title: LangGraph
---

<script>
  // This script only runs in MkDocs, not on GitHub
  var hideGitHubVersion = function() {
    document.querySelectorAll('.github-only').forEach(el => el.style.display = 'none');
  };

  // Handle both initial load and subsequent navigation
  document.addEventListener('DOMContentLoaded', hideGitHubVersion);
  document$.subscribe(hideGitHubVersion);
</script>

<p class="mkdocs-only">
  <img class="logo-light" src="static/wordmark_dark.svg" alt="LangGraph Logo" width="80%">
  <img class="logo-dark" src="static/wordmark_light.svg" alt="LangGraph Logo" width="80%">
</p>

<style>
.md-content h1 {
  display: none;
}
.md-header__topic {
  display: none;
}
</style>

{% include-markdown "../../README.md" %}