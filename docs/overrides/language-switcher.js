function applyLanguageSwitching() {
    const selector = document.getElementById("global-language-selector");

    const langBlocks = {
        python: document.querySelectorAll(".lang-python"),
        javascript: document.querySelectorAll(".lang-javascript"),
    };

    const setLanguage = (lang) => {
        for (const [key, blocks] of Object.entries(langBlocks)) {
            blocks.forEach((block) => {
                block.style.display = key === lang ? "block" : "none";
            });
        }
        localStorage.setItem("preferredLang", lang);
    };

    const saved = localStorage.getItem("preferredLang") || "python";

    if (selector) {
        selector.value = saved;
        selector.addEventListener("change", (e) => setLanguage(e.target.value));
    }

    setLanguage(saved);
}

// Run on initial load
document.addEventListener("DOMContentLoaded", applyLanguageSwitching);

// Re-run after client-side navigation (MkDocs Material)
document.addEventListener("pjax:success", applyLanguageSwitching);

// Optional: observe DOM changes (e.g., for late-loaded content)
if (window.MutationObserver) {
    const observer = new MutationObserver(() => applyLanguageSwitching());
    observer.observe(document.body, { childList: true, subtree: true });
}
