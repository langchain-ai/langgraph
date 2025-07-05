// Simple page actions - no bloat!

/**
 * Escapes special HTML characters to prevent XSS.
 * @param {string} str - The string to escape.
 * @returns {string} - The escaped string.
 */
function escapeHTML(str) {
    return str.replace(/[&<>"']/g, function (char) {
        switch (char) {
            case '&': return '&amp;';
            case '<': return '&lt;';
            case '>': return '&gt;';
            case '"': return '&quot;';
            case "'": return '&#39;';
            default: return char;
        }
    });
}

function copyPageText() {
    const content = document.querySelector('.md-content__inner') || document.querySelector('main');
    if (content && navigator.clipboard) {
        navigator.clipboard.writeText(content.textContent.trim());
        showFeedback('Copied!');
    }
}

function copyAsMarkdown() {
    const content = document.querySelector('.md-content__inner') || document.querySelector('main');
    if (content && navigator.clipboard) {
        const title = document.querySelector('h1')?.textContent || document.title;
        const markdown = `# ${title}\n\n${content.textContent.trim()}`;
        navigator.clipboard.writeText(markdown);
        showFeedback('Copied as Markdown!');
    }
}

function viewAsMarkdown() {
    const content = document.querySelector('.md-content__inner') || document.querySelector('main');
    if (content) {
        const title = document.querySelector('h1')?.textContent || document.title;
        const markdown = `# ${title}\n\n${content.textContent.trim()}`;
        const newWindow = window.open('', '_blank');
        if (newWindow) {
            newWindow.document.write(`
                <html>
                <head><title>Markdown - ${escapeHTML(title)}</title></head>
                <body style="font-family: monospace; white-space: pre-wrap; padding: 20px;">
                ${escapeHTML(markdown)}
                </body>
                </html>
            `);
        }
    }
}

function showFeedback(message) {
    const button = document.querySelector('.copy-text-btn');
    if (button) {
        const originalText = button.textContent;
        button.textContent = message;
        setTimeout(() => button.textContent = originalText, 2000);
    }
}

function toggleDropdown() {
    const dropdown = document.querySelector('.dropdown-menu');
    if (dropdown) {
        dropdown.classList.toggle('show');
    }
}

function closeDropdown() {
    const dropdown = document.querySelector('.dropdown-menu');
    if (dropdown) {
        dropdown.classList.remove('show');
    }
}

// Simple event handling
document.addEventListener('click', function(e) {
    const target = e.target.closest('.page-action, .dropdown-item');
    if (!target) {
        closeDropdown();
        return;
    }
    
    if (target.classList.contains('copy-text-btn')) {
        copyPageText();
    } else if (target.classList.contains('dropdown-btn')) {
        e.stopPropagation();
        toggleDropdown();
    } else if (target.classList.contains('copy-markdown-btn')) {
        copyAsMarkdown();
        closeDropdown();
    } else if (target.classList.contains('view-markdown-btn')) {
        viewAsMarkdown();
        closeDropdown();
    } else if (target.classList.contains('llm-txt-link')) {
        closeDropdown();
    }
}); 