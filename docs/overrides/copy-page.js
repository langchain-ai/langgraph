// Simple copy page functionality - just copy the markdown content
function copyPageAsMarkdown() {
    const markdownScript = document.getElementById('page-markdown-content');
    if (!markdownScript) {
        alert('Markdown content not available for this page');
        return;
    }
    
    try {
        const data = JSON.parse(markdownScript.textContent);
        const content = `# ${data.title}\n\nSource: ${window.location.href}\n\n${data.markdown}`;
        
        navigator.clipboard.writeText(content).then(() => {
            // Simple notification
            const notification = document.createElement('div');
            notification.textContent = 'Page content copied to clipboard';
            notification.style.cssText = 'position:fixed;top:20px;right:20px;background:#4CAF50;color:white;padding:10px;border-radius:4px;z-index:9999;';
            document.body.appendChild(notification);
            setTimeout(() => notification.remove(), 3000);
        }).catch(() => {
            alert('Failed to copy content');
        });
    } catch (e) {
        alert('Failed to parse page content');
    }
}

// Add button to header - simpler approach
document.addEventListener('DOMContentLoaded', function() {
    const headerSource = document.querySelector('.md-header__source');
    if (headerSource) {
        const button = document.createElement('button');
        button.textContent = 'Copy page';
        button.onclick = copyPageAsMarkdown;
        button.style.cssText = 'background:none;border:1px solid #ddd;padding:6px 12px;margin-right:8px;border-radius:4px;cursor:pointer;';
        headerSource.parentNode.insertBefore(button, headerSource);
    }
});