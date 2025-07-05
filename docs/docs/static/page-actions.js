// Simple, working page actions script

console.log('Page actions script loaded!');

// Function to convert HTML elements to Markdown
function htmlToMarkdown(element) {
    let markdown = '';
    
    for (const node of element.childNodes) {
        if (node.nodeType === Node.TEXT_NODE) {
            markdown += node.textContent;
        } else if (node.nodeType === Node.ELEMENT_NODE) {
            const tagName = node.tagName.toLowerCase();
            const text = node.textContent.trim();
            
            switch (tagName) {
                case 'h1':
                    markdown += `\n# ${text}\n\n`;
                    break;
                case 'h2':
                    markdown += `\n## ${text}\n\n`;
                    break;
                case 'h3':
                    markdown += `\n### ${text}\n\n`;
                    break;
                case 'h4':
                    markdown += `\n#### ${text}\n\n`;
                    break;
                case 'h5':
                    markdown += `\n##### ${text}\n\n`;
                    break;
                case 'h6':
                    markdown += `\n###### ${text}\n\n`;
                    break;
                case 'p':
                    markdown += `${text}\n\n`;
                    break;
                case 'strong':
                case 'b':
                    markdown += `**${text}**`;
                    break;
                case 'em':
                case 'i':
                    markdown += `*${text}*`;
                    break;
                case 'code':
                    if (node.parentElement && node.parentElement.tagName.toLowerCase() === 'pre') {
                        markdown += text;
                    } else {
                        markdown += `\`${text}\``;
                    }
                    break;
                case 'pre':
                    const codeElement = node.querySelector('code');
                    const codeText = codeElement ? codeElement.textContent : text;
                    markdown += `\n\`\`\`\n${codeText}\n\`\`\`\n\n`;
                    break;
                case 'a':
                    const href = node.getAttribute('href');
                    if (href) {
                        markdown += `[${text}](${href})`;
                    } else {
                        markdown += text;
                    }
                    break;
                case 'ul':
                    markdown += '\n';
                    for (const li of node.querySelectorAll('li')) {
                        markdown += `- ${li.textContent.trim()}\n`;
                    }
                    markdown += '\n';
                    break;
                case 'ol':
                    markdown += '\n';
                    const items = node.querySelectorAll('li');
                    items.forEach((li, index) => {
                        markdown += `${index + 1}. ${li.textContent.trim()}\n`;
                    });
                    markdown += '\n';
                    break;
                case 'blockquote':
                    const lines = text.split('\n');
                    markdown += '\n';
                    lines.forEach(line => {
                        if (line.trim()) {
                            markdown += `> ${line.trim()}\n`;
                        }
                    });
                    markdown += '\n';
                    break;
                case 'br':
                    markdown += '\n';
                    break;
                case 'hr':
                    markdown += '\n---\n\n';
                    break;
                default:
                    if (node.children.length > 0) {
                        markdown += htmlToMarkdown(node);
                    } else {
                        markdown += text;
                    }
            }
        }
    }
    
    return markdown;
}

// Function to extract page content as proper Markdown
function getPageContent() {
    console.log('Getting page content...');
    
    const contentSelectors = [
        'article.md-content__inner',
        '.md-content__inner',
        'main',
        '.md-content',
        'article'
    ];
    
    let content = null;
    for (const selector of contentSelectors) {
        content = document.querySelector(selector);
        console.log(`Trying selector ${selector}:`, content);
        if (content) break;
    }
    
    if (!content) {
        console.error('Could not find page content');
        content = document.body;
    }
    
    const clone = content.cloneNode(true);
    
    const unwantedSelectors = [
        '.page-actions',
        '.md-content__button',
        '.headerlink',
        'script',
        'style',
        '.md-nav',
        '.md-sidebar',
        '.md-header',
        '.md-footer',
        '.md-tabs',
        '.md-search',
        '.notebook-links'
    ];
    
    unwantedSelectors.forEach(selector => {
        clone.querySelectorAll(selector).forEach(el => el.remove());
    });
    
    let markdown = htmlToMarkdown(clone);
    
    markdown = markdown
        .replace(/\n{3,}/g, '\n\n')
        .replace(/^\s+|\s+$/g, '')
        .replace(/\n\s+\n/g, '\n\n')
        .trim();
    
    console.log('Extracted markdown length:', markdown.length);
    console.log('First 200 chars:', markdown.substring(0, 200));
    
    return markdown;
}

// Function to copy text to clipboard
function copyToClipboard(text, successMessage = 'Copied!') {
    console.log('Attempting to copy text:', text.substring(0, 50) + '...');
    
    if (!window.isSecureContext) {
        console.warn('Not in secure context, clipboard API may not work');
    }
    
    if (!navigator.clipboard) {
        console.log('Using fallback copy method');
        try {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            textarea.setSelectionRange(0, 99999);
            const successful = document.execCommand('copy');
            document.body.removeChild(textarea);
            
            if (successful) {
                console.log('Fallback copy successful');
                showFeedback(successMessage);
            } else {
                console.error('Fallback copy failed');
                showFeedback('Copy failed');
            }
        } catch (err) {
            console.error('Fallback copy error:', err);
            showFeedback('Copy failed');
        }
        return;
    }
    
    navigator.clipboard.writeText(text).then(() => {
        console.log('Clipboard copy successful');
        showFeedback(successMessage);
    }).catch(err => {
        console.error('Clipboard copy failed:', err);
        showFeedback('Copy failed');
        
        try {
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            const successful = document.execCommand('copy');
            document.body.removeChild(textarea);
            
            if (successful) {
                console.log('Fallback after clipboard failure successful');
                showFeedback(successMessage);
            }
        } catch (fallbackErr) {
            console.error('Both clipboard and fallback failed:', fallbackErr);
        }
    });
}

// Function to show feedback
function showFeedback(message) {
    console.log('Showing feedback:', message);
    const button = document.querySelector('.copy-text-btn');
    if (!button) {
        console.error('Copy button not found for feedback');
        return;
    }
    
    const originalText = button.textContent;
    button.textContent = message;
    
    setTimeout(() => {
        button.textContent = originalText;
    }, 2000);
}

// Function to copy as plain text
function copyAsText() {
    console.log('Copy as text clicked');
    const content = getPageContent();
    if (content) {
        const plainText = content
            .replace(/^#{1,6}\s+/gm, '')
            .replace(/\*\*(.*?)\*\*/g, '$1')
            .replace(/\*(.*?)\*/g, '$1')
            .replace(/`(.*?)`/g, '$1')
            .replace(/```[\s\S]*?```/g, '')
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
            .replace(/^[-*+]\s+/gm, '')
            .replace(/^\d+\.\s+/gm, '')
            .replace(/^>\s+/gm, '')
            .trim();
        copyToClipboard(plainText, 'Copied!');
    } else {
        console.error('No content to copy');
    }
}

// Function to copy as markdown for LLMs
function copyAsMarkdown() {
    console.log('Copy as markdown clicked');
    const content = getPageContent();
    if (content) {
        const title = document.querySelector('h1')?.textContent || document.title;
        const markdown = `# ${title}\n\n${content}`;
        copyToClipboard(markdown, 'Copied as Markdown!');
    } else {
        console.error('No content to copy as markdown');
    }
}

// Function to view as markdown
function viewAsMarkdown() {
    console.log('View as markdown clicked');
    const content = getPageContent();
    if (content) {
        const title = document.querySelector('h1')?.textContent || document.title;
        const markdown = `# ${title}\n\n${content}`;
        
        const newWindow = window.open('', '_blank');
        if (newWindow) {
            newWindow.document.write(`
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Markdown - ${title}</title>
                    <style>
                        body {
                            font-family: 'Monaco', 'Menlo', 'Consolas', monospace;
                            white-space: pre-wrap;
                            padding: 20px;
                            background: #f8f9fa;
                            color: #333;
                            line-height: 1.5;
                            max-width: 900px;
                            margin: 0 auto;
                        }
                        @media (prefers-color-scheme: dark) {
                            body {
                                background: #1a1a1a;
                                color: #e0e0e0;
                            }
                        }
                    </style>
                </head>
                <body>${markdown}</body>
                </html>
            `);
            newWindow.document.close();
        } else {
            console.error('Failed to open new window - popup blocked?');
        }
    } else {
        console.error('No content to view as markdown');
    }
}

// Function to toggle dropdown
function toggleDropdown() {
    console.log('Toggle dropdown clicked');
    const dropdown = document.querySelector('.dropdown-menu');
    if (dropdown) {
        dropdown.classList.toggle('show');
        console.log('Dropdown toggled, show class:', dropdown.classList.contains('show'));
    } else {
        console.error('Dropdown menu not found');
    }
}

// Function to close dropdown
function closeDropdown() {
    const dropdown = document.querySelector('.dropdown-menu');
    if (dropdown) {
        dropdown.classList.remove('show');
    }
}

// Function to handle all page action clicks using event delegation
function handlePageActionClick(event) {
    const target = event.target.closest('.page-action, .dropdown-item');
    if (!target) return;
    
    console.log('Page action clicked:', target.className);
    
    if (target.classList.contains('copy-text-btn')) {
        event.preventDefault();
        copyAsText();
    } else if (target.classList.contains('dropdown-btn')) {
        event.preventDefault();
        event.stopPropagation();
        toggleDropdown();
    } else if (target.classList.contains('copy-markdown-btn')) {
        event.preventDefault();
        event.stopPropagation();
        copyAsMarkdown();
        closeDropdown();
    } else if (target.classList.contains('view-markdown-btn')) {
        event.preventDefault();
        viewAsMarkdown();
    }
}

// Function to initialize page actions
function initializePageActions() {
    console.log('Initializing page actions...');
    
    document.body.removeEventListener('click', handlePageActionClick);
    document.body.addEventListener('click', handlePageActionClick);
    
    document.body.addEventListener('click', function(event) {
        if (!event.target.closest('.page-actions')) {
            closeDropdown();
        }
    });
    
    console.log('Page actions initialization complete');
}

initializePageActions();

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, setting up MkDocs Material navigation listeners...');
    
    document.addEventListener('DOMContentLoaded', initializePageActions);
    
    let lastUrl = location.href;
    
    const observer = new MutationObserver(function(mutations) {
        if (location.href !== lastUrl) {
            console.log('URL changed from', lastUrl, 'to', location.href);
            lastUrl = location.href;
            setTimeout(initializePageActions, 100);
            return;
        }
        
        let pageActionsChanged = false;
        mutations.forEach(function(mutation) {
            if (mutation.type === 'childList') {
                for (const node of mutation.addedNodes) {
                    if (node.nodeType === Node.ELEMENT_NODE) {
                        if (node.matches && node.matches('.page-actions')) {
                            pageActionsChanged = true;
                            break;
                        }
                        if (node.querySelector && node.querySelector('.page-actions')) {
                            pageActionsChanged = true;
                            break;
                        }
                    }
                }
            }
        });
        
        if (pageActionsChanged) {
            console.log('Page actions detected in DOM, re-initializing...');
            setTimeout(initializePageActions, 50);
        }
    });
    
    observer.observe(document.body, {
        childList: true,
        subtree: true
    });
    
    console.log('MkDocs Material navigation observer set up');
});

window.addEventListener('popstate', function(event) {
    console.log('Popstate event detected, re-initializing page actions...');
    setTimeout(initializePageActions, 100);
});

(function() {
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;
    
    history.pushState = function() {
        originalPushState.apply(history, arguments);
        console.log('PushState detected, re-initializing page actions...');
        setTimeout(initializePageActions, 100);
    };
    
    history.replaceState = function() {
        originalReplaceState.apply(history, arguments);
        console.log('ReplaceState detected, re-initializing page actions...');
        setTimeout(initializePageActions, 100);
    };
})();
