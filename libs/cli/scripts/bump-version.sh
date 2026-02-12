#!/usr/bin/env bash
# Bumps the version across Cargo.toml and all npm package.json files.
# Usage: ./scripts/bump-version.sh 0.3.0

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 0.3.0"
    exit 1
fi

VERSION="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI_DIR="$(dirname "$SCRIPT_DIR")"

echo "Bumping version to $VERSION"

# Cargo.toml
sed -i.bak "s/^version = \".*\"/version = \"$VERSION\"/" "$CLI_DIR/Cargo.toml"
rm -f "$CLI_DIR/Cargo.toml.bak"
echo "  Updated Cargo.toml"

# All npm package.json files
for pkg in "$CLI_DIR"/npm/*/package.json; do
    # Update the package's own version
    sed -i.bak "s/\"version\": \".*\"/\"version\": \"$VERSION\"/" "$pkg"
    rm -f "$pkg.bak"

    # Update optionalDependencies versions (main package only)
    if grep -q "optionalDependencies" "$pkg" 2>/dev/null; then
        sed -i.bak "s/\"@langchain\/langgraph-cli-\([^\"]*\)\": \"[^\"]*\"/\"@langchain\/langgraph-cli-\1\": \"$VERSION\"/" "$pkg"
        rm -f "$pkg.bak"
    fi

    echo "  Updated $(basename "$(dirname "$pkg")")/package.json"
done

# Update Cargo.lock
cd "$CLI_DIR"
cargo update -p langgraph-cli 2>/dev/null || true
echo "  Updated Cargo.lock"

echo "Done! Version is now $VERSION"
echo ""
echo "Next steps:"
echo "  1. git add -A && git commit -m 'chore(cli): bump version to $VERSION'"
echo "  2. git tag cli-v$VERSION"
echo "  3. git push origin main --tags"
