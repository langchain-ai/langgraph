package main

import (
	"os"

	"github.com/langchain-ai/langgraph/libs/cli/internal/root"
)

func main() {
	os.Exit(root.Run(os.Args[1:], os.Stdout, os.Stderr))
}
