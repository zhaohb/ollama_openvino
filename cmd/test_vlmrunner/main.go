package main

import (
	"fmt"
	"os"

	"github.com/ollama/ollama/genai/vlmrunner"
)

func main() {
	if err := vlmrunner.Execute(os.Args); err != nil {
		fmt.Fprintf(os.Stderr, "error: %s\n", err)
		os.Exit(1)
	}
}
