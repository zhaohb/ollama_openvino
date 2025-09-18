package main

import (
	"fmt"
	"os"

	// 替换为你项目的 module 路径
	"github.com/ollama/ollama/genai/vlmrunner"
)

func main() {
	if err := vlmrunner.Execute(os.Args); err != nil {
		fmt.Fprintf(os.Stderr, "error: %s\n", err)
		os.Exit(1)
	}
}
