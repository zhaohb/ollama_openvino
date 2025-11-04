package main

import (
	"fmt"
	"os"

	// 替换为你项目的 module 路径
	"github.com/ollama/ollama/genai/runner"
)

func main() {
	args := append([]string{os.Args[0], "--genai-vlm-engine"}, os.Args[1:]...)

	if err := runner.Execute(args); err != nil {
		fmt.Fprintf(os.Stderr, "error: %s\n", err)
		os.Exit(1)
	}
}
