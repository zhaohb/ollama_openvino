package genai

import (
	"github.com/ollama/ollama/genai/runner"
	"github.com/ollama/ollama/genai/vlmrunner"
)

func Execute(args []string) error {
	// if args[0] == "runner" {
	// 	args = args[1:]
	// }
	args = args[1:]

	var vlmRunner bool
	if args[0] == "--genai-vlm-engine" {
		args = args[1:]
		vlmRunner = true
	}

	if vlmRunner {
		return vlmrunner.Execute(args)
	} else {
		return runner.Execute(args)
	}
}
