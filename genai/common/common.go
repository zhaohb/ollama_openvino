package common

import (
	"encoding/json"
	"fmt"
	"log"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/alpkeskin/gotoon"
	llamaserver "github.com/ollama/ollama/llm/llama"
)

// input is an element of the prompt to process, either
// a token or an image embedding (generated from a vision projector)
type Input struct {
	Prompt string
}

type VlmInput struct {
	Prompt string
	Images []llamaserver.ImageData
}

type Sequence struct {
	// batch index
	iBatch int

	// prompt inputs left to evaluate
	inputs []Input

	vlminputs []VlmInput

	// tokens that have been generated but not returned yet (e.g. for stop sequences)
	pendingResponses []string

	// channel to send responses over
	responses chan string

	// channel to stop decoding (such as if the remote connection is closed)
	quit chan bool

	// number of tokens to predict
	numPredict int

	// stop sequences
	stop []string

	samplingparameters *SamplingParams

	// number of inputs to keep at the beginning when shifting context window
	numKeep int

	doneReason string

	// Metrics
	startProcessingTime time.Time
	startGenerationTime time.Time
	numDecoded          int
	numPromptInputs     int
}

func FlushPending(seq *Sequence) bool {
	joined := strings.Join(seq.pendingResponses, "")
	seq.pendingResponses = []string{}

	// Check if there are any partial UTF-8 characters remaining.
	// We already check and queue as we are generating but some may
	// still make it here:
	// - Sequence is ending, e.g. generation limit has been hit
	// - Invalid characters in the middle of a string
	// This is a stricter check to ensure we never output invalid Unicode.
	for !utf8.ValidString(joined) {
		joined = joined[:len(joined)-1]
	}

	if len(joined) == 0 {
		return true
	}

	select {
	case seq.responses <- joined:
		return true
	case <-seq.quit:
		return false
	}
}

func NewSequence(inputs []Input, numPromptInputs int, startTime time.Time,
	samplingParams *SamplingParams, numPredict int) *Sequence {
	return &Sequence{
		inputs:              inputs,
		numPromptInputs:     numPromptInputs,
		startProcessingTime: startTime,
		samplingparameters:  samplingParams,
		numPredict:          numPredict,
		pendingResponses:    make([]string, 0),
		responses:           make(chan string, 100),
		quit:                make(chan bool, 1),
	}
}

func VlmNewSequence(inputs []VlmInput, numPromptInputs int, startTime time.Time,
	samplingParams *SamplingParams, numPredict int) *Sequence {
	return &Sequence{
		vlminputs:           inputs,
		numPromptInputs:     numPromptInputs,
		startProcessingTime: startTime,
		samplingparameters:  samplingParams,
		numPredict:          numPredict,
		pendingResponses:    make([]string, 0),
		responses:           make(chan string, 100),
		quit:                make(chan bool, 1),
	}
}

func (s *Sequence) SetDoneReason(reason string) {
	s.doneReason = reason
}

func (s *Sequence) CloseResponses() {
	close(s.responses)
}

func (s *Sequence) SetStartGenerationTime(t time.Time) {
	s.startGenerationTime = t
}

func (s *Sequence) GetInputs() []Input {
	return s.inputs
}

func (s *Sequence) GetVlmInputs() []VlmInput {
	return s.vlminputs
}

func (i *Input) GetPrompt() string {
	return i.Prompt
}

func (i *VlmInput) GetPrompt() string {
	return i.Prompt
}

func (i *VlmInput) GetImages() []llamaserver.ImageData {
	return i.Images
}

func (s *Sequence) GetSamplingParameters() *SamplingParams {
	return s.samplingparameters
}

func (s *Sequence) AppendPendingResponse(response string) {
	s.pendingResponses = append(s.pendingResponses, response)
}

func (s *Sequence) CloseQuit() {
	close(s.quit)
}

func (s *Sequence) GetResponses() <-chan string {
	return s.responses
}

func (s *Sequence) GetDoneReason() string {
	return s.doneReason
}

func (s *Sequence) GetpendingResponses() []string {
	return s.pendingResponses
}

func (s *Sequence) GetNumPromptInputs() int {
	return s.numPromptInputs
}

func (s *Sequence) GetStartGenerationTime() time.Time {
	return s.startGenerationTime
}

func (s *Sequence) GetStartProcessingTime() time.Time {
	return s.startProcessingTime
}

func (s *Sequence) GetNumDecoded() int {
	return s.numDecoded
}

// JSONMatch represents a detected JSON structure with its position
type JSONMatch struct {
	Content string // The JSON content
	Start   int    // Start position in the text
	End     int    // End position in the text (inclusive)
}

// ExtractJSON extracts all JSON structures from text and returns them with positions
func ExtractJSON(text string) []JSONMatch {
	// Quick check: if text doesn't contain { or [, it can't have JSON
	if !strings.Contains(text, "{") && !strings.Contains(text, "[") {
		return nil
	}

	// Find and validate JSON structures
	return findAndValidateJSON(text)
}

// ConvertJSONToTOON converts JSON matches to TOON format and replaces them in the prompt
func ConvertJSONToTOON(prompt string, jsonMatches []JSONMatch) string {
	if len(jsonMatches) == 0 {
		return prompt
	}

	// Process matches in reverse order to maintain correct indices
	result := prompt
	for i := len(jsonMatches) - 1; i >= 0; i-- {
		match := jsonMatches[i]

		// Parse JSON
		var jsonData interface{}
		if err := json.Unmarshal([]byte(match.Content), &jsonData); err != nil {
			log.Printf("Failed to parse JSON at [%d:%d]: %v", match.Start, match.End, err)
			continue
		}

		// Convert to TOON
		toonStr, err := gotoon.Encode(jsonData)
		if err != nil {
			log.Printf("Failed to encode JSON to TOON at [%d:%d]: %v", match.Start, match.End, err)
			continue
		}

		// Wrap TOON in code block with format description
		toonBlock := fmt.Sprintf("Data is TOON format (2-space indent, arrays show length and fields)\n```toon\n%s\n```", toonStr)

		// Replace JSON with TOON block in the prompt
		result = result[:match.Start] + toonBlock + result[match.End+1:]

		log.Printf("Converted JSON[%d] at [%d:%d] to TOON format", i, match.Start, match.End)
	}

	return result
}

// findAndValidateJSON finds all JSON structures in text and validates them
func findAndValidateJSON(text string) []JSONMatch {
	var matches []JSONMatch

	// Find all potential JSON matches by looking for { or [
	for i := 0; i < len(text); i++ {
		if text[i] == '{' {
			end := findMatchingBrace(text, i, '{', '}')
			if end > i {
				jsonStr := text[i : end+1]
				if isValidJSON(jsonStr) {
					matches = append(matches, JSONMatch{
						Content: jsonStr,
						Start:   i,
						End:     end,
					})
					// Skip past this JSON to avoid overlapping matches
					i = end
					continue
				}
			}
		} else if text[i] == '[' {
			end := findMatchingBrace(text, i, '[', ']')
			if end > i {
				jsonStr := text[i : end+1]
				if isValidJSON(jsonStr) {
					matches = append(matches, JSONMatch{
						Content: jsonStr,
						Start:   i,
						End:     end,
					})
					// Skip past this JSON to avoid overlapping matches
					i = end
					continue
				}
			}
		}
	}
	return matches
}

// findMatchingBrace finds the matching closing brace/bracket
func findMatchingBrace(text string, start int, open, close byte) int {
	if start >= len(text) || text[start] != open {
		return -1
	}

	depth := 1
	inString := false
	escape := false

	for i := start + 1; i < len(text); i++ {
		char := text[i]

		if escape {
			escape = false
			continue
		}

		if char == '\\' {
			escape = true
			continue
		}

		if char == '"' {
			inString = !inString
			continue
		}

		if inString {
			continue
		}

		if char == open {
			depth++
		} else if char == close {
			depth--
			if depth == 0 {
				return i
			}
		}
	}

	return -1
}

// isValidJSON checks if a string is valid JSON
func isValidJSON(s string) bool {
	var js interface{}
	return json.Unmarshal([]byte(s), &js) == nil
}
