package common

import (
	"encoding/json"
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
	Content    string      // The JSON content
	ParsedData interface{} // Parsed JSON data (cached to avoid re-parsing)
	Start      int         // Start position in the text
	End        int         // End position in the text (inclusive)
}

// ExtractJSON extracts all JSON structures from text and returns them with positions
// Optimized: parses JSON during extraction to avoid re-parsing later
func ExtractJSON(text string) []JSONMatch {
	// Quick check: if text doesn't contain { or [, it can't have JSON
	// Use byte-level check for better performance
	hasJSON := false
	for i := 0; i < len(text); i++ {
		if text[i] == '{' || text[i] == '[' {
			hasJSON = true
			break
		}
	}
	if !hasJSON {
		return nil
	}

	// Find and validate JSON structures (with parsing)
	return findAndValidateJSON(text)
}

// ConvertJSONToTOON converts JSON matches to TOON format and replaces them in the prompt
// Optimized: uses pre-parsed JSON data and strings.Builder for efficient string building
func ConvertJSONToTOON(prompt string, jsonMatches []JSONMatch) string {
	if len(jsonMatches) == 0 {
		return prompt
	}

	// Use strings.Builder for efficient string concatenation
	var builder strings.Builder
	// Pre-allocate capacity: original prompt + sum of JSON sizes + TOON overhead
	// TOON format typically adds ~100 bytes per JSON (header + formatting)
	estimatedSize := len(prompt)
	for _, match := range jsonMatches {
		estimatedSize += len(match.Content) + 100 // JSON size + TOON overhead
	}
	builder.Grow(estimatedSize)

	// Process matches in forward order, building result efficiently
	lastPos := 0
	for _, match := range jsonMatches {
		// Add text before this match
		builder.WriteString(prompt[lastPos:match.Start])

		// Convert to TOON using cached parsed data
		if match.ParsedData == nil {
			// Fallback: parse if not cached (shouldn't happen in optimized path)
			var jsonData interface{}
			if err := json.Unmarshal([]byte(match.Content), &jsonData); err != nil {
				log.Printf("Failed to parse JSON at [%d:%d]: %v", match.Start, match.End, err)
				// Keep original JSON if conversion fails
				builder.WriteString(match.Content)
				lastPos = match.End + 1
				continue
			}
			match.ParsedData = jsonData
		}

		// Convert to TOON
		toonStr, err := gotoon.Encode(match.ParsedData)
		if err != nil {
			log.Printf("Failed to encode JSON to TOON at [%d:%d]: %v", match.Start, match.End, err)
			// Keep original JSON if conversion fails
			builder.WriteString(match.Content)
			lastPos = match.End + 1
			continue
		}

		// Wrap TOON in code block with format description
		builder.WriteString("Data is TOON format (2-space indent, arrays show length and fields)\n```toon\n")
		builder.WriteString(toonStr)
		builder.WriteString("\n```")

		lastPos = match.End + 1
	}

	// Add remaining text after last match
	if lastPos < len(prompt) {
		builder.WriteString(prompt[lastPos:])
	}

	return builder.String()
}

// findAndValidateJSON finds all JSON structures in text and validates them
// Optimized: parses JSON during validation to cache parsed data
func findAndValidateJSON(text string) []JSONMatch {
	// Pre-allocate matches slice with estimated capacity (typically 1-5 JSON per prompt)
	matches := make([]JSONMatch, 0, 4)
	textBytes := []byte(text) // Convert to []byte once for better performance

	// Find all potential JSON matches by looking for { or [
	for i := 0; i < len(textBytes); i++ {
		var end int
		var jsonStr string

		// Handle both { and [ in unified way
		if textBytes[i] == '{' {
			end = findMatchingBraceBytes(textBytes, i, '{', '}')
			if end > i {
				jsonStr = text[i : end+1]
			}
		} else if textBytes[i] == '[' {
			end = findMatchingBraceBytes(textBytes, i, '[', ']')
			if end > i {
				jsonStr = text[i : end+1]
			}
		} else {
			continue
		}

		// Parse and validate in one step
		if jsonStr != "" {
			var jsonData interface{}
			if err := json.Unmarshal([]byte(jsonStr), &jsonData); err == nil {
				matches = append(matches, JSONMatch{
					Content:    jsonStr,
					ParsedData: jsonData,
					Start:      i,
					End:        end,
				})
				// Skip past this JSON to avoid overlapping matches
				i = end
			}
		}
	}
	return matches
}

// findMatchingBraceBytes finds the matching closing brace/bracket using []byte
// Optimized: uses []byte to avoid string boundary checks and improve performance
func findMatchingBraceBytes(text []byte, start int, open, close byte) int {
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

// findMatchingBrace finds the matching closing brace/bracket (legacy, kept for compatibility)
func findMatchingBrace(text string, start int, open, close byte) int {
	return findMatchingBraceBytes([]byte(text), start, open, close)
}
