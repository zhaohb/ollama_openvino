package common

/*
#cgo CFLAGS: -std=c11
#cgo CXXFLAGS: -std=c++17

#cgo LDFLAGS: -lopenvino_genai
#cgo LDFLAGS: -lopenvino
#cgo LDFLAGS: -lopenvino_c
#cgo LDFLAGS: -lopenvino_genai_c

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "openvino/genai/c/llm_pipeline.h"
#include "openvino/genai/c/vlm_pipeline.h"
#include "openvino/genai/c/visibility.h"
#include "openvino/genai/c/chat_history.h"
#include "openvino/genai/c/json_container.h"
#include "openvino/genai/c/lora_adapter.h"

#include "openvino/c/openvino.h"
#include "openvino/c/ov_common.h"
#include <stdbool.h>

typedef int (*callback_function)(const char*, void*);

extern int goCallbackBridge(char* input, void* ptr);

// Compatibility macros using numeric values (same in both versions)
#define OV_GENAI_STREAMING_STATUS_RUNNING_COMPAT 0 // Continue to run inference
#define OV_GENAI_STREAMING_STATUS_STOP_COMPAT 1 // Stop generation, keep history
#define OV_GENAI_STREAMING_STATUS_CANCEL_COMPAT 2 // Stop and drop last prompt

static ov_status_e ov_genai_llm_pipeline_create_npu_output_2048(const char* models_path,
																  const char* device,
                                                                  ov_genai_llm_pipeline** pipe) {
	return 	ov_genai_llm_pipeline_create(models_path, "NPU", 4, pipe, "MAX_PROMPT_LEN", "2048", "MIN_RESPONSE_LEN", "256");
}

static ov_status_e ov_genai_llm_pipeline_create_cgo(const char* models_path,
																  const char* device,
                                                                  ov_genai_llm_pipeline** pipe) {
	return 	ov_genai_llm_pipeline_create(models_path, device, 0, pipe);
}

static ov_status_e ov_genai_vlm_pipeline_create_cgo(const char* models_path,
																  const char* device,
                                                                  ov_genai_vlm_pipeline** pipe) {
	return 	ov_genai_vlm_pipeline_create(models_path, device, 0, pipe);
}

// ─── LoRA wrappers ──────────────────────────────────────────────────────────
// Dynamic LoRA support. Design (matches OpenVINO GenAI's hard constraint that
// runtime switching only works among adapters registered at construction):
//   1. At load time, register ALL adapters in MODE_DYNAMIC via *_create_dynamic.
//      The adapter handles are kept alive on the Go side (NOT freed) because the
//      controller identifies adapters by object identity at apply() time.
//   2. Per request, build a transient AdapterConfig holding the chosen adapter
//      (or none) and set it on the generation_config via *_genconfig_set_one /
//      _clear. generate() then applies it without reloading the model.
// Everything here is additive: the non-LoRA create_cgo paths are untouched and
// only used when no adapter is configured.

// Build an LLM pipeline in MODE_DYNAMIC with the given adapters all registered.
// On success, out_adapters[i] receives the handle for lora_paths[i]; the caller
// owns these handles and must keep them alive for the pipeline's lifetime, then
// free them with ov_genai_adapter_free.
static ov_status_e ov_genai_llm_pipeline_create_dynamic_cgo(const char* models_path,
                                                            const char* device,
                                                            const char** lora_paths,
                                                            float* alphas,
                                                            size_t n,
                                                            ov_genai_adapter** out_adapters,
                                                            ov_genai_llm_pipeline** pipe) {
	ov_genai_adapter_config* cfg = NULL;
	ov_status_e st = ov_genai_adapter_config_create(OV_GENAI_ADAPTER_MODE_DYNAMIC, &cfg);
	if (st != OK) {
		return st;
	}
	size_t i = 0;
	for (i = 0; i < n; i++) {
		out_adapters[i] = NULL;
		st = ov_genai_adapter_create(lora_paths[i], &out_adapters[i]);
		if (st != OK) {
			goto fail;
		}
		st = ov_genai_adapter_config_add(cfg, out_adapters[i], alphas[i]);
		if (st != OK) {
			goto fail;
		}
	}
	st = ov_genai_llm_pipeline_create_with_adapters(models_path, device, cfg, 0, pipe);
	if (st != OK) {
		goto fail;
	}
	ov_genai_adapter_config_free(cfg);
	return OK;
fail:
	for (size_t j = 0; j < n; j++) {
		if (out_adapters[j]) { ov_genai_adapter_free(out_adapters[j]); out_adapters[j] = NULL; }
	}
	ov_genai_adapter_config_free(cfg);
	return st;
}

static ov_status_e ov_genai_vlm_pipeline_create_dynamic_cgo(const char* models_path,
                                                            const char* device,
                                                            const char** lora_paths,
                                                            float* alphas,
                                                            size_t n,
                                                            ov_genai_adapter** out_adapters,
                                                            ov_genai_vlm_pipeline** pipe) {
	ov_genai_adapter_config* cfg = NULL;
	ov_status_e st = ov_genai_adapter_config_create(OV_GENAI_ADAPTER_MODE_DYNAMIC, &cfg);
	if (st != OK) {
		return st;
	}
	size_t i = 0;
	for (i = 0; i < n; i++) {
		out_adapters[i] = NULL;
		st = ov_genai_adapter_create(lora_paths[i], &out_adapters[i]);
		if (st != OK) {
			goto fail;
		}
		st = ov_genai_adapter_config_add(cfg, out_adapters[i], alphas[i]);
		if (st != OK) {
			goto fail;
		}
	}
	st = ov_genai_vlm_pipeline_create_with_adapters(models_path, device, cfg, 0, pipe);
	if (st != OK) {
		goto fail;
	}
	ov_genai_adapter_config_free(cfg);
	return OK;
fail:
	for (size_t j = 0; j < n; j++) {
		if (out_adapters[j]) { ov_genai_adapter_free(out_adapters[j]); out_adapters[j] = NULL; }
	}
	ov_genai_adapter_config_free(cfg);
	return st;
}

// Per-request multi-adapter selection: stack n (already-registered) adapters,
// each at its own alpha, onto a generation_config. n==0 selects "no adapter"
// (empty dynamic config), cleanly disabling LoRA for this call. GenAI sums the
// stacked LoRA deltas; note adapters carrying constant tensors cannot be stacked
// (GenAI restriction) — those must be used one at a time.
static ov_status_e ov_genai_genconfig_set_adapters_multi(ov_genai_generation_config* gc,
                                                         ov_genai_adapter** adapters,
                                                         float* alphas,
                                                         size_t n) {
	ov_genai_adapter_config* cfg = NULL;
	ov_status_e st = ov_genai_adapter_config_create(OV_GENAI_ADAPTER_MODE_DYNAMIC, &cfg);
	if (st != OK) {
		return st;
	}
	for (size_t i = 0; i < n; i++) {
		if (!adapters[i]) {
			continue;
		}
		st = ov_genai_adapter_config_add(cfg, adapters[i], alphas[i]);
		if (st != OK) {
			ov_genai_adapter_config_free(cfg);
			return st;
		}
	}
	st = ov_genai_generation_config_set_adapters(gc, cfg);
	ov_genai_adapter_config_free(cfg);
	return st;
}

*/
import "C"

import (
	"archive/tar"
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"image"
	_ "image/gif"  // 支持 GIF
	_ "image/jpeg" // 支持 JPEG
	_ "image/png"  // 支持 PNG
	"io"
	"log"
	"os"
	"path/filepath"
	"runtime/cgo"
	"strconv"
	"unsafe"

	"github.com/ollama/ollama/api"
	llamaserver "github.com/ollama/ollama/llm/llama"
)

// NOTE on thread-safety: the OpenVINO GenAI LLM/VLM pipelines are NOT
// thread-safe — each holds its own inference and KV-cache state and allows only
// one in-flight generate. These Generate* functions therefore must be called
// serially. That is guaranteed by the runners: every generate happens inside
// processBatch, which runs in a single run() goroutine and holds Server.mu for
// the whole call. Do NOT call generate from any other goroutine/path without
// adding serialization here.

type SamplingParams struct {
	TopK           int
	TopP           float32
	Temp           float32
	MaxNewToken    int
	RepeatPenalty  float32
	StopString     []string
	StopIds        []string
	RepeatLastN    int
	EnableThinking bool

	// LoRA (optional, per-request). Lora is the registry of adapters registered
	// at load time; nil = no LoRA involved at all (legacy behavior).
	//
	// Two ways to select adapters for this generation (LoraSpecs wins if set):
	//   - LoraSpecs: stack one OR many adapters, each with its own alpha.
	//   - LoraName/LoraAlpha: convenience for the single-adapter case.
	// An empty selection (Lora non-nil but no name/specs) explicitly disables
	// LoRA for this call.
	Lora      *LoraRegistry
	LoraName  string
	LoraAlpha float32
	LoraSpecs []LoraSpec

	// LoraDisable forces "no adapter" for this generation even when adapters
	// are registered at load time. Without it, an empty selection leaves the
	// load-time adapter(s) active (the session-fixed `ollama run --lora` case).
	LoraDisable bool
}

// LoraSpec selects one registered adapter by name at a blending weight for a
// single generation. Multiple specs stack (their LoRA deltas sum).
type LoraSpec struct {
	Name  string
	Alpha float32
}

// LoraRegistry holds the LoRA adapters registered with a pipeline at load time
// (MODE_DYNAMIC). The adapter handles must outlive the pipeline, so they are
// kept here and only released by Free(). Switching among them per request does
// NOT reload the model.
type LoraRegistry struct {
	adapters map[string]*C.ov_genai_adapter // name → handle
}

// Names returns the registered adapter names (for logging / validation).
func (r *LoraRegistry) Names() []string {
	if r == nil {
		return nil
	}
	names := make([]string, 0, len(r.adapters))
	for n := range r.adapters {
		names = append(names, n)
	}
	return names
}

// Has reports whether an adapter with the given name is registered.
func (r *LoraRegistry) Has(name string) bool {
	if r == nil {
		return false
	}
	_, ok := r.adapters[name]
	return ok
}

// Free releases all adapter handles. Call after the pipeline is freed.
func (r *LoraRegistry) Free() {
	if r == nil {
		return
	}
	for n, a := range r.adapters {
		if a != nil {
			C.ov_genai_adapter_free(a)
		}
		delete(r.adapters, n)
	}
}

// type Model struct {
// 	pipe *C.LLMPipelineHandle
// }

type Model *C.ov_genai_llm_pipeline
type VlmModel *C.ov_genai_vlm_pipeline

func IsGGUF(filePath string) (bool, error) {
	file, err := os.Open(filePath)
	if err != nil {
		return false, fmt.Errorf("failed to open file: %v", err)
	}
	defer file.Close()

	// Read the first 4 bytes (magic number for GGUF)
	reader := bufio.NewReader(file)
	magicBytes := make([]byte, 4)
	_, err = reader.Read(magicBytes)
	if err != nil {
		return false, fmt.Errorf("failed to read magic number: %v", err)
	}

	// Compare the magic number (GGUF in ASCII)
	expectedMagic := []byte{0x47, 0x47, 0x55, 0x46} // "GGUF" in hex
	for i := 0; i < 4; i++ {
		if magicBytes[i] != expectedMagic[i] {
			return false, nil
		}
	}

	return true, nil
}

func IsGzipByMagicBytes(filepath string) (bool, error) {
	file, err := os.Open(filepath)
	if err != nil {
		return false, err
	}
	defer file.Close()

	magicBytes := make([]byte, 2)
	_, err = file.Read(magicBytes)
	if err != nil {
		return false, err
	}

	return bytes.Equal(magicBytes, []byte{0x1F, 0x8B}), nil
}

func CopyFile(src, dst string) error {
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	_, err = io.Copy(dstFile, srcFile)
	if err != nil {
		return err
	}

	err = dstFile.Sync()
	if err != nil {
		return err
	}

	return nil
}

func UnpackTarGz(tarGzPath string, destDir string) error {
	file, err := os.Open(tarGzPath)
	if err != nil {
		return err
	}
	defer file.Close()

	gzr, err := gzip.NewReader(file)
	if err != nil {
		return err
	}
	defer gzr.Close()

	tr := tar.NewReader(gzr)

	for {
		header, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return err
		}

		target := filepath.Join(destDir, header.Name)

		switch header.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(target, os.ModePerm); err != nil {
				return err
			}
		case tar.TypeReg:
			outFile, err := os.Create(target)
			if err != nil {
				return err
			}
			defer outFile.Close()

			if _, err := io.Copy(outFile, tr); err != nil {
				return err
			}
		default:
			return fmt.Errorf("unsupported type: %c in %s", header.Typeflag, header.Name)
		}
	}
	return nil
}

func CreatePipeline(modelsPath string, device string) Model {
	cModelsPath := C.CString(modelsPath)
	cDevice := C.CString(device)

	var pipeline *C.ov_genai_llm_pipeline

	defer C.free(unsafe.Pointer(cModelsPath))
	defer C.free(unsafe.Pointer(cDevice))

	// C.ov_genai_llm_pipeline_create(cModelsPath, cDevice, &pipeline)
	if device == "NPU" {
		C.ov_genai_llm_pipeline_create_npu_output_2048(cModelsPath, cDevice, &pipeline)
	} else {
		C.ov_genai_llm_pipeline_create_cgo(cModelsPath, cDevice, &pipeline)
	}
	return pipeline
}

func CreateVlmPipeline(modelsPath string, device string) VlmModel {
	cModelsPath := C.CString(modelsPath)
	cDevice := C.CString(device)

	var pipeline *C.ov_genai_vlm_pipeline

	defer C.free(unsafe.Pointer(cModelsPath))
	defer C.free(unsafe.Pointer(cDevice))

	C.ov_genai_vlm_pipeline_create_cgo(cModelsPath, cDevice, &pipeline)
	return pipeline
}

// loraName derives a short, stable adapter name from its file path (the base
// filename without extension), used as the per-request selection key.
func loraName(path string) string {
	base := filepath.Base(path)
	if ext := filepath.Ext(base); ext != "" {
		base = base[:len(base)-len(ext)]
	}
	return base
}

// buildLoraCArrays converts adapter paths into the C arrays expected by the
// dynamic-create wrappers. Returns the C string array, alpha array, and a
// cleanup func to free the C strings. All adapters default to alpha=1.0.
func buildLoraCArrays(loraPaths []string) (**C.char, *C.float, func()) {
	n := len(loraPaths)
	cPaths := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(uintptr(0))))
	cAlphas := C.malloc(C.size_t(n) * C.size_t(unsafe.Sizeof(C.float(0))))
	pathSlice := (*[1 << 16]*C.char)(cPaths)[:n:n]
	alphaSlice := (*[1 << 16]C.float)(cAlphas)[:n:n]
	for i, p := range loraPaths {
		pathSlice[i] = C.CString(p)
		alphaSlice[i] = C.float(1.0)
	}
	cleanup := func() {
		for i := 0; i < n; i++ {
			C.free(unsafe.Pointer(pathSlice[i]))
		}
		C.free(cPaths)
		C.free(cAlphas)
	}
	return (**C.char)(cPaths), (*C.float)(cAlphas), cleanup
}

// CreatePipelineWithLora builds an LLM pipeline in MODE_DYNAMIC with ALL given
// LoRA adapters registered, so generation can switch among them per request
// (see SamplingParams.LoraName). It is the LoRA-aware counterpart of
// CreatePipeline; callers fall back to CreatePipeline when loraPaths is empty so
// the existing path is completely unaffected. Returns the pipeline plus a
// LoraRegistry that owns the adapter handles (keep it alive for the pipeline's
// lifetime; Free() it after the pipeline is freed). NPU note: the NPU
// constructor doesn't take adapters, so LoRA there falls back to a plain load.
func CreatePipelineWithLora(modelsPath string, device string, loraPaths []string) (Model, *LoraRegistry) {
	if len(loraPaths) == 0 || device == "NPU" {
		if len(loraPaths) > 0 {
			log.Printf("LoRA requested but unsupported on device %s; loading without adapters", device)
		}
		return CreatePipeline(modelsPath, device), nil
	}

	cModelsPath := C.CString(modelsPath)
	cDevice := C.CString(device)
	defer C.free(unsafe.Pointer(cModelsPath))
	defer C.free(unsafe.Pointer(cDevice))

	n := len(loraPaths)
	cPaths, cAlphas, cleanup := buildLoraCArrays(loraPaths)
	defer cleanup()

	outAdapters := make([]*C.ov_genai_adapter, n)
	var pipeline *C.ov_genai_llm_pipeline
	st := C.ov_genai_llm_pipeline_create_dynamic_cgo(
		cModelsPath, cDevice, cPaths, cAlphas, C.size_t(n),
		(**C.ov_genai_adapter)(unsafe.Pointer(&outAdapters[0])), &pipeline)
	if st != C.OK {
		log.Printf("failed to create LLM pipeline with %d LoRA adapter(s) (status %d); falling back to plain load", n, int(st))
		return CreatePipeline(modelsPath, device), nil
	}

	reg := &LoraRegistry{adapters: make(map[string]*C.ov_genai_adapter, n)}
	for i, p := range loraPaths {
		reg.adapters[loraName(p)] = outAdapters[i]
	}
	log.Printf("LLM pipeline loaded with %d LoRA adapter(s) (MODE_DYNAMIC): %v", n, reg.Names())
	return pipeline, reg
}

// CreateVlmPipelineWithLora is the VLM counterpart of CreatePipelineWithLora.
// Adapters apply to the language model; the vision encoder is unchanged.
func CreateVlmPipelineWithLora(modelsPath string, device string, loraPaths []string) (VlmModel, *LoraRegistry) {
	if len(loraPaths) == 0 || device == "NPU" {
		if len(loraPaths) > 0 {
			log.Printf("LoRA requested but unsupported on device %s; loading VLM without adapters", device)
		}
		return CreateVlmPipeline(modelsPath, device), nil
	}

	cModelsPath := C.CString(modelsPath)
	cDevice := C.CString(device)
	defer C.free(unsafe.Pointer(cModelsPath))
	defer C.free(unsafe.Pointer(cDevice))

	n := len(loraPaths)
	cPaths, cAlphas, cleanup := buildLoraCArrays(loraPaths)
	defer cleanup()

	outAdapters := make([]*C.ov_genai_adapter, n)
	var pipeline *C.ov_genai_vlm_pipeline
	st := C.ov_genai_vlm_pipeline_create_dynamic_cgo(
		cModelsPath, cDevice, cPaths, cAlphas, C.size_t(n),
		(**C.ov_genai_adapter)(unsafe.Pointer(&outAdapters[0])), &pipeline)
	if st != C.OK {
		log.Printf("failed to create VLM pipeline with %d LoRA adapter(s) (status %d); falling back to plain load", n, int(st))
		return CreateVlmPipeline(modelsPath, device), nil
	}

	reg := &LoraRegistry{adapters: make(map[string]*C.ov_genai_adapter, n)}
	for i, p := range loraPaths {
		reg.adapters[loraName(p)] = outAdapters[i]
	}
	log.Printf("VLM pipeline loaded with %d LoRA adapter(s) (MODE_DYNAMIC): %v", n, reg.Names())
	return pipeline, reg
}

// applyLoraToConfig sets the per-request adapter selection on a generation
// config, based on SamplingParams. No-op when no LoRA registry is present.
// Returns nothing; failures are logged but never abort generation.
func applyLoraToConfig(cConfig *C.ov_genai_generation_config, p *SamplingParams) {
	if p == nil || p.Lora == nil {
		return // no LoRA involved — legacy behavior, config untouched
	}

	// Normalize selection into a list of (name, alpha). LoraSpecs takes
	// precedence; otherwise fall back to the single LoraName/LoraAlpha.
	specs := p.LoraSpecs
	if len(specs) == 0 && p.LoraName != "" {
		specs = []LoraSpec{{Name: p.LoraName, Alpha: p.LoraAlpha}}
	}

	// No per-request selection and no explicit disable: leave the config
	// untouched so the adapter(s) registered at load time stay active. This is
	// the session-fixed case (`ollama run --lora ...`) — the load-time adapter
	// should apply to every generation without the request having to re-name it.
	if len(specs) == 0 && !p.LoraDisable {
		return
	}

	// Resolve names → handles, dropping (with a warning) any unregistered name.
	// An empty resolved set + LoraDisable means "no adapter this call".
	handles := make([]*C.ov_genai_adapter, 0, len(specs))
	alphas := make([]C.float, 0, len(specs))
	for _, s := range specs {
		a, ok := p.Lora.adapters[s.Name]
		if !ok {
			log.Printf("LoRA %q not registered (have %v); skipping it", s.Name, p.Lora.Names())
			continue
		}
		alpha := s.Alpha
		if alpha == 0 {
			alpha = 1.0
		}
		handles = append(handles, a)
		alphas = append(alphas, C.float(alpha))
	}

	if len(handles) == 0 {
		// Explicitly select "no adapter" for this generation (disable LoRA).
		if st := C.ov_genai_genconfig_set_adapters_multi(cConfig, nil, nil, 0); st != C.OK {
			log.Printf("failed to clear LoRA on generation config (status %d)", int(st))
		}
		return
	}

	if st := C.ov_genai_genconfig_set_adapters_multi(
		cConfig,
		(**C.ov_genai_adapter)(unsafe.Pointer(&handles[0])),
		(*C.float)(unsafe.Pointer(&alphas[0])),
		C.size_t(len(handles)),
	); st != C.OK {
		log.Printf("failed to set %d LoRA adapter(s) on generation config (status %d)", len(handles), int(st))
	}
}

func PrintGenaiMetrics(metrics *C.ov_genai_perf_metrics) {

	log.Printf("Genai Metrics info:")

	var loadtime C.float
	C.ov_genai_perf_metrics_get_load_time(metrics, &loadtime)
	log.Printf("Load time: %.2f", loadtime)

	var gen_mean C.float
	var gen_std C.float
	C.ov_genai_perf_metrics_get_inference_duration(metrics, &gen_mean, &gen_std)
	log.Printf("Generate time: %.2f ± %.2f ms\n", gen_mean, gen_std)

	var token_mean C.float
	var token_std C.float
	C.ov_genai_perf_metrics_get_tokenization_duration(metrics, &token_mean, &token_std)
	log.Printf("Tokenization time: %.2f ± %.2f ms\n", token_mean, token_std)

	var detoken_mean C.float
	var detoken_std C.float
	C.ov_genai_perf_metrics_get_detokenization_duration(metrics, &detoken_mean, &detoken_std)
	log.Printf("Detokenization time: %.2f ± %.2f ms\n", detoken_mean, detoken_std)

	var ttft_mean C.float
	var ttft_std C.float
	C.ov_genai_perf_metrics_get_ttft(metrics, &ttft_mean, &ttft_std)
	log.Printf("TTFT: %.2f ± %.2f ms\n", ttft_mean, ttft_std)

	var tpot_mean C.float
	var tpot_std C.float
	C.ov_genai_perf_metrics_get_tpot(metrics, &tpot_mean, &tpot_std)
	log.Printf("TPOT: %.2f ± %.2f ms/token\n", tpot_mean, tpot_std)

	var num_generation_tokens C.size_t
	C.ov_genai_perf_metrics_get_num_generation_tokens(metrics, &num_generation_tokens)
	log.Printf("Num of generation tokens: %d\n", num_generation_tokens)

	var num_input_tokens C.size_t
	C.ov_genai_perf_metrics_get_num_input_tokens(metrics, &num_input_tokens)
	log.Printf("Num of input tokens: %d\n", num_input_tokens)

	var tput_mean C.float
	var tput_std C.float
	C.ov_genai_perf_metrics_get_throughput(metrics, &tput_mean, &tput_std)
	log.Printf("Throughput: %.2f ± %.2f tokens/s\n", tput_mean, tput_std)
}

// applyPerfMetricsToSequence copies OpenVINO-reported token counts onto the sequence
// so the genairunner HTTP layer can populate PromptN / PredictedN for Ollama usage APIs.
func applyPerfMetricsToSequence(metrics *C.ov_genai_perf_metrics, seq *Sequence) {
	if metrics == nil || seq == nil {
		return
	}
	var numGen C.size_t
	var numIn C.size_t
	C.ov_genai_perf_metrics_get_num_generation_tokens(metrics, &numGen)
	C.ov_genai_perf_metrics_get_num_input_tokens(metrics, &numIn)
	seq.SetOpenVINOTokenCounts(int(numIn), int(numGen))
	seq.SetOVPerfApplied(true)
}

func SetSamplingParams(samplingparameters *SamplingParams) *C.ov_genai_generation_config {
	var cConfig *C.ov_genai_generation_config
	C.ov_genai_generation_config_create(&cConfig)

	log.Printf("Sampling Parameters - Temperature: %.2f, TopP: %.2f, TopK: %d, RepeatPenalty: %.2f",
		samplingparameters.Temp,
		samplingparameters.TopP,
		samplingparameters.TopK,
		samplingparameters.RepeatPenalty)

	C.ov_genai_generation_config_set_max_new_tokens(cConfig, C.size_t(samplingparameters.MaxNewToken))
	C.ov_genai_generation_config_set_temperature(cConfig, C.float(samplingparameters.Temp))
	C.ov_genai_generation_config_set_top_p(cConfig, C.float(samplingparameters.TopP))
	// C.ov_genai_generation_config_set_top_k(cConfig, C.size_t(samplingparameters.TopK))

	if samplingparameters.StopIds != nil {
		cStopArray := C.malloc(C.size_t(len(samplingparameters.StopIds)) * C.size_t(unsafe.Sizeof(C.int64_t(0))))
		defer C.free(cStopArray)

		var stopArray []int64

		for i := 0; i < int(len(samplingparameters.StopIds)); i++ {
			stop_id, _ := strconv.ParseInt(samplingparameters.StopIds[i], 10, 64)
			stopArray = append(stopArray, stop_id)
		}

		for i, v := range stopArray {
			*(*C.int64_t)(unsafe.Pointer(uintptr(cStopArray) + uintptr(i)*unsafe.Sizeof(C.int64_t(0)))) = C.int64_t(v)
		}
		C.ov_genai_generation_config_set_stop_token_ids(cConfig, (*C.int64_t)(cStopArray), C.size_t(len(stopArray)))
	}

	C.ov_genai_generation_config_set_repetition_penalty(cConfig, C.float(samplingparameters.RepeatPenalty))

	if samplingparameters.StopString != nil {
		cStopStrings := C.malloc(C.size_t(len(samplingparameters.StopString)) * C.size_t(unsafe.Sizeof(uintptr(0))))
		defer C.free(cStopStrings)
		for i, s := range samplingparameters.StopString {
			cStr := C.CString(s)               // Convert Go string to C string
			defer C.free(unsafe.Pointer(cStr)) // Free each C string after use
			*(*uintptr)(unsafe.Pointer(uintptr(cStopStrings) + uintptr(i)*unsafe.Sizeof(uintptr(0)))) = uintptr(unsafe.Pointer(cStr))
		}
		C.ov_genai_generation_config_set_stop_strings(cConfig, (**C.char)(cStopStrings), C.size_t(len(samplingparameters.StopString)))
	}

	// Per-request LoRA selection (no-op unless a registry + name is set).
	applyLoraToConfig(cConfig, samplingparameters)

	return cConfig
}

func GenerateTextWithMetrics(pipeline *C.ov_genai_llm_pipeline, input string, samplingparameters *SamplingParams, seq *Sequence) string {
	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_decoded_results
	output_size := C.size_t(0)

	var streamer_callback C.streamer_callback
	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

	handle := cgo.NewHandle(seq)
	defer handle.Delete()
	streamer_callback.args = unsafe.Pointer(uintptr(handle))

	// 构建 chat_history
	var chatHistory *C.ov_genai_chat_history
	chatStatus := C.ov_genai_chat_history_create(&chatHistory)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("failed to create chat history: %d", chatStatus)
		return ""
	}
	defer C.ov_genai_chat_history_free(chatHistory)

	// 将 prompt 构建为 user message JSON 并加入 history
	messageJSON, err := json.Marshal(map[string]string{
		"role":    "user",
		"content": input,
	})
	if err != nil {
		log.Printf("failed to marshal user message: %v", err)
		return ""
	}

	cMessageJSON := C.CString(string(messageJSON))
	defer C.free(unsafe.Pointer(cMessageJSON))

	var messageContainer *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&messageContainer, cMessageJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("failed to create message json container: %d", jsonStatus)
		return ""
	}
	defer C.ov_genai_json_container_free(messageContainer)

	chatStatus = C.ov_genai_chat_history_push_back(chatHistory, messageContainer)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("failed to push message to chat history: %d", chatStatus)
		return ""
	}

	// 如果有 tools，通过 chat_history 设置
	tools := seq.GetTools()
	if len(tools) > 0 {
		toolsJSON, err := json.Marshal(tools)
		if err != nil {
			log.Printf("failed to marshal tools: %v", err)
		} else {
			// log.Printf("setting tools on chat_history: %s", string(toolsJSON))
			log.Printf("setting tools on chat_history: %d tool(s)", len(tools))
			cToolsJSON := C.CString(string(toolsJSON))
			defer C.free(unsafe.Pointer(cToolsJSON))

			var toolsContainer *C.ov_genai_json_container
			jsonStatus = C.ov_genai_json_container_create_from_json_string(&toolsContainer, cToolsJSON)
			if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
				log.Printf("failed to create tools json container: %d", jsonStatus)
			} else {
				defer C.ov_genai_json_container_free(toolsContainer)
				chatStatus = C.ov_genai_chat_history_set_tools(chatHistory, toolsContainer)
				if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
					log.Printf("failed to set tools on chat history: %d", chatStatus)
				} else {
					log.Printf("tools set on chat_history successfully")
				}
			}
		}
	}

	// Set enable_thinking via extra_context on chat_history
	extraContextJSON := fmt.Sprintf(`{"enable_thinking": %t}`, samplingparameters.EnableThinking)
	cExtraContextJSON := C.CString(extraContextJSON)
	defer C.free(unsafe.Pointer(cExtraContextJSON))

	var extraContextContainer *C.ov_genai_json_container
	jsonStatus = C.ov_genai_json_container_create_from_json_string(&extraContextContainer, cExtraContextJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("failed to create extra_context json container: %d", jsonStatus)
	} else {
		defer C.ov_genai_json_container_free(extraContextContainer)
		chatStatus = C.ov_genai_chat_history_set_extra_context(chatHistory, extraContextContainer)
		if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
			log.Printf("failed to set extra_context on chat history: %d", chatStatus)
		} else {
			log.Printf("enable_thinking=%t set on chat_history extra_context", samplingparameters.EnableThinking)
		}
	}

	// 使用 generate_with_history 替代 generate
	C.ov_genai_llm_pipeline_generate_with_history(pipeline,
		chatHistory,
		(*C.ov_genai_generation_config)(cConfig),
		&streamer_callback,
		&result)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)

	var metrics *C.ov_genai_perf_metrics
	C.ov_genai_decoded_results_get_perf_metrics(result, &metrics)

	applyPerfMetricsToSequence(metrics, seq)
	PrintGenaiMetrics(metrics)

	out := C.GoString((*C.char)(cOutput))
	log.Printf("genai decoded output (final cOutput):\n%s", out)
	return out
}

// addMessageToChatHistory marshals a message map to JSON and pushes it into the C chat_history.
func addMessageToChatHistory(chatHistory *C.ov_genai_chat_history, msg map[string]any) error {
	msgJSON, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %v", err)
	}

	cMsgJSON := C.CString(string(msgJSON))
	defer C.free(unsafe.Pointer(cMsgJSON))

	var container *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&container, cMsgJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		return fmt.Errorf("failed to create json container: %d", jsonStatus)
	}
	defer C.ov_genai_json_container_free(container)

	chatStatus := C.ov_genai_chat_history_push_back(chatHistory, container)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		return fmt.Errorf("failed to push message to chat history: %d", chatStatus)
	}
	return nil
}

// GenerateWithChatHistory builds a full chat_history from []api.Message,
// sets tools and extra_context, then calls generate_with_history.
// This follows the OpenVINO GenAI architecture: tool responses are added
// as separate messages with role "tool" into chat_history, and the chat
// template handles all formatting internally.
func GenerateWithChatHistory(pipeline *C.ov_genai_llm_pipeline, messages []api.Message,
	tools []api.Tool, samplingparameters *SamplingParams, seq *Sequence) string {

	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_decoded_results
	output_size := C.size_t(0)

	var streamer_callback C.streamer_callback
	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

	handle := cgo.NewHandle(seq)
	defer handle.Delete()
	streamer_callback.args = unsafe.Pointer(uintptr(handle))

	var chatHistory *C.ov_genai_chat_history
	chatStatus := C.ov_genai_chat_history_create(&chatHistory)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("failed to create chat history: %d", chatStatus)
		return ""
	}
	defer C.ov_genai_chat_history_free(chatHistory)

	for _, msg := range messages {
		msgMap := map[string]any{
			"role":    msg.Role,
			"content": msg.Content,
		}

		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			var toolCallsData []map[string]any
			for _, tc := range msg.ToolCalls {
				tcMap := map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Function.Name,
						"arguments": tc.Function.Arguments.ToMap(),
					},
				}
				toolCallsData = append(toolCallsData, tcMap)
			}
			msgMap["tool_calls"] = toolCallsData
		}

		if msg.Role == "tool" {
			if msg.ToolCallID != "" {
				msgMap["tool_call_id"] = msg.ToolCallID
			}
			if msg.ToolName != "" {
				msgMap["name"] = msg.ToolName
			}
		}

		if err := addMessageToChatHistory(chatHistory, msgMap); err != nil {
			log.Printf("failed to add %s message to chat history: %v", msg.Role, err)
			return ""
		}
	}

	if len(tools) > 0 {
		toolsJSON, err := json.Marshal(tools)
		if err != nil {
			log.Printf("failed to marshal tools: %v", err)
		} else {
			// log.Printf("setting tools on chat_history: %s", string(toolsJSON))
			cToolsJSON := C.CString(string(toolsJSON))
			defer C.free(unsafe.Pointer(cToolsJSON))

			var toolsContainer *C.ov_genai_json_container
			jsonStatus := C.ov_genai_json_container_create_from_json_string(&toolsContainer, cToolsJSON)
			if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
				log.Printf("failed to create tools json container: %d", jsonStatus)
			} else {
				defer C.ov_genai_json_container_free(toolsContainer)
				chatStatus = C.ov_genai_chat_history_set_tools(chatHistory, toolsContainer)
				if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
					log.Printf("failed to set tools on chat history: %d", chatStatus)
				}
			}
		}
	}

	log.Printf("enable_thinking: %t", samplingparameters.EnableThinking)
	extraContextJSON := fmt.Sprintf(`{"enable_thinking": %t}`, samplingparameters.EnableThinking)
	cExtraContextJSON := C.CString(extraContextJSON)
	defer C.free(unsafe.Pointer(cExtraContextJSON))

	var extraContextContainer *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&extraContextContainer, cExtraContextJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("failed to create extra_context json container: %d", jsonStatus)
	} else {
		defer C.ov_genai_json_container_free(extraContextContainer)
		chatStatus = C.ov_genai_chat_history_set_extra_context(chatHistory, extraContextContainer)
		if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
			log.Printf("failed to set extra_context on chat history: %d", chatStatus)
		}
	}

	C.ov_genai_llm_pipeline_generate_with_history(pipeline,
		chatHistory,
		(*C.ov_genai_generation_config)(cConfig),
		&streamer_callback,
		&result)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)

	var metrics *C.ov_genai_perf_metrics
	C.ov_genai_decoded_results_get_perf_metrics(result, &metrics)

	applyPerfMetricsToSequence(metrics, seq)
	PrintGenaiMetrics(metrics)

	out := C.GoString((*C.char)(cOutput))
	log.Printf("genai decoded output (final cOutput):\n%s", out)
	return out
}

type ImageInfo struct {
	Width    int
	Height   int
	Format   string
	Channels int
}

func getImageInfo(imageData []byte) (*ImageInfo, error) {
	reader := bytes.NewReader(imageData)

	config, format, err := image.DecodeConfig(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image config: %v", err)
	}

	channels := 3

	return &ImageInfo{
		Width:    config.Width,
		Height:   config.Height,
		Format:   format,
		Channels: channels,
	}, nil
}

func decodeImageToRGB(imageData []byte) ([]byte, error) {
	reader := bytes.NewReader(imageData)

	img, _, err := image.Decode(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}

	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	rgbData := make([]byte, width*height*3)
	idx := 0

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := img.At(x, y).RGBA()

			rgbData[idx] = byte(r >> 8)   // R
			rgbData[idx+1] = byte(g >> 8) // G
			rgbData[idx+2] = byte(b >> 8) // B
			idx += 3
		}
	}

	return rgbData, nil
}

func createTensorFromImageData(imageData []byte) (*C.ov_tensor_t, error) {
	if len(imageData) == 0 {
		return nil, fmt.Errorf("empty image data")
	}

	imgInfo, err := getImageInfo(imageData)
	if err != nil {
		return nil, fmt.Errorf("failed to get image info: %v", err)
	}

	height := int64(imgInfo.Height)
	width := int64(imgInfo.Width)
	channels := int64(imgInfo.Channels)

	decodedData, err := decodeImageToRGB(imageData)
	if err != nil {
		return nil, fmt.Errorf("failed to decode image: %v", err)
	}

	expectedSize := height * width * channels
	if int64(len(decodedData)) < expectedSize {
		return nil, fmt.Errorf("decoded image data size mismatch: expected %d, got %d", expectedSize, len(decodedData))
	}

	dims := []int64{1, height, width, channels}

	// create ov_shape_t
	var shape C.ov_shape_t
	status := C.ov_shape_create(
		C.int64_t(4),                           // rank
		(*C.int64_t)(unsafe.Pointer(&dims[0])), // dims
		&shape,
	)
	if status != C.OK {
		return nil, fmt.Errorf("failed to create shape: %d", status)
	}
	defer C.ov_shape_free(&shape)

	var tensor *C.ov_tensor_t
	status = C.ov_tensor_create_from_host_ptr(
		C.U8,                            // element type (uint8)
		shape,                           // shape
		unsafe.Pointer(&decodedData[0]), // host pointer to decoded data
		&tensor,
	)
	if status != C.OK {
		return nil, fmt.Errorf("failed to create tensor: %d", status)
	}

	return tensor, nil
}

func vlmImageTensors(images []llamaserver.ImageData) ([]*C.ov_tensor_t, **C.ov_tensor_t) {
	var rgbs []*C.ov_tensor_t
	for i, imageData := range images {
		tensor, err := createTensorFromImageData(imageData.Data)
		if err != nil {
			log.Printf("vlm: error creating tensor for image %d: %v", i, err)
			continue
		}
		rgbs = append(rgbs, tensor)
	}
	var cRgbs **C.ov_tensor_t
	if len(rgbs) > 0 {
		cRgbs = (**C.ov_tensor_t)(unsafe.Pointer(&rgbs[0]))
	}
	return rgbs, cRgbs
}

// VlmGenerateWithChatHistory builds chat_history from messages (same structure as LLM),
// sets tools and enable_thinking on history, then calls ov_genai_vlm_pipeline_generate_with_history.
// rgbs must be images for the last user turn per OpenVINO VLM C API.
func VlmGenerateWithChatHistory(pipeline *C.ov_genai_vlm_pipeline, messages []api.Message,
	images []llamaserver.ImageData, tools []api.Tool, samplingparameters *SamplingParams, seq *Sequence) string {

	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_vlm_decoded_results
	output_size := C.size_t(0)

	var streamer_callback C.streamer_callback
	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

	handle := cgo.NewHandle(seq)
	defer handle.Delete()
	streamer_callback.args = unsafe.Pointer(uintptr(handle))

	var chatHistory *C.ov_genai_chat_history
	chatStatus := C.ov_genai_chat_history_create(&chatHistory)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("vlm: failed to create chat history: %d", chatStatus)
		return ""
	}
	defer C.ov_genai_chat_history_free(chatHistory)

	for _, msg := range messages {
		msgMap := map[string]any{
			"role":    msg.Role,
			"content": msg.Content,
		}

		if msg.Role == "assistant" && len(msg.ToolCalls) > 0 {
			var toolCallsData []map[string]any
			for _, tc := range msg.ToolCalls {
				tcMap := map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Function.Name,
						"arguments": tc.Function.Arguments.ToMap(),
					},
				}
				toolCallsData = append(toolCallsData, tcMap)
			}
			msgMap["tool_calls"] = toolCallsData
		}

		if msg.Role == "tool" {
			if msg.ToolCallID != "" {
				msgMap["tool_call_id"] = msg.ToolCallID
			}
			if msg.ToolName != "" {
				msgMap["name"] = msg.ToolName
			}
		}

		if err := addMessageToChatHistory(chatHistory, msgMap); err != nil {
			log.Printf("vlm: failed to add %s message to chat history: %v", msg.Role, err)
			return ""
		}
	}

	if len(tools) > 0 {
		toolsJSON, err := json.Marshal(tools)
		if err != nil {
			log.Printf("vlm: failed to marshal tools: %v", err)
		} else {
			cToolsJSON := C.CString(string(toolsJSON))
			defer C.free(unsafe.Pointer(cToolsJSON))

			var toolsContainer *C.ov_genai_json_container
			jsonStatus := C.ov_genai_json_container_create_from_json_string(&toolsContainer, cToolsJSON)
			if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
				log.Printf("vlm: failed to create tools json container: %d", jsonStatus)
			} else {
				defer C.ov_genai_json_container_free(toolsContainer)
				chatStatus = C.ov_genai_chat_history_set_tools(chatHistory, toolsContainer)
				if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
					log.Printf("vlm: failed to set tools on chat history: %d", chatStatus)
				}
			}
		}
	}

	log.Printf("vlm enable_thinking: %t", samplingparameters.EnableThinking)
	extraContextJSON := fmt.Sprintf(`{"enable_thinking": %t}`, samplingparameters.EnableThinking)
	cExtraContextJSON := C.CString(extraContextJSON)
	defer C.free(unsafe.Pointer(cExtraContextJSON))

	var extraContextContainer *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&extraContextContainer, cExtraContextJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("vlm: failed to create extra_context json container: %d", jsonStatus)
	} else {
		defer C.ov_genai_json_container_free(extraContextContainer)
		chatStatus = C.ov_genai_chat_history_set_extra_context(chatHistory, extraContextContainer)
		if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
			log.Printf("vlm: failed to set extra_context on chat history: %d", chatStatus)
		}
	}

	rgbs, cRgbs := vlmImageTensors(images)
	log.Printf("vlm generate_with_history: %d messages, %d image tensors", len(messages), len(rgbs))

	C.ov_genai_vlm_pipeline_generate_with_history(pipeline,
		chatHistory,
		cRgbs, C.size_t(len(rgbs)),
		(*C.ov_genai_generation_config)(cConfig),
		&streamer_callback,
		&result)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)

	var metrics *C.ov_genai_perf_metrics
	C.ov_genai_vlm_decoded_results_get_perf_metrics(result, &metrics)

	applyPerfMetricsToSequence(metrics, seq)
	PrintGenaiMetrics(metrics)

	out := C.GoString((*C.char)(cOutput))
	log.Printf("vlm genai decoded output (final cOutput):\n%s", out)
	return out
}

// VlmGenerateTextWithMetrics uses a single user message in chat_history (LLM GenerateTextWithMetrics parity),
// including tools and enable_thinking, then ov_genai_vlm_pipeline_generate_with_history.
func VlmGenerateTextWithMetrics(pipeline *C.ov_genai_vlm_pipeline, input string, images []llamaserver.ImageData, samplingparameters *SamplingParams, seq *Sequence) string {
	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_vlm_decoded_results
	output_size := C.size_t(0)

	var streamer_callback C.streamer_callback
	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

	handle := cgo.NewHandle(seq)
	defer handle.Delete()
	streamer_callback.args = unsafe.Pointer(uintptr(handle))

	var chatHistory *C.ov_genai_chat_history
	chatStatus := C.ov_genai_chat_history_create(&chatHistory)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("vlm: failed to create chat history: %d", chatStatus)
		return ""
	}
	defer C.ov_genai_chat_history_free(chatHistory)

	messageJSON, err := json.Marshal(map[string]string{
		"role":    "user",
		"content": input,
	})
	if err != nil {
		log.Printf("vlm: failed to marshal user message: %v", err)
		return ""
	}

	cMessageJSON := C.CString(string(messageJSON))
	defer C.free(unsafe.Pointer(cMessageJSON))

	var messageContainer *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&messageContainer, cMessageJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("vlm: failed to create message json container: %d", jsonStatus)
		return ""
	}
	defer C.ov_genai_json_container_free(messageContainer)

	chatStatus = C.ov_genai_chat_history_push_back(chatHistory, messageContainer)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("vlm: failed to push message to chat history: %d", chatStatus)
		return ""
	}

	tools := seq.GetTools()
	if len(tools) > 0 {
		toolsJSON, err := json.Marshal(tools)
		if err != nil {
			log.Printf("vlm: failed to marshal tools: %v", err)
		} else {
			log.Printf("vlm: setting tools on chat_history: %d tool(s)", len(tools))
			cToolsJSON := C.CString(string(toolsJSON))
			defer C.free(unsafe.Pointer(cToolsJSON))

			var toolsContainer *C.ov_genai_json_container
			jsonStatus = C.ov_genai_json_container_create_from_json_string(&toolsContainer, cToolsJSON)
			if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
				log.Printf("vlm: failed to create tools json container: %d", jsonStatus)
			} else {
				defer C.ov_genai_json_container_free(toolsContainer)
				chatStatus = C.ov_genai_chat_history_set_tools(chatHistory, toolsContainer)
				if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
					log.Printf("vlm: failed to set tools on chat history: %d", chatStatus)
				}
			}
		}
	}

	extraContextJSON := fmt.Sprintf(`{"enable_thinking": %t}`, samplingparameters.EnableThinking)
	cExtraContextJSON := C.CString(extraContextJSON)
	defer C.free(unsafe.Pointer(cExtraContextJSON))

	var extraContextContainer *C.ov_genai_json_container
	jsonStatus = C.ov_genai_json_container_create_from_json_string(&extraContextContainer, cExtraContextJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("vlm: failed to create extra_context json container: %d", jsonStatus)
	} else {
		defer C.ov_genai_json_container_free(extraContextContainer)
		chatStatus = C.ov_genai_chat_history_set_extra_context(chatHistory, extraContextContainer)
		if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
			log.Printf("vlm: failed to set extra_context on chat history: %d", chatStatus)
		} else {
			log.Printf("vlm: enable_thinking=%t set on chat_history extra_context", samplingparameters.EnableThinking)
		}
	}

	rgbs, cRgbs := vlmImageTensors(images)
	log.Printf("vlm: prompt_len=%d images=%d (successful tensors=%d)", len(input), len(images), len(rgbs))

	C.ov_genai_vlm_pipeline_generate_with_history(pipeline,
		chatHistory,
		cRgbs, C.size_t(len(rgbs)),
		(*C.ov_genai_generation_config)(cConfig),
		&streamer_callback,
		&result)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)

	var metrics *C.ov_genai_perf_metrics
	C.ov_genai_vlm_decoded_results_get_perf_metrics(result, &metrics)

	applyPerfMetricsToSequence(metrics, seq)
	PrintGenaiMetrics(metrics)

	out := C.GoString((*C.char)(cOutput))
	log.Printf("vlm result: %s", out)
	return out
}

func VlmGenerateText(pipeline *C.ov_genai_vlm_pipeline, input string, images []llamaserver.ImageData, samplingparameters *SamplingParams) string {
	cConfig := SetSamplingParams(samplingparameters)
	var result *C.ov_genai_vlm_decoded_results
	output_size := C.size_t(0)

	var chatHistory *C.ov_genai_chat_history
	chatStatus := C.ov_genai_chat_history_create(&chatHistory)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("vlm: failed to create chat history: %d", chatStatus)
		return ""
	}
	defer C.ov_genai_chat_history_free(chatHistory)

	messageJSON, err := json.Marshal(map[string]string{
		"role":    "user",
		"content": input,
	})
	if err != nil {
		log.Printf("vlm: failed to marshal user message: %v", err)
		return ""
	}

	cMessageJSON := C.CString(string(messageJSON))
	defer C.free(unsafe.Pointer(cMessageJSON))

	var messageContainer *C.ov_genai_json_container
	jsonStatus := C.ov_genai_json_container_create_from_json_string(&messageContainer, cMessageJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("vlm: failed to create message json container: %d", jsonStatus)
		return ""
	}
	defer C.ov_genai_json_container_free(messageContainer)

	chatStatus = C.ov_genai_chat_history_push_back(chatHistory, messageContainer)
	if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
		log.Printf("vlm: failed to push message to chat history: %d", chatStatus)
		return ""
	}

	extraContextJSON := fmt.Sprintf(`{"enable_thinking": %t}`, samplingparameters.EnableThinking)
	cExtraContextJSON := C.CString(extraContextJSON)
	defer C.free(unsafe.Pointer(cExtraContextJSON))

	var extraContextContainer *C.ov_genai_json_container
	jsonStatus = C.ov_genai_json_container_create_from_json_string(&extraContextContainer, cExtraContextJSON)
	if jsonStatus != C.OV_GENAI_JSON_CONTAINER_OK {
		log.Printf("vlm: failed to create extra_context json container: %d", jsonStatus)
	} else {
		defer C.ov_genai_json_container_free(extraContextContainer)
		chatStatus = C.ov_genai_chat_history_set_extra_context(chatHistory, extraContextContainer)
		if chatStatus != C.OV_GENAI_CHAT_HISTORY_OK {
			log.Printf("vlm: failed to set extra_context on chat history: %d", chatStatus)
		}
	}

	rgbs, cRgbs := vlmImageTensors(images)

	C.ov_genai_vlm_pipeline_generate_with_history(pipeline,
		chatHistory,
		cRgbs, C.size_t(len(rgbs)),
		(*C.ov_genai_generation_config)(cConfig),
		nil,
		&result)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_vlm_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)
	return C.GoString((*C.char)(cOutput))
}
func GenerateText(pipeline *C.ov_genai_llm_pipeline, input string, streamer_callback C.streamer_callback) string {
	cInput := C.CString(input)
	defer C.free(unsafe.Pointer(cInput))

	output_size := C.size_t(0)

	var result *C.ov_genai_decoded_results

	C.ov_genai_llm_pipeline_start_chat(pipeline)
	C.ov_genai_llm_pipeline_generate(pipeline, cInput, (*C.ov_genai_generation_config)(nil), &streamer_callback, &result)
	C.ov_genai_llm_pipeline_finish_chat(pipeline)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(nil), &output_size)
	cOutput := C.malloc(output_size)
	defer C.free(cOutput)

	C.ov_genai_decoded_results_get_string(result, (*C.char)(cOutput), &output_size)
	return C.GoString((*C.char)(cOutput))
}

func FreeModel(model Model) {
	C.ov_genai_llm_pipeline_free(model)
}

func GetGenaiAvailableDevices() []map[string]string {
	var core *(C.ov_core_t)
	defer C.ov_core_free(core)

	C.ov_core_create(&core)

	var devices C.ov_available_devices_t
	C.ov_core_get_available_devices(core, &devices)

	// goDevice := make(map[string]map[string][]string)
	var goDevices []map[string]string

	cString := devices.devices
	for i := 0; i < int(devices.size); i++ {
		var version_list C.ov_core_version_list_t

		C.ov_core_get_versions_by_device_name(core, (*C.char)(*cString), &version_list)

		cversion_list := version_list
		for i := 0; i < int(version_list.size); i++ {
			version := (*C.ov_core_version_t)(unsafe.Pointer(uintptr(unsafe.Pointer(cversion_list.versions)) + uintptr(i)*unsafe.Sizeof(*cversion_list.versions)))
			deviceName := C.GoString(version.device_name) // 获取 device_name
			buildNumber := C.GoString(version.version.buildNumber)
			description := C.GoString(version.version.description)

			item := map[string]string{"device_name": deviceName, "buildNumber": buildNumber, "description": description}
			goDevices = append(goDevices, item)
		}

		cString = (**C.char)(unsafe.Pointer(uintptr(unsafe.Pointer(cString)) + unsafe.Sizeof(*cString)))

	}
	defer C.ov_available_devices_free(&devices)

	return goDevices
}

func GetOvVersion() {
	var ov_version C.ov_version_t
	C.ov_get_openvino_version(&ov_version)
	log.Printf("---- OpenVINO INFO----\n")
	log.Printf("Description : %s \n", C.GoString(ov_version.description))
	log.Printf("Build number: %s \n", C.GoString(ov_version.buildNumber))

	defer C.ov_version_free(&ov_version)
}

//export goCallbackBridge
func goCallbackBridge(args *C.char, gen_result unsafe.Pointer) C.int {
	if args != nil {
		// 将 unsafe.Pointer 转换回结构体指针
		handle := cgo.Handle(uintptr(gen_result))
		result := handle.Value().(*Sequence)

		// 将 C 字符串转换为 Go 字符串并追加到切片中
		goStr := C.GoString(args)
		result.AppendPendingResponse(goStr)

		// fmt.Printf("%s", goStr)
		// os.Stdout.Sync()
		FlushPending((*Sequence)(result))
		return C.int(C.OV_GENAI_STREAMING_STATUS_RUNNING_COMPAT)
	} else {
		fmt.Println("Callback executed with NULL message!")
		return C.int(C.OV_GENAI_STREAMING_STATUS_STOP_COMPAT)
	}
}

func FreeVlmModel(pipeline *C.ov_genai_vlm_pipeline) {
	C.ov_genai_vlm_pipeline_free(pipeline)
}

// func main() {
// 	// 使用 createPipeline 函数创建 pipeline
// 	// log.Printf(GetGenaiAvailableDevices()[0]["device_name"])
// 	GetOvVersion()
// 	pipeline := CreatePipeline("/home/hongbo/ollama_model/TinyLlama-1.1B-Chat-v1.0-int4-ov", "CPU")
// 	if pipeline == nil {
// 		fmt.Println("创建 pipeline 失败")
// 		return
// 	}
// 	fmt.Println("成功创建 pipeline")

// 	// 创建 streamer_callback
// 	var streamer_callback C.streamer_callback
// 	streamer_callback.callback_func = (C.callback_function)(unsafe.Pointer(C.goCallbackBridge))

// 	// 初始化 gena_result 结构体
// 	gena_result := &gen_result_struct{
// 		pendingResponses: make([]string, 0),
// 	}

// 	// 使用 runtime.Pinner 固定指针
// 	var pinner runtime.Pinner
// 	pinner.Pin(gena_result)
// 	defer pinner.Unpin()

// 	streamer_callback.args = unsafe.Pointer(gena_result)

// 	result := GenerateText(pipeline, "who you are?", streamer_callback)
// 	fmt.Println("生成的结果 result:", result)

// 	if gena_result != nil && len(gena_result.pendingResponses) > 0 {
// 		fmt.Println("生成的结果 gen_result:", gena_result.pendingResponses)
// 	} else {
// 		fmt.Println("生成的结果 gen_result: 无响应")
// 	}
// 	// 释放 pipeline 资源
// 	FreeModel(pipeline)
// }
