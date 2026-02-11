package config

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"os"
	"slices"
	"strings"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/progress"
	"github.com/spf13/cobra"
)

// Runner executes the launching of an integration with a model.
type Runner interface {
	Run(model string, args []string) error
	// String returns the human-readable name of the integration.
	String() string
}

// Editor can edit config files (supports multi-model selection).
type Editor interface {
	// Paths returns the paths to the config files for the integration.
	Paths() []string
	// Edit updates the config files for the integration with the given models.
	Edit(models []string) error
	// Models returns the models currently configured for the integration.
	Models() []string
}

// integrations is the registry of available integrations (OpenVINO fork: only OpenClaw).
var integrations = map[string]Runner{
	"openclaw": &Openclaw{},
	"clawdbot": &Openclaw{}, // alias
	"moltbot":  &Openclaw{}, // alias
}

// integrationAliases are hidden from the interactive selector but work as CLI arguments.
var integrationAliases = map[string]bool{
	"clawdbot": true,
	"moltbot":  true,
}

// recommendedModels are shown when the user has no models or as suggestions (local only in OpenVINO fork).
// Order matters.
var recommendedModels = []ModelItem{
	{Name: "qwen3-coder", Description: "Recommended"},
	{Name: "glm-4.7", Description: "Recommended"},
	{Name: "gpt-oss:20b", Description: "Recommended"},
}

// ModelItem represents a model for selection.
type ModelItem struct {
	Name        string
	Description string
}

type modelInfo struct {
	Name string
}

// SelectModelWithSelector prompts the user to select a model using the provided selector.
func SelectModelWithSelector(ctx context.Context, selector func(title string, items []ModelItem) (string, error)) (string, error) {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return "", err
	}

	models, err := client.List(ctx)
	if err != nil {
		return "", err
	}

	var existing []modelInfo
	for _, m := range models.Models {
		existing = append(existing, modelInfo{Name: m.Name})
	}

	items, _, existingModels := buildModelList(existing, nil, "")
	if len(items) == 0 {
		return "", fmt.Errorf("no models available, run 'ollama pull <model>' first")
	}

	selected, err := selector("Select model to run:", items)
	if err != nil {
		return "", err
	}

	// If the selected model isn't installed, pull it first.
	if !existingModels[selected] {
		msg := fmt.Sprintf("Download %s?", selected)
		if ok, err := confirmPrompt(msg); err != nil {
			return "", err
		} else if !ok {
			return "", errCancelled
		}
		fmt.Fprintf(os.Stderr, "\n")
		if err := pullModel(ctx, client, selected); err != nil {
			return "", fmt.Errorf("failed to pull %s: %w", selected, err)
		}
	}

	return selected, nil
}

func defaultSingleSelector(title string, items []ModelItem) (string, error) {
	selectItems := make([]selectItem, len(items))
	for i, item := range items {
		selectItems[i] = selectItem(item)
	}
	return selectPrompt(title, selectItems)
}

func defaultMultiSelector(title string, items []ModelItem, preChecked []string) ([]string, error) {
	selectItems := make([]selectItem, len(items))
	for i, item := range items {
		selectItems[i] = selectItem(item)
	}
	return multiSelectPrompt(title, selectItems, preChecked)
}

func selectIntegration() (string, error) {
	var items []selectItem
	for name, r := range integrations {
		if integrationAliases[name] {
			continue
		}
		items = append(items, selectItem{Name: name, Description: r.String()})
	}
	slices.SortFunc(items, func(a, b selectItem) int {
		return strings.Compare(strings.ToLower(a.Name), strings.ToLower(b.Name))
	})
	return selectPrompt("Select integration:", items)
}

func selectModels(ctx context.Context, name, current string) ([]string, error) {
	r, ok := integrations[name]
	if !ok {
		return nil, fmt.Errorf("unknown integration: %s", name)
	}

	client, err := api.ClientFromEnvironment()
	if err != nil {
		return nil, err
	}

	list, err := client.List(ctx)
	if err != nil {
		return nil, err
	}

	var existing []modelInfo
	for _, m := range list.Models {
		existing = append(existing, modelInfo{Name: m.Name})
	}

	var preChecked []string
	if saved, err := loadIntegration(name); err == nil {
		preChecked = saved.Models
	} else if editor, ok := r.(Editor); ok {
		preChecked = editor.Models()
	}

	items, preChecked, existingModels := buildModelList(existing, preChecked, current)
	if len(items) == 0 {
		return nil, fmt.Errorf("no models available")
	}

	if _, ok := r.(Editor); ok {
		selected, err := defaultMultiSelector(fmt.Sprintf("Select models for %s:", r), items, preChecked)
		if errors.Is(err, errCancelled) {
			return nil, errCancelled
		}
		if err != nil {
			return nil, err
		}

		// Pull any that are not installed.
		for _, m := range selected {
			if !existingModels[m] {
				if ok, err := confirmPrompt(fmt.Sprintf("Download %s?", m)); err != nil {
					return nil, err
				} else if !ok {
					return nil, errCancelled
				}
				fmt.Fprintf(os.Stderr, "\n")
				if err := pullModel(ctx, client, m); err != nil {
					return nil, fmt.Errorf("failed to pull %s: %w", m, err)
				}
			}
		}
		return selected, nil
	}

	// Non-editor integration would be single model selection (not used here).
	model, err := defaultSingleSelector(fmt.Sprintf("Select model for %s:", r), items)
	if errors.Is(err, errCancelled) {
		return nil, errCancelled
	}
	if err != nil {
		return nil, err
	}
	if !existingModels[model] {
		if ok, err := confirmPrompt(fmt.Sprintf("Download %s?", model)); err != nil {
			return nil, err
		} else if !ok {
			return nil, errCancelled
		}
		fmt.Fprintf(os.Stderr, "\n")
		if err := pullModel(ctx, client, model); err != nil {
			return nil, fmt.Errorf("failed to pull %s: %w", model, err)
		}
	}
	return []string{model}, nil
}

func runIntegration(name, modelName string, args []string) error {
	r, ok := integrations[name]
	if !ok {
		return fmt.Errorf("unknown integration: %s", name)
	}

	fmt.Fprintf(os.Stderr, "\nLaunching %s with %s...\n", r, modelName)
	return r.Run(modelName, args)
}

// showOrPull checks if a model exists via client.Show and offers to pull it if not found.
func showOrPull(ctx context.Context, client *api.Client, model string) error {
	if _, err := client.Show(ctx, &api.ShowRequest{Model: model}); err == nil {
		return nil
	}
	if ok, err := confirmPrompt(fmt.Sprintf("Download %s?", model)); err != nil {
		return err
	} else if !ok {
		return errCancelled
	}
	fmt.Fprintf(os.Stderr, "\n")
	return pullModel(ctx, client, model)
}

// LaunchCmd returns the cobra command for launching integrations.
// The runDefault callback is called when no arguments are provided (OpenVINO fork: show help/usage).
func LaunchCmd(checkServerHeartbeat func(cmd *cobra.Command, args []string) error, runDefault func(cmd *cobra.Command)) *cobra.Command {
	var modelFlag string
	var configFlag bool

	cmd := &cobra.Command{
		Use:   "launch [INTEGRATION] [-- [EXTRA_ARGS...]]",
		Short: "Launch an integration",
		Long: `Launch a specific integration.

Supported integrations:
  openclaw  OpenClaw (aliases: clawdbot, moltbot)

Examples:
  ollama launch openclaw
  ollama launch openclaw --model <model>
  ollama launch openclaw --config (does not auto-launch)
  ollama launch openclaw -- --some-arg (pass extra args to integration)`,
		Args:    cobra.ArbitraryArgs,
		PreRunE: checkServerHeartbeat,
		RunE: func(cmd *cobra.Command, args []string) error {
			// No args - default behavior (do not change existing openvino root behavior).
			if len(args) == 0 && modelFlag == "" && !configFlag {
				runDefault(cmd)
				return nil
			}

			// Extract integration name and args to pass through using -- separator.
			var name string
			var passArgs []string
			dashIdx := cmd.ArgsLenAtDash()

			if dashIdx == -1 {
				// No "--" separator: only allow 0 or 1 args (integration name).
				if len(args) > 1 {
					return fmt.Errorf("unexpected arguments: %v\nUse '--' to pass extra arguments to the integration", args[1:])
				}
				if len(args) == 1 {
					name = args[0]
				}
			} else {
				// "--" was used: args before it = integration name, args after = passthrough.
				if dashIdx > 1 {
					return fmt.Errorf("expected at most 1 integration name before '--', got %d", dashIdx)
				}
				if dashIdx == 1 {
					name = args[0]
				}
				passArgs = args[dashIdx:]
			}

			if name == "" {
				var err error
				name, err = selectIntegration()
				if errors.Is(err, errCancelled) {
					return nil
				}
				if err != nil {
					return err
				}
			}

			name = strings.ToLower(name)
			r, ok := integrations[name]
			if !ok {
				return fmt.Errorf("unknown integration: %s", name)
			}

			// Validate --model flag if provided.
			if modelFlag != "" {
				client, err := api.ClientFromEnvironment()
				if err != nil {
					return err
				}
				if err := showOrPull(cmd.Context(), client, modelFlag); err != nil {
					if errors.Is(err, errCancelled) {
						return nil
					}
					return err
				}
			}

			var models []string
			if modelFlag != "" {
				models = []string{modelFlag}
				if existing, err := loadIntegration(name); err == nil && len(existing.Models) > 0 {
					for _, m := range existing.Models {
						if m != modelFlag {
							models = append(models, m)
						}
					}
				}
			} else if saved, err := loadIntegration(name); err == nil && len(saved.Models) > 0 && !configFlag {
				return runIntegration(name, saved.Models[0], passArgs)
			} else {
				var err error
				models, err = selectModels(cmd.Context(), name, "")
				if errors.Is(err, errCancelled) {
					return nil
				}
				if err != nil {
					return err
				}
			}

			if editor, isEditor := r.(Editor); isEditor {
				paths := editor.Paths()
				if len(paths) > 0 {
					fmt.Fprintf(os.Stderr, "This will modify your %s configuration:\n", r)
					for _, p := range paths {
						fmt.Fprintf(os.Stderr, "  %s\n", p)
					}
					fmt.Fprintf(os.Stderr, "Backups will be saved to %s/\n\n", backupDir())

					if ok, _ := confirmPrompt("Proceed?"); !ok {
						return nil
					}
				}
			}

			if err := saveIntegration(name, models); err != nil {
				return fmt.Errorf("failed to save: %w", err)
			}

			if editor, isEditor := r.(Editor); isEditor {
				if err := editor.Edit(models); err != nil {
					return fmt.Errorf("setup failed: %w", err)
				}
			}

			if _, isEditor := r.(Editor); isEditor {
				if len(models) == 1 {
					fmt.Fprintf(os.Stderr, "Added %s to %s\n", models[0], r)
				} else {
					fmt.Fprintf(os.Stderr, "Added %d models to %s (default: %s)\n", len(models), r, models[0])
				}
			}

			if configFlag {
				if launch, _ := confirmPrompt(fmt.Sprintf("\nLaunch %s now?", r)); launch {
					return runIntegration(name, models[0], passArgs)
				}
				fmt.Fprintf(os.Stderr, "Run 'ollama launch %s' to start with %s\n", strings.ToLower(name), models[0])
				return nil
			}

			return runIntegration(name, models[0], passArgs)
		},
	}

	cmd.Flags().StringVar(&modelFlag, "model", "", "Model to use")
	cmd.Flags().BoolVar(&configFlag, "config", false, "Configure without launching")
	return cmd
}

// buildModelList merges existing models with recommendations, sorts them, and returns
// the ordered items along with a map of existing model names.
func buildModelList(existing []modelInfo, preChecked []string, current string) (items []ModelItem, orderedChecked []string, existingModels map[string]bool) {
	existingModels = make(map[string]bool)
	recommended := make(map[string]bool)

	for _, rec := range recommendedModels {
		recommended[rec.Name] = true
	}

	for _, m := range existing {
		existingModels[m.Name] = true
		displayName := strings.TrimSuffix(m.Name, ":latest")
		existingModels[displayName] = true

		item := ModelItem{Name: displayName}
		if recommended[displayName] {
			item.Description = "recommended"
		}
		items = append(items, item)
	}

	for _, rec := range recommendedModels {
		if existingModels[rec.Name] || existingModels[rec.Name+":latest"] {
			continue
		}
		items = append(items, rec)
	}

	checked := make(map[string]bool, len(preChecked))
	for _, n := range preChecked {
		checked[n] = true
	}

	// Resolve current to full name (e.g., "llama3.2" -> "llama3.2:latest").
	for _, item := range items {
		if item.Name == current || strings.HasPrefix(item.Name, current+":") {
			current = item.Name
			break
		}
	}

	if checked[current] {
		preChecked = append([]string{current}, slices.DeleteFunc(preChecked, func(m string) bool { return m == current })...)
	}

	// Non-existing models get "install?" suffix and are pushed to the bottom.
	notInstalled := make(map[string]bool)
	for i := range items {
		if !existingModels[items[i].Name] {
			notInstalled[items[i].Name] = true
			if items[i].Description != "" {
				items[i].Description += ", install?"
			} else {
				items[i].Description = "install?"
			}
		}
	}

	slices.SortStableFunc(items, func(a, b ModelItem) int {
		ac, bc := checked[a.Name], checked[b.Name]
		aNew, bNew := notInstalled[a.Name], notInstalled[b.Name]

		if ac != bc {
			if ac {
				return -1
			}
			return 1
		}
		if !ac && !bc && aNew != bNew {
			if aNew {
				return 1
			}
			return -1
		}
		return strings.Compare(strings.ToLower(a.Name), strings.ToLower(b.Name))
	})

	return items, preChecked, existingModels
}

func pullModel(ctx context.Context, client *api.Client, model string) error {
	p := progress.NewProgress(os.Stderr)
	defer p.Stop()

	bars := make(map[string]*progress.Bar)
	var status string
	var spinner *progress.Spinner

	fn := func(resp api.ProgressResponse) error {
		if resp.Digest != "" {
			if resp.Completed == 0 {
				return nil
			}

			if spinner != nil {
				spinner.Stop()
			}

			bar, ok := bars[resp.Digest]
			if !ok {
				name, isDigest := strings.CutPrefix(resp.Digest, "sha256:")
				name = strings.TrimSpace(name)
				if isDigest {
					name = name[:min(12, len(name))]
				}
				bar = progress.NewBar(fmt.Sprintf("pulling %s:", name), resp.Total, resp.Completed)
				bars[resp.Digest] = bar
				p.Add(resp.Digest, bar)
			}

			bar.Set(resp.Completed)
		} else if status != resp.Status {
			if spinner != nil {
				spinner.Stop()
			}

			status = resp.Status
			spinner = progress.NewSpinner(status)
			p.Add(status, spinner)
		}

		return nil
	}

	request := api.PullRequest{Name: model, Model: model}
	// OpenVINO fork client ignores Insecure; keep fields for compatibility.
	return client.Pull(ctx, &request, fn)
}

// quick check to avoid hard-to-understand errors when ollama isn't reachable.
func checkServerReachable(ctx context.Context) error {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		return err
	}
	if err := client.Heartbeat(ctx); err != nil {
		// Mirror common error classification in other commands.
		if strings.Contains(err.Error(), " refused") || strings.Contains(err.Error(), "could not connect") {
			return fmt.Errorf("ollama server not responding: %w", err)
		}
		return err
	}
	return nil
}

// Optional: ensure HTTP status errors get printed nicely by cobra.
func isNotFound(err error) bool {
	var se api.StatusError
	if errors.As(err, &se) && se.StatusCode == http.StatusNotFound {
		return true
	}
	return false
}

