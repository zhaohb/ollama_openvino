// Command ollama-registry runs a self-hosted Ollama Registry v2 service that
// preserves OpenVINO-specific manifest layers end-to-end.
//
// Typical usage:
//
//	ollama-registry serve --addr :5000 --root C:\ollama-registry
//	# then on a client:
//	ollama push --insecure http://127.0.0.1:5000/zhaohb/qwen3-4b-ov:v1
//	ollama pull --insecure http://127.0.0.1:5000/zhaohb/qwen3-4b-ov:v1
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/ollama/ollama/registryserver"
)

func main() {
	if len(os.Args) < 2 {
		usage(os.Stderr)
		os.Exit(2)
	}

	switch os.Args[1] {
	case "serve":
		if err := runServe(os.Args[2:]); err != nil {
			fmt.Fprintf(os.Stderr, "ollama-registry: %v\n", err)
			os.Exit(1)
		}
	case "-h", "--help", "help":
		usage(os.Stdout)
	default:
		fmt.Fprintf(os.Stderr, "ollama-registry: unknown command %q\n\n", os.Args[1])
		usage(os.Stderr)
		os.Exit(2)
	}
}

func usage(w *os.File) {
	fmt.Fprintln(w, `Usage: ollama-registry serve [flags]

Self-hosted Ollama Registry v2 server that supports OpenVINO model layers.

Flags:
  --addr string      Listen address (default ":5000")
  --root string      Storage root for blobs, uploads and manifests (required)
  --token string     Optional global Bearer token required on every request
  --disable-signup   Disable open registration (existing accounts can still log in)
  --cookie-secure    Mark the session cookie Secure (HTTPS deployments)

Environment:
  OLLAMA_REGISTRY_TOKEN  Same effect as --token, useful for service managers.

This registry has a built-in username/password account system (always on):
  * Anonymous visitors may browse and pull PUBLIC models only.
  * Register an account on the dashboard; your username becomes your namespace.
  * Only the namespace owner may push (login == namespace).
  * New pushes are PRIVATE by default — mark a model public from its page on
    the dashboard. A logged-in user sees their own models plus everyone's public
    models; other users' private models are hidden from the dashboard, the JSON
    API, and OCI pulls.
  * To pull a PRIVATE model with the CLI, create a personal token on your
    namespace page and pass it as a Bearer token (OLLAMA_REGISTRY_TOKEN).

Once the server is running, point Ollama at it with the http scheme and the
--insecure flag, e.g.:

  ollama push --insecure http://127.0.0.1:5000/<namespace>/<model>:<tag>
  ollama pull --insecure http://127.0.0.1:5000/<namespace>/<model>:<tag>`)
}

func runServe(args []string) error {
	fs := flag.NewFlagSet("serve", flag.ContinueOnError)
	addr := fs.String("addr", ":5000", "listen address")
	root := fs.String("root", "", "storage root directory")
	token := fs.String("token", os.Getenv("OLLAMA_REGISTRY_TOKEN"), "optional Bearer token")
	disableSignup := fs.Bool("disable-signup", false, "disable open registration")
	cookieSecure := fs.Bool("cookie-secure", false, "mark session cookie Secure (HTTPS only)")
	if err := fs.Parse(args); err != nil {
		return err
	}
	if *root == "" {
		return errors.New("--root is required")
	}

	store, err := registryserver.NewStore(*root)
	if err != nil {
		return fmt.Errorf("init store: %w", err)
	}
	defer store.Close()

	auth := &registryserver.AuthConfig{
		DisableSignup: *disableSignup,
		CookieSecure:  *cookieSecure,
	}

	logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelInfo}))
	handler := registryserver.NewServer(store, logger, *token, auth)

	server := &http.Server{
		Addr:              *addr,
		Handler:           handler,
		ReadHeaderTimeout: 30 * time.Second,
	}

	logger.Info("ollama-registry serving", "addr", *addr, "root", store.Root,
		"token_auth", *token != "", "signup", !auth.DisableSignup)

	idle := make(chan struct{})
	go func() {
		sig := make(chan os.Signal, 1)
		signal.Notify(sig, os.Interrupt, syscall.SIGTERM)
		<-sig
		logger.Info("shutdown signal received")
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := server.Shutdown(ctx); err != nil {
			logger.Error("graceful shutdown failed", "err", err)
		}
		close(idle)
	}()

	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		return err
	}
	<-idle
	return nil
}
