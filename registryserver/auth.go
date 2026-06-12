package registryserver

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"strings"
	"sync"
	"time"
)

// AuthConfig controls the built-in username/password account system.
//
// The account system is always on: the dashboard shows Sign in / Register and
// pushing requires login. DisableSignup turns off open registration (existing
// accounts can still log in; new ones must be created out of band).
type AuthConfig struct {
	// DisableSignup, when true, hides the registration page and rejects the
	// register endpoint. Open registration is the default.
	DisableSignup bool

	// CookieSecure marks the session cookie Secure (HTTPS only). Leave false for
	// plain-http local use.
	CookieSecure bool
}

const (
	sessionCookie = "registry_session"
	sessionTTL    = 7 * 24 * time.Hour
)

// session is a logged-in browser session keyed by an opaque random cookie.
// Sessions are in-memory only: a server restart invalidates them (accounts
// themselves are persisted; users just log in again).
type session struct {
	Login   string
	Expires time.Time
}

// authState is the in-memory session table. It is safe for concurrent use.
type authState struct {
	mu       sync.Mutex
	sessions map[string]session
}

func newAuthState() *authState {
	return &authState{sessions: make(map[string]session)}
}

// randToken returns a URL-safe random string with 256 bits of entropy.
func randToken() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return hex.EncodeToString(b), nil
}

func (a *authState) create(login string) (string, error) {
	id, err := randToken()
	if err != nil {
		return "", err
	}
	a.mu.Lock()
	a.sessions[id] = session{Login: login, Expires: time.Now().Add(sessionTTL)}
	a.mu.Unlock()
	return id, nil
}

func (a *authState) lookup(id string) (string, bool) {
	a.mu.Lock()
	defer a.mu.Unlock()
	s, ok := a.sessions[id]
	if !ok {
		return "", false
	}
	if time.Now().After(s.Expires) {
		delete(a.sessions, id)
		return "", false
	}
	return s.Login, true
}

func (a *authState) destroy(id string) {
	a.mu.Lock()
	delete(a.sessions, id)
	a.mu.Unlock()
}

// currentUser resolves the identity behind a request. It returns the account
// login and true when the caller is authenticated, via (in order):
//
//  1. a valid session cookie (browser), or
//  2. an "Authorization: Bearer <personal-token>" header (CLI pull/push).
//
// When no identity is present it returns ("", false): callers treat that as
// "anonymous" — anonymous users may browse and pull public models only.
func (s *Server) currentUser(r *http.Request) (string, bool) {
	if c, err := r.Cookie(sessionCookie); err == nil && c.Value != "" {
		if login, ok := s.auth.lookup(c.Value); ok {
			return login, true
		}
	}
	if authz := r.Header.Get("Authorization"); strings.HasPrefix(authz, "Bearer ") {
		tok := strings.TrimSpace(strings.TrimPrefix(authz, "Bearer "))
		// The global admin token (if set) is not a user; skip it here so it does
		// not masquerade as a namespace owner.
		if tok != "" && tok != s.Token {
			if login, ok := s.Store.UserForToken(tok); ok {
				return login, true
			}
		}
	}
	return "", false
}

// hasAdminToken reports whether the request carries the global admin Bearer
// token (Server.Token). When set, that token grants full read/write access
// across all namespaces — intended for operators and CI, not end users.
func (s *Server) hasAdminToken(r *http.Request) bool {
	if s.Token == "" {
		return false
	}
	return r.Header.Get("Authorization") == "Bearer "+s.Token
}

// canView reports whether a request may see namespace/model.
//
//   - public models are visible to everyone, including anonymous callers;
//   - private models are visible only to their owner (login == namespace);
//   - the global admin token sees everything.
func (s *Server) canView(r *http.Request, namespace, model string) bool {
	if s.Store.IsPublic(namespace, model) {
		return true
	}
	if s.hasAdminToken(r) {
		return true
	}
	login, ok := s.currentUser(r)
	return ok && login == namespace
}

// isOwner reports whether the request's user owns namespace (login == namespace).
func (s *Server) isOwner(r *http.Request, namespace string) bool {
	login, ok := s.currentUser(r)
	return ok && login == namespace
}

// canWrite reports whether a request may push to namespace. Only the namespace
// owner (or the global admin token) may push; anonymous pushes are rejected.
func (s *Server) canWrite(r *http.Request, namespace string) bool {
	return s.hasAdminToken(r) || s.isOwner(r, namespace)
}

// ---- login / register / logout --------------------------------------------

// startSession mints a session for login and sets the cookie.
func (s *Server) startSession(w http.ResponseWriter, login string) error {
	id, err := s.auth.create(login)
	if err != nil {
		return err
	}
	http.SetCookie(w, &http.Cookie{
		Name:     sessionCookie,
		Value:    id,
		Path:     "/",
		HttpOnly: true,
		Secure:   s.Auth.CookieSecure,
		SameSite: http.SameSiteLaxMode,
		Expires:  time.Now().Add(sessionTTL),
	})
	return nil
}

// handleLogin serves the login form (GET) and processes credentials (POST).
func (s *Server) handleLogin(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.renderAuthForm(w, r, "login", "")
	case http.MethodPost:
		login := strings.TrimSpace(r.FormValue("username"))
		password := r.FormValue("password")
		if ok, err := s.Store.VerifyUser(login, password); err != nil {
			s.renderAuthForm(w, r, "login", "internal error")
			return
		} else if !ok {
			s.renderAuthForm(w, r, "login", "invalid username or password")
			return
		}
		if err := s.startSession(w, login); err != nil {
			s.renderAuthForm(w, r, "login", "internal error")
			return
		}
		http.Redirect(w, r, "/"+login, http.StatusFound)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

// handleRegister serves the registration form (GET) and creates an account
// (POST). Open registration unless DisableSignup is set.
func (s *Server) handleRegister(w http.ResponseWriter, r *http.Request) {
	if s.Auth.DisableSignup {
		http.Error(w, "registration is disabled", http.StatusForbidden)
		return
	}
	switch r.Method {
	case http.MethodGet:
		s.renderAuthForm(w, r, "register", "")
	case http.MethodPost:
		login := strings.TrimSpace(r.FormValue("username"))
		password := r.FormValue("password")
		// The username becomes a namespace, so it must satisfy the same rules as
		// any path segment.
		if err := ValidateName(login, "x", "x"); err != nil {
			s.renderAuthForm(w, r, "register", "username may only contain letters, digits, _, . and -")
			return
		}
		if len(password) < 8 {
			s.renderAuthForm(w, r, "register", "password must be at least 8 characters")
			return
		}
		if err := s.Store.CreateUser(login, password); err != nil {
			if err == ErrUserExists {
				s.renderAuthForm(w, r, "register", "that username is already taken")
				return
			}
			s.renderAuthForm(w, r, "register", "internal error")
			return
		}
		if err := s.startSession(w, login); err != nil {
			s.renderAuthForm(w, r, "register", "account created — please sign in")
			return
		}
		http.Redirect(w, r, "/"+login, http.StatusFound)
	default:
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
	}
}

func (s *Server) handleLogout(w http.ResponseWriter, r *http.Request) {
	if c, err := r.Cookie(sessionCookie); err == nil && c.Value != "" {
		s.auth.destroy(c.Value)
	}
	http.SetCookie(w, &http.Cookie{Name: sessionCookie, Value: "", Path: "/", MaxAge: -1})
	http.Redirect(w, r, "/", http.StatusFound)
}

// hashToken returns the storage form of a personal access token. We never store
// the plaintext: only its sha256, so a leaked token file can't be used to pull.
func hashToken(token string) string {
	sum := sha256.Sum256([]byte(token))
	return hex.EncodeToString(sum[:])
}
