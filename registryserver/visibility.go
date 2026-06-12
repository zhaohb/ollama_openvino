package registryserver

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"golang.org/x/crypto/bcrypt"
)

// ErrUserExists is returned by CreateUser when the username is already taken.
var ErrUserExists = errors.New("user already exists")

// Visibility and personal-access-token persistence for the registry.
//
// Both are kept as small JSON files under the Store root, matching the existing
// file-based design (no database, no cgo):
//
//	auth/visibility.json   { "<namespace>/<model>": "public", ... }
//	auth/tokens.json       { "<sha256(token)>": "<login>", ... }
//
// Models absent from visibility.json are PRIVATE by default. Tokens are stored
// only as sha256 hashes, never in plaintext.

var (
	visMu   sync.Mutex
	tokenMu sync.Mutex
	userMu  sync.Mutex
)

func (s *Store) authDir() string  { return filepath.Join(s.Root, "auth") }
func (s *Store) visPath() string  { return filepath.Join(s.authDir(), "visibility.json") }
func (s *Store) tokPath() string  { return filepath.Join(s.authDir(), "tokens.json") }
func (s *Store) userPath() string { return filepath.Join(s.authDir(), "users.json") }

func repoKey(namespace, model string) string { return namespace + "/" + model }

// loadJSONMap reads a string->string JSON map, treating a missing file as an
// empty map.
func loadJSONMap(path string) (map[string]string, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return map[string]string{}, nil
	}
	if err != nil {
		return nil, err
	}
	m := map[string]string{}
	if len(data) == 0 {
		return m, nil
	}
	if err := json.Unmarshal(data, &m); err != nil {
		return nil, err
	}
	return m, nil
}

// writeJSONMap writes m atomically (temp file + rename).
func writeJSONMap(path string, m map[string]string) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(m, "", "  ")
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// IsPublic reports whether namespace/model is marked public. Unknown models are
// private by default.
func (s *Store) IsPublic(namespace, model string) bool {
	visMu.Lock()
	defer visMu.Unlock()
	m, err := loadJSONMap(s.visPath())
	if err != nil {
		// Fail closed: an unreadable visibility file means we can't prove the
		// model is public, so treat it as private.
		return false
	}
	return m[repoKey(namespace, model)] == "public"
}

// SetVisibility marks namespace/model public (public==true) or private. Setting
// private removes the entry so the file only ever lists public models.
func (s *Store) SetVisibility(namespace, model string, public bool) error {
	if err := ValidateName(namespace, model, "x"); err != nil {
		return err
	}
	visMu.Lock()
	defer visMu.Unlock()
	m, err := loadJSONMap(s.visPath())
	if err != nil {
		return err
	}
	key := repoKey(namespace, model)
	if public {
		m[key] = "public"
	} else {
		delete(m, key)
	}
	return writeJSONMap(s.visPath(), m)
}

// PublicModels returns the set of "<ns>/<model>" keys currently public, for
// callers that want to filter listings in bulk.
func (s *Store) PublicModels() (map[string]bool, error) {
	visMu.Lock()
	defer visMu.Unlock()
	m, err := loadJSONMap(s.visPath())
	if err != nil {
		return nil, err
	}
	out := make(map[string]bool, len(m))
	for k, v := range m {
		if v == "public" {
			out[k] = true
		}
	}
	return out, nil
}

// ---- personal access tokens ------------------------------------------------

// CreateToken issues a new personal access token for login, persists only its
// hash, and returns the plaintext token exactly once (the caller must show it
// to the user immediately; it cannot be recovered later).
func (s *Store) CreateToken(login string) (string, error) {
	plain, err := randToken()
	if err != nil {
		return "", err
	}
	tokenMu.Lock()
	defer tokenMu.Unlock()
	m, err := loadJSONMap(s.tokPath())
	if err != nil {
		return "", err
	}
	m[hashToken(plain)] = login
	if err := writeJSONMap(s.tokPath(), m); err != nil {
		return "", err
	}
	return plain, nil
}

// UserForToken resolves the login that owns a personal access token, by hash.
func (s *Store) UserForToken(token string) (string, bool) {
	tokenMu.Lock()
	defer tokenMu.Unlock()
	m, err := loadJSONMap(s.tokPath())
	if err != nil {
		return "", false
	}
	login, ok := m[hashToken(token)]
	return login, ok
}

// RevokeTokensFor deletes all personal tokens owned by login. Returns the count
// removed.
func (s *Store) RevokeTokensFor(login string) (int, error) {
	tokenMu.Lock()
	defer tokenMu.Unlock()
	m, err := loadJSONMap(s.tokPath())
	if err != nil {
		return 0, err
	}
	removed := 0
	for h, owner := range m {
		if owner == login {
			delete(m, h)
			removed++
		}
	}
	if removed == 0 {
		return 0, nil
	}
	return removed, writeJSONMap(s.tokPath(), m)
}

// TokenCountFor returns how many personal tokens login currently has, so the
// dashboard can show whether a token already exists.
func (s *Store) TokenCountFor(login string) (int, error) {
	tokenMu.Lock()
	defer tokenMu.Unlock()
	m, err := loadJSONMap(s.tokPath())
	if err != nil {
		return 0, err
	}
	n := 0
	for _, owner := range m {
		if owner == login {
			n++
		}
	}
	return n, nil
}

// ---- user accounts ---------------------------------------------------------

// CreateUser registers a new account. The password is stored only as a bcrypt
// hash. Usernames are case-insensitive (normalised to lower case) so "Alice"
// and "alice" cannot both be claimed. Returns ErrUserExists if taken.
func (s *Store) CreateUser(login, password string) error {
	login = strings.ToLower(strings.TrimSpace(login))
	userMu.Lock()
	defer userMu.Unlock()
	m, err := loadJSONMap(s.userPath())
	if err != nil {
		return err
	}
	if _, exists := m[login]; exists {
		return ErrUserExists
	}
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return err
	}
	m[login] = string(hash)
	return writeJSONMap(s.userPath(), m)
}

// VerifyUser reports whether login+password match a stored account. A missing
// user and a wrong password are indistinguishable to the caller (both return
// false, nil) so the login form can't be used to enumerate usernames.
func (s *Store) VerifyUser(login, password string) (bool, error) {
	login = strings.ToLower(strings.TrimSpace(login))
	userMu.Lock()
	defer userMu.Unlock()
	m, err := loadJSONMap(s.userPath())
	if err != nil {
		return false, err
	}
	hash, ok := m[login]
	if !ok {
		// Still run a bcrypt comparison against a dummy hash to keep timing
		// roughly constant whether or not the user exists.
		bcrypt.CompareHashAndPassword([]byte("$2a$10$"+strings.Repeat("x", 53)), []byte(password))
		return false, nil
	}
	if err := bcrypt.CompareHashAndPassword([]byte(hash), []byte(password)); err != nil {
		return false, nil
	}
	return true, nil
}

// UserExists reports whether an account with login exists.
func (s *Store) UserExists(login string) bool {
	login = strings.ToLower(strings.TrimSpace(login))
	userMu.Lock()
	defer userMu.Unlock()
	m, err := loadJSONMap(s.userPath())
	if err != nil {
		return false
	}
	_, ok := m[login]
	return ok
}

// visibleNamespaces filters a namespace list down to those the viewer may see:
// a namespace is visible if it is the viewer's own, or it contains at least one
// public model. login may be "" for anonymous callers.
func (s *Store) visibleNamespaces(all []string, login string) ([]string, error) {
	pub, err := s.PublicModels()
	if err != nil {
		return nil, err
	}
	publicNS := map[string]bool{}
	for key := range pub {
		if i := strings.IndexByte(key, '/'); i > 0 {
			publicNS[key[:i]] = true
		}
	}
	out := make([]string, 0, len(all))
	for _, ns := range all {
		if ns == login || publicNS[ns] {
			out = append(out, ns)
		}
	}
	sort.Strings(out)
	return out, nil
}
