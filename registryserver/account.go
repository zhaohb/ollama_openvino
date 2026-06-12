package registryserver

import (
	"net/http"
	"strings"
)

// handleAccountAPI serves the logged-in user's self-service endpoints:
//
//	POST /api/account/token                  -> issue a new personal token (returns it once)
//	POST /api/account/token/revoke           -> revoke all of the user's tokens
//	POST /api/account/visibility/{model}     -> set own model public/private (form: public=true|false)
//
// All require a logged-in session and act only on the caller's own namespace.
func (s *Server) handleAccountAPI(w http.ResponseWriter, r *http.Request) {
	login, ok := s.currentUser(r)
	if !ok {
		w.Header().Set("WWW-Authenticate", `Bearer realm="registry"`)
		writeError(w, http.StatusUnauthorized, "UNAUTHORIZED", "login required")
		return
	}
	if r.Method != http.MethodPost {
		writeError(w, http.StatusMethodNotAllowed, "METHOD_NOT_ALLOWED", r.Method)
		return
	}

	rest := strings.Trim(strings.TrimPrefix(r.URL.Path, "/api/account"), "/")
	parts := strings.Split(rest, "/")

	switch {
	case len(parts) == 1 && parts[0] == "token":
		tok, err := s.Store.CreateToken(login)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		// The plaintext token is returned exactly once.
		s.writeJSON(w, map[string]any{"token": tok, "login": login})

	case len(parts) == 2 && parts[0] == "token" && parts[1] == "revoke":
		n, err := s.Store.RevokeTokensFor(login)
		if err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		s.writeJSON(w, map[string]any{"revoked": n})

	case len(parts) == 2 && parts[0] == "visibility":
		model := parts[1]
		if err := ValidateName(login, model, "x"); err != nil {
			writeError(w, http.StatusBadRequest, "NAME_INVALID", err.Error())
			return
		}
		// A user can only ever change visibility within their own namespace,
		// which is login — there is no namespace parameter to spoof.
		public := strings.EqualFold(r.FormValue("public"), "true")
		if err := s.Store.SetVisibility(login, model, public); err != nil {
			writeError(w, http.StatusInternalServerError, "INTERNAL", err.Error())
			return
		}
		s.writeJSON(w, map[string]any{
			"namespace": login,
			"model":     model,
			"public":    public,
		})

	default:
		writeError(w, http.StatusNotFound, "NOT_FOUND", "unrecognised account path")
	}
}
