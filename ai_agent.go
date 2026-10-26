package main

// AI agent: an in-app evaluator that talks to a LOCAL, OpenAI-compatible LLM
// endpoint (Ollama, LM Studio, llama.cpp's server, vLLM, …) and reviews the
// output of the tools this suite runs.
//
// The operator points the agent at their own model server (base URL + model
// name, optionally an API key) and hands it a chunk of output — a scan result,
// a response body, terminal text — plus a task. The agent then runs a small
// two-pass loop:
//
//   1. analyze  — read the output against the task and lay out what it sees.
//   2. verdict  — review its own analysis and distil a concise, structured
//                 verdict (key findings, confidence, recommended next steps).
//
// Everything streams back token-by-token as newline-delimited JSON, the same
// transport /api/run uses, so the request context cancels the upstream call
// when the browser aborts. The endpoint is gated behind FeatureAI, and the
// configured API key is kept server-side (never echoed back to the browser).

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// aiConfig is the operator's saved LLM connection. It persists next to the
// shared project store so it survives restarts and is shared with the TUI if it
// ever grows an AI panel of its own.
type aiConfig struct {
	BaseURL string `json:"baseURL"` // e.g. http://localhost:11434  or  http://localhost:1234/v1
	Model   string `json:"model"`   // e.g. llama3.1  or  qwen2.5-coder
	APIKey  string `json:"apiKey"`  // optional; most local servers ignore it
}

// aiConfigPath resolves the ai.json alongside projects.json.
func aiConfigPath() string {
	dir, err := os.UserConfigDir()
	if err != nil || dir == "" {
		return sharedProjectsDir + "-ai.json"
	}
	return filepath.Join(dir, sharedProjectsDir, "ai.json")
}

// loadAIConfig reads the saved config, returning a zero config when absent.
func loadAIConfig() aiConfig {
	var c aiConfig
	data, err := os.ReadFile(aiConfigPath())
	if err != nil {
		return c
	}
	_ = json.Unmarshal(data, &c)
	return c
}

// saveAIConfig persists the config, best-effort-created directory.
func saveAIConfig(c aiConfig) error {
	path := aiConfigPath()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600) // 0600: it can hold an API key
}

// handleAIConfig serves and updates the saved connection. The API key is never
// sent back to the browser; the page only learns whether one is set.
func handleAIConfig(w http.ResponseWriter, r *http.Request) {
	if !sameOrigin(r) {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-store")

	if r.Method == http.MethodGet {
		c := loadAIConfig()
		json.NewEncoder(w).Encode(map[string]any{
			"baseURL": c.BaseURL,
			"model":   c.Model,
			"hasKey":  c.APIKey != "",
		})
		return
	}

	// POST: update. An omitted/blank apiKey keeps the stored one, so the page
	// can save base-URL/model edits without re-typing the key.
	var body struct {
		BaseURL string  `json:"baseURL"`
		Model   string  `json:"model"`
		APIKey  *string `json:"apiKey"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&body); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	c := loadAIConfig()
	c.BaseURL = strings.TrimSpace(body.BaseURL)
	c.Model = strings.TrimSpace(body.Model)
	if body.APIKey != nil && strings.TrimSpace(*body.APIKey) != "" {
		c.APIKey = strings.TrimSpace(*body.APIKey)
	}
	if err := saveAIConfig(c); err != nil {
		http.Error(w, "could not save config", http.StatusInternalServerError)
		return
	}
	json.NewEncoder(w).Encode(map[string]any{
		"baseURL": c.BaseURL,
		"model":   c.Model,
		"hasKey":  c.APIKey != "",
		"ok":      true,
	})
}

// aiEvalReq is the body the page POSTs to /api/ai/evaluate.
type aiEvalReq struct {
	Task  string `json:"task"`  // what to evaluate for
	Input string `json:"input"` // the output/content being evaluated
}

// chatMessage is one OpenAI-style chat message.
type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// handleAIEvaluate runs the two-pass evaluation agent against the configured
// local model, streaming each step's tokens back as NDJSON records:
//
//	{"step":"analyze","title":"…"}   — a new step begins
//	{"delta":"…"}                    — streamed model text
//	{"final":true}                   — done   (adds "err" on failure)
func handleAIEvaluate(w http.ResponseWriter, r *http.Request) {
	if !sameOrigin(r) {
		http.Error(w, "forbidden", http.StatusForbidden)
		return
	}

	var req aiEvalReq
	if err := json.NewDecoder(io.LimitReader(r.Body, 8<<20)).Decode(&req); err != nil {
		http.Error(w, "bad request", http.StatusBadRequest)
		return
	}
	req.Task = strings.TrimSpace(req.Task)
	req.Input = strings.TrimSpace(req.Input)

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming unsupported", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/x-ndjson")
	w.Header().Set("Cache-Control", "no-store")
	w.Header().Set("X-Accel-Buffering", "no")
	enc := json.NewEncoder(w)
	send := func(v any) {
		enc.Encode(v)
		flusher.Flush()
	}
	finalErr := func(msg string) { send(map[string]any{"final": true, "err": msg}) }

	cfg := loadAIConfig()
	if cfg.BaseURL == "" || cfg.Model == "" {
		finalErr("no model configured — set a base URL and model name above")
		return
	}
	if req.Input == "" {
		finalErr("nothing to evaluate — paste some output first")
		return
	}
	if req.Task == "" {
		req.Task = "Evaluate this tool output: explain what it shows, flag anything notable or security-relevant, and note what it does not establish."
	}

	client := &http.Client{} // no client timeout; the request context governs cancellation

	// Pass 1 — analyze.
	send(map[string]any{"step": "analyze", "title": "Analyzing output"})
	analysis, err := streamChat(r.Context(), client, cfg, []chatMessage{
		{Role: "system", Content: systemAnalyze},
		{Role: "user", Content: "TASK:\n" + req.Task + "\n\nOUTPUT TO EVALUATE:\n" + clip(req.Input, 24000)},
	}, func(delta string) { send(map[string]any{"delta": delta}) })
	if err != nil {
		finalErr(evalErrMsg(err))
		return
	}

	// Pass 2 — review its own analysis into a structured verdict.
	send(map[string]any{"step": "verdict", "title": "Evaluation & verdict"})
	_, err = streamChat(r.Context(), client, cfg, []chatMessage{
		{Role: "system", Content: systemVerdict},
		{Role: "user", Content: "ANALYSIS YOU PRODUCED:\n" + clip(analysis, 24000) +
			"\n\nNow produce the final verdict in this shape:\n" +
			"Findings: short bullet list of what matters.\n" +
			"Confidence: low / medium / high, with one line of why.\n" +
			"Next steps: concrete, ordered suggestions for the operator.\n" +
			"Flag explicitly if the output is inconclusive or if anything looks like a false positive."},
	}, func(delta string) { send(map[string]any{"delta": delta}) })
	if err != nil {
		finalErr(evalErrMsg(err))
		return
	}

	send(map[string]any{"final": true})
}

const systemAnalyze = "You are a security-tooling output evaluator embedded in a penetration-testing " +
	"assistant used by an authorized operator on their own engagements. You read raw output from " +
	"recon, scanning, and exploitation tools and explain it plainly. Be precise and skeptical: " +
	"distinguish what the output actually proves from what it merely suggests, call out noise and " +
	"likely false positives, and never invent results that are not present in the text. Keep it tight."

const systemVerdict = "You are the same evaluator performing a self-review pass. Turn your own analysis " +
	"into a concise, well-structured verdict for the operator. Do not pad. If the evidence is weak, " +
	"say so. Only recommend next steps that follow from the evidence."

// streamChat POSTs an OpenAI-compatible chat-completion with stream=true,
// forwarding each content delta to onDelta, and returns the full concatenated
// text. It falls back to a single non-streamed request if the server does not
// stream.
func streamChat(ctx context.Context, client *http.Client, cfg aiConfig, msgs []chatMessage, onDelta func(string)) (string, error) {
	payload, _ := json.Marshal(map[string]any{
		"model":       cfg.Model,
		"messages":    msgs,
		"stream":      true,
		"temperature": 0.2,
	})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, chatEndpoint(cfg.BaseURL), bytes.NewReader(payload))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
	}

	resp, err := client.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return "", fmt.Errorf("model server %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}

	// Non-SSE response (server ignored stream=true): decode the whole object.
	if !strings.Contains(resp.Header.Get("Content-Type"), "event-stream") {
		var full chatResponse
		if err := json.NewDecoder(resp.Body).Decode(&full); err != nil {
			return "", err
		}
		text := full.firstContent()
		if text != "" {
			onDelta(text)
		}
		return text, nil
	}

	var sb strings.Builder
	sc := bufio.NewScanner(resp.Body)
	sc.Buffer(make([]byte, 0, 64*1024), 1<<20)
	for sc.Scan() {
		line := strings.TrimSpace(sc.Text())
		if line == "" || !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "[DONE]" {
			break
		}
		var chunk chatStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue // ignore keep-alives / non-JSON frames
		}
		if len(chunk.Choices) == 0 {
			continue
		}
		delta := chunk.Choices[0].Delta.Content
		if delta != "" {
			sb.WriteString(delta)
			onDelta(delta)
		}
	}
	if err := sc.Err(); err != nil {
		// A context cancel surfaces here; report the stream so far isn't lost.
		return sb.String(), err
	}
	return sb.String(), nil
}

// chatEndpoint normalises a user-entered base URL to the chat-completions path.
// It accepts a bare host ("http://localhost:11434"), a "…/v1" root, or a full
// endpoint already ending in "/chat/completions".
func chatEndpoint(base string) string {
	b := strings.TrimRight(strings.TrimSpace(base), "/")
	switch {
	case strings.HasSuffix(b, "/chat/completions"):
		return b
	case strings.HasSuffix(b, "/v1"):
		return b + "/chat/completions"
	default:
		return b + "/v1/chat/completions"
	}
}

type chatStreamChunk struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content"`
		} `json:"delta"`
	} `json:"choices"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func (c chatResponse) firstContent() string {
	if len(c.Choices) == 0 {
		return ""
	}
	return c.Choices[0].Message.Content
}

// clip trims text to at most n runes, appending a marker when truncated, so a
// huge paste can't blow the model's context window.
func clip(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "\n…[truncated " + fmt.Sprintf("%d", len(r)-n) + " more characters]"
}

// evalErrMsg turns transport errors into something an operator can act on.
func evalErrMsg(err error) string {
	if errors.Is(err, context.Canceled) {
		return "canceled"
	}
	msg := err.Error()
	if strings.Contains(msg, "connection refused") {
		return "could not reach the model server — is it running at the base URL you set?"
	}
	return msg
}
