#!/bin/bash
# Pre-commit code review hook for Claude Code.
# Intercepts 'git commit' commands, reviews staged changes,
# and blocks the commit if issues are found.
set -e

INPUT=$(cat)
COMMAND=$(echo "$INPUT" | jq -r '.tool_input.command // empty')

# Only intercept git commit commands
if ! echo "$COMMAND" | grep -qE '^git\s+(commit|&.*commit)'; then
  exit 0
fi

# Check if there are staged changes to review
STAGED_DIFF=$(git diff --cached --stat 2>/dev/null)
if [ -z "$STAGED_DIFF" ]; then
  exit 0
fi

# Get the full diff for review
FULL_DIFF=$(git diff --cached 2>/dev/null)
CHANGED_FILES=$(git diff --cached --name-only 2>/dev/null)

# Check for common issues

ISSUES=""

# 1. Check for debug prints / breakpoints left in
if echo "$FULL_DIFF" | grep -n '^\+.*\(breakpoint()\|pdb\.set_trace\|import pdb\)' >/dev/null 2>&1; then
  ISSUES="${ISSUES}\n- DEBUG: Found breakpoint()/pdb left in code"
fi

# 2. Check for secrets / credentials
if echo "$FULL_DIFF" | grep -nE '^\+.*(password|secret|api_key|token)\s*=\s*["\x27][^"\x27]+["\x27]' >/dev/null 2>&1; then
  ISSUES="${ISSUES}\n- SECURITY: Possible hardcoded secret/credential detected"
fi

# 3. Check for .env or credential files
if echo "$CHANGED_FILES" | grep -qE '\.(env|pem|key|credentials)$'; then
  ISSUES="${ISSUES}\n- SECURITY: Committing potential secret file (.env, .pem, .key)"
fi

# 4. Check for large files (>1MB)
for f in $CHANGED_FILES; do
  if [ -f "$f" ]; then
    SIZE=$(wc -c < "$f" 2>/dev/null || echo 0)
    if [ "$SIZE" -gt 1048576 ]; then
      ISSUES="${ISSUES}\n- SIZE: Large file (>1MB): $f ($(( SIZE / 1024 ))KB)"
    fi
  fi
done

# 5. Check for TODO/FIXME/HACK additions (warn, don't block)
TODO_COUNT=$(echo "$FULL_DIFF" | grep -c '^\+.*\(TODO\|FIXME\|HACK\)' 2>/dev/null || true)

# Build review output
echo "=== Pre-Commit Review ===" >&2
echo "Files: $(echo "$CHANGED_FILES" | wc -l | tr -d ' ')" >&2
echo "$STAGED_DIFF" >&2

if [ "$TODO_COUNT" -gt 0 ]; then
  echo "Note: $TODO_COUNT new TODO/FIXME/HACK markers added" >&2
fi

if [ -n "$ISSUES" ]; then
  echo "" >&2
  echo "BLOCKED — issues found:" >&2
  echo -e "$ISSUES" >&2
  echo "" >&2

  # Output JSON to block the commit
  cat <<EOF
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Pre-commit review failed:$(echo -e "$ISSUES" | tr '\n' ' ')"
  }
}
EOF
  exit 0
fi

echo "Review passed." >&2
exit 0
