#!/usr/bin/env bash
# Store an OpenAI API key on a benchmark host without it touching argv, shell history, or logs.
#
#   scripts/push-openai-key.sh ubuntu@HOST [SSH_IDENTITY_FILE]
#
# The key is read with hidden input, streamed over SSH stdin, and written to
# ~/.config/zeroshot-bench/openai.env (mode 0600) on the host. scripts/zsbench mounts that file
# read-only into the runner; the runner passes it to agent containers per exec, never in argv.
set -euo pipefail

host=${1:?usage: push-openai-key.sh user@host [ssh-identity-file]}
identity=${2:-}
ssh_args=(-o BatchMode=yes)
[[ -n $identity ]] && ssh_args+=(-i "$identity")

IFS= read -rsp "OpenAI API key (input hidden): " key
echo
key=${key//[[:space:]]/}
if [[ ${#key} -lt 20 ]]; then
  echo "That does not look like an API key (too short). Nothing was stored." >&2
  exit 1
fi

printf 'OPENAI_API_KEY=%s\n' "$key" | ssh "${ssh_args[@]}" "$host" '
  set -e
  umask 077
  mkdir -p ~/.config/zeroshot-bench
  rm -f ~/.config/zeroshot-bench/openai.env.tmp
  cat > ~/.config/zeroshot-bench/openai.env.tmp
  mv ~/.config/zeroshot-bench/openai.env.tmp ~/.config/zeroshot-bench/openai.env
  chmod 600 ~/.config/zeroshot-bench/openai.env
'
unset key
echo "Stored on $host at ~/.config/zeroshot-bench/openai.env (0600)."
