#!/bin/bash
#
# Sync weights between local machine and cluster.
#
# This script uses rsync over SSH to synchronize weight files between your
# local machine and the cluster. It's designed to work with SSH keys for
# passwordless authentication.
#
# Usage:
#   # Pull weights from cluster to local
#   ./sync_weights.sh pull user@cluster:/path/to/shared_weights ~/shared_weights
#
#   # Push weights from local to cluster
#   ./sync_weights.sh push user@cluster:/path/to/shared_weights ~/shared_weights
#
#   # Bidirectional sync (pull then push)
#   ./sync_weights.sh bidirectional user@cluster:/path/to/shared_weights ~/shared_weights
#
#   # Continuous pull every N seconds
#   ./sync_weights.sh pull user@cluster:/path/to/shared_weights ~/shared_weights 5
#
# Prerequisites:
#   1. SSH key setup: ssh-keygen -t ed25519 && ssh-copy-id user@cluster
#   2. rsync installed on both machines
#
# Default settings:
#   CLUSTER_USER: a.a.baggio
#   CLUSTER_HOST: shell.engr.wustl.edu
#   CLUSTER_PATH: /project/scratch01/compiling/a.a.baggio/PokemonRedExperiments/shared_weights
#

set -e

# Default configuration
DEFAULT_CLUSTER_USER="a.a.baggio"
DEFAULT_CLUSTER_HOST="shell.engr.wustl.edu"
DEFAULT_CLUSTER_PATH="/project/scratch01/compiling/a.a.baggio/PokemonRedExperiments/shared_weights"
DEFAULT_LOCAL_PATH="${HOME}/shared_weights"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[sync]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[sync]${NC} $1"
}

log_error() {
    echo -e "${RED}[sync]${NC} $1"
}

show_usage() {
    cat << EOF
Usage: $0 <command> [remote_path] [local_path] [interval_seconds]

Commands:
  pull          Pull weights from cluster to local
  push          Push weights from local to cluster
  bidirectional Pull then push (sync both ways)
  status        Show sync status
  setup         Interactive setup helper

Arguments:
  remote_path   Remote path in format user@host:/path (default: configured)
  local_path    Local directory path (default: ~/shared_weights)
  interval      If provided, repeat sync every N seconds

Examples:
  $0 pull
  $0 push
  $0 pull a.a.baggio@shell.engr.wustl.edu:/project/.../shared_weights ~/weights 5

Environment variables:
  SYNC_CLUSTER_USER  Override default cluster username
  SYNC_CLUSTER_HOST  Override default cluster hostname
  SYNC_CLUSTER_PATH  Override default cluster path
  SYNC_LOCAL_PATH    Override default local path
EOF
}

parse_remote() {
    local remote="$1"
    
    if [[ -z "$remote" ]]; then
        # Use defaults
        CLUSTER_USER="${SYNC_CLUSTER_USER:-$DEFAULT_CLUSTER_USER}"
        CLUSTER_HOST="${SYNC_CLUSTER_HOST:-$DEFAULT_CLUSTER_HOST}"
        CLUSTER_PATH="${SYNC_CLUSTER_PATH:-$DEFAULT_CLUSTER_PATH}"
        REMOTE="${CLUSTER_USER}@${CLUSTER_HOST}:${CLUSTER_PATH}"
    elif [[ "$remote" == *"@"* && "$remote" == *":"* ]]; then
        # Parse user@host:/path format
        REMOTE="$remote"
        CLUSTER_USER=$(echo "$remote" | cut -d'@' -f1)
        local host_path=$(echo "$remote" | cut -d'@' -f2)
        CLUSTER_HOST=$(echo "$host_path" | cut -d':' -f1)
        CLUSTER_PATH=$(echo "$host_path" | cut -d':' -f2)
    else
        log_error "Invalid remote format. Use: user@host:/path"
        exit 1
    fi
}

do_pull() {
    local remote="$1"
    local local_path="$2"
    
    log_info "Pulling from $remote to $local_path"
    
    mkdir -p "$local_path"
    
    rsync -avz --progress \
        --include='*.pt' \
        --include='*.json' \
        --exclude='*' \
        -e "ssh -o ConnectTimeout=10 -o BatchMode=yes" \
        "$remote/" "$local_path/" 2>&1 || {
            log_warn "rsync failed (cluster may be unreachable)"
            return 1
        }
    
    log_info "Pull complete"
    return 0
}

do_push() {
    local remote="$1"
    local local_path="$2"
    
    log_info "Pushing from $local_path to $remote"
    
    if [[ ! -d "$local_path" ]]; then
        log_error "Local path does not exist: $local_path"
        return 1
    fi
    
    rsync -avz --progress \
        --include='*.pt' \
        --include='*.json' \
        --exclude='*' \
        -e "ssh -o ConnectTimeout=10 -o BatchMode=yes" \
        "$local_path/" "$remote/" 2>&1 || {
            log_warn "rsync failed (cluster may be unreachable)"
            return 1
        }
    
    log_info "Push complete"
    return 0
}

do_status() {
    local remote="$1"
    local local_path="$2"
    
    echo "=== Sync Status ==="
    echo "Remote: $remote"
    echo "Local:  $local_path"
    echo ""
    
    # Check SSH connection
    log_info "Testing SSH connection..."
    if ssh -o ConnectTimeout=5 -o BatchMode=yes "${CLUSTER_USER}@${CLUSTER_HOST}" "echo 'Connected'" 2>/dev/null; then
        echo -e "${GREEN}SSH: Connected${NC}"
    else
        echo -e "${RED}SSH: Failed (check your SSH key setup)${NC}"
    fi
    echo ""
    
    # List remote files
    log_info "Remote files:"
    ssh -o ConnectTimeout=5 -o BatchMode=yes "${CLUSTER_USER}@${CLUSTER_HOST}" \
        "ls -lh ${CLUSTER_PATH}/*.pt 2>/dev/null | tail -5" 2>/dev/null || echo "  (none or unreachable)"
    echo ""
    
    # List local files
    log_info "Local files:"
    if [[ -d "$local_path" ]]; then
        ls -lh "$local_path"/*.pt 2>/dev/null | tail -5 || echo "  (none)"
    else
        echo "  (directory does not exist)"
    fi
}

do_setup() {
    echo "=== Sync Setup Helper ==="
    echo ""
    echo "This will help you set up SSH key authentication for passwordless sync."
    echo ""
    
    # Check for existing SSH key
    if [[ -f ~/.ssh/id_ed25519 ]]; then
        log_info "SSH key already exists: ~/.ssh/id_ed25519"
    elif [[ -f ~/.ssh/id_rsa ]]; then
        log_info "SSH key already exists: ~/.ssh/id_rsa"
    else
        log_warn "No SSH key found. Creating one..."
        ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" -C "pokemon_rl_sync"
        log_info "SSH key created: ~/.ssh/id_ed25519"
    fi
    echo ""
    
    # Copy key to cluster
    log_info "To copy your SSH key to the cluster, run:"
    echo ""
    echo "  ssh-copy-id ${CLUSTER_USER}@${CLUSTER_HOST}"
    echo ""
    echo "You will be prompted for your cluster password once."
    echo "After that, passwordless SSH should work."
    echo ""
    
    # Test connection
    read -p "Would you like to test the connection now? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log_info "Testing SSH connection..."
        if ssh -o ConnectTimeout=10 -o BatchMode=yes "${CLUSTER_USER}@${CLUSTER_HOST}" "echo 'Success! SSH key authentication is working.'" 2>/dev/null; then
            echo -e "${GREEN}Connection successful!${NC}"
        else
            log_warn "Connection failed. You may need to run ssh-copy-id first."
        fi
    fi
}

# Main
COMMAND="${1:-}"
REMOTE_ARG="${2:-}"
LOCAL_ARG="${3:-${SYNC_LOCAL_PATH:-$DEFAULT_LOCAL_PATH}}"
INTERVAL="${4:-}"

if [[ -z "$COMMAND" ]]; then
    show_usage
    exit 1
fi

parse_remote "$REMOTE_ARG"
LOCAL_PATH="$LOCAL_ARG"

case "$COMMAND" in
    pull)
        if [[ -n "$INTERVAL" ]]; then
            log_info "Starting continuous pull (every ${INTERVAL}s). Press Ctrl+C to stop."
            while true; do
                do_pull "$REMOTE" "$LOCAL_PATH" || true
                sleep "$INTERVAL"
            done
        else
            do_pull "$REMOTE" "$LOCAL_PATH"
        fi
        ;;
    push)
        if [[ -n "$INTERVAL" ]]; then
            log_info "Starting continuous push (every ${INTERVAL}s). Press Ctrl+C to stop."
            while true; do
                do_push "$REMOTE" "$LOCAL_PATH" || true
                sleep "$INTERVAL"
            done
        else
            do_push "$REMOTE" "$LOCAL_PATH"
        fi
        ;;
    bidirectional)
        if [[ -n "$INTERVAL" ]]; then
            log_info "Starting continuous bidirectional sync (every ${INTERVAL}s). Press Ctrl+C to stop."
            while true; do
                do_pull "$REMOTE" "$LOCAL_PATH" || true
                do_push "$REMOTE" "$LOCAL_PATH" || true
                sleep "$INTERVAL"
            done
        else
            do_pull "$REMOTE" "$LOCAL_PATH"
            do_push "$REMOTE" "$LOCAL_PATH"
        fi
        ;;
    status)
        do_status "$REMOTE" "$LOCAL_PATH"
        ;;
    setup)
        do_setup
        ;;
    *)
        log_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac

