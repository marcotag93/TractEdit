#!/usr/bin/env bash
set -euo pipefail

# Use the runner's Ubuntu mirrors without querying preinstalled vendor feeds.
sources=/etc/apt/sources.list.d/ubuntu.sources
if [[ ! -s "$sources" ]]; then
    sources=/etc/apt/sources.list
fi
if [[ ! -s "$sources" ]]; then
    echo '::error::Ubuntu APT sources file is missing or empty'
    exit 1
fi

apt_options=(
    -o "Dir::Etc::sourcelist=$sources"
    -o Dir::Etc::sourceparts=/dev/null
    -o Acquire::Retries=3
)
sudo apt-get "${apt_options[@]}" --no-list-cleanup -o APT::Update::Error-Mode=any update
sudo apt-get "${apt_options[@]}" install -y "$@"
