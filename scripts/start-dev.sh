#!/bin/bash
# Load nvm so npm/node are available
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"

cd /Users/osilkin/Programming/learning/llm-curriculum
exec npm run dev -- --host 2>&1
