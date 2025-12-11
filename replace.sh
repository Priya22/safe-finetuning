#!/bin/bash
FOLDER="SafeLoRA/commands"

grep -rl "Safelora" "$FOLDER" | xargs sed -i 's|Safelora|SafeLoRA|g'
