#!/bin/bash

# List of common shell configuration files
config_files=(
  ~/.bash_profile
  ~/.bashrc
  ~/.zshrc
  ~/.zprofile
  ~/.profile
  ~/.config/fish/config.fish
)

echo "Checking shell configuration files for Nix entries..."
echo "======================================================"

for file in "${config_files[@]}"; do
  if [ -f "$file" ]; then
    echo "Checking $file:"
    grep -n "nix\|NIX" "$file" || echo "  No Nix entries found."
    echo ""
  fi
done

echo "======================================================"
echo "To remove Nix entries, edit the files listed above and delete the relevant lines."
echo "After editing, restart your terminal or run 'source <filename>' to apply changes."