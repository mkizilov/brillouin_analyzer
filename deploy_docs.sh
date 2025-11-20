#!/bin/bash
# Script to deploy documentation to GitHub Pages (gh-pages branch)

echo "Building and deploying documentation to GitHub Pages..."

# Build the documentation
mkdocs build

# Deploy to gh-pages branch
mkdocs gh-deploy

echo "Documentation deployed to gh-pages branch!"
echo "Your docs will be available at: https://<your-username>.github.io/<repo-name>/"

