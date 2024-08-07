#!/bin/bash

# Check if arguments are passed to the container
if [ $# -eq 0 ]; then
  # No arguments provided open a shell
  exec bash
else
  # Arguments provided, run them as a command
  exec "$@"
fi